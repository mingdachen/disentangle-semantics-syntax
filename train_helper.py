import time
import logging
import argparse
import os
import data_utils

import numpy as np

from config import get_base_parser
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from decorators import auto_init_args

BOS = 1
EOS = 2


def register_exit_handler(exit_handler):
    import atexit
    import signal

    atexit.register(exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT, exit_handler)


def get_kl_anneal_function(anneal_function, max_value, slope):
    if anneal_function.lower() == 'exp':
        return lambda step, curr_value: \
            min(max_value, float(1 / (1 + np.exp(-slope * step + 100))))
    elif anneal_function.lower() == 'linear':
        return lambda step, curr_value: \
            min(max_value, curr_value + slope * max_value / (step + 100))
    elif anneal_function.lower() == 'linear2':
        return lambda step, curr_value: min(max_value, slope * step)
    else:
        raise ValueError("invalid anneal function: {}".format(anneal_function))


class tracker:
    @auto_init_args
    def __init__(self, names):
        assert len(names) > 0
        self.reset()

    def __getitem__(self, name):
        return self.values.get(name, 0) / self.counter if self.counter else 0

    def __len__(self):
        return len(self.names)

    def reset(self):
        self.values = dict({name: 0. for name in self.names})
        self.counter = 0
        self.create_time = time.time()

    def update(self, named_values, count):
        """
        named_values: dictionary with each item as name: value
        """
        self.counter += count
        for name, value in named_values.items():
            self.values[name] += value.data.cpu().numpy()[0] * count

    def summarize(self, output=""):
        if output:
            output += ", "
        for name in self.names:
            output += "{}: {:.3f}, ".format(
                name, self.values[name] / self.counter if self.counter else 0)
        output += "elapsed time: {:.1f}(s)".format(
            time.time() - self.create_time)
        return output

    @property
    def stats(self):
        return {n: v / self.counter if self.counter else 0
                for n, v in self.values.items()}


class experiment:
    @auto_init_args
    def __init__(self, config, experiments_prefix, logfile_name="log"):
        """Create a new Experiment instance.

        Modified based on: https://github.com/ex4sperans/mag

        Args:
            logfile_name: str, naming for log file. This can be useful to
                separate logs for different runs on the same experiment
            experiments_prefix: str, a prefix to the path where
                experiment will be saved
        """

        # get all defaults
        all_defaults = {}
        for key in vars(config):
            all_defaults[key] = get_base_parser().get_default(key)

        self.default_config = all_defaults

        config.resume = False
        if not config.debug:
            if os.path.isdir(self.experiment_dir):
                print("log exists: {}".format(self.experiment_dir))
                config.resume = True

            print(config)
            self._makedir()

        self._make_misc_dir()

    def _makedir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _make_misc_dir(self):
        os.makedirs(self.config.vocab_file, exist_ok=True)

    @property
    def experiment_dir(self):
        if self.config.debug:
            return "./"
        else:
            # get namespace for each group of args
            arg_g = dict()
            for group in get_base_parser()._action_groups:
                group_d = {a.dest: self.default_config.get(a.dest, None)
                           for a in group._group_actions}
                arg_g[group.title] = argparse.Namespace(**group_d)

            # skip default value
            identifier = ""
            for key, value in sorted(vars(arg_g["model_configs"]).items()):
                if getattr(self.config, key) != value:
                    identifier += key + str(getattr(self.config, key))
            return os.path.join(self.experiments_prefix, identifier)

    @property
    def log_file(self):
        return os.path.join(self.experiment_dir, self.logfile_name)

    def register_directory(self, dirname):
        directory = os.path.join(self.experiment_dir, dirname)
        os.makedirs(directory, exist_ok=True)
        setattr(self, dirname, directory)

    def _register_existing_directories(self):
        for item in os.listdir(self.experiment_dir):
            fullpath = os.path.join(self.experiment_dir, item)
            if os.path.isdir(fullpath):
                setattr(self, item, fullpath)

    def __enter__(self):

        if self.config.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')
        else:
            print("log saving to", self.log_file)
            logging.basicConfig(
                filename=self.log_file,
                filemode='a+', level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')

        self.log = logging.getLogger()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        logging.shutdown()

    @property
    def elapsed_time(self):
        return (time.time() - self.start_time) / 3600


class evaluator:
    @auto_init_args
    def __init__(self, model, experiment):
        self.expe = experiment

    def evaluate(self, all_data, func_name, verbose=False):
        self.model.eval()
        stats = dict()
        stsbm = None
        tot_domain = tot_pearson = tot_spear = 0
        for y, data in sorted(all_data.items()):
            n_pred = 0
            p_score = []
            s_score = []
            for name, d in data.items():
                preds = []
                golds = d[2]
                for s1, m1, s2, m2, _, _, _, _, \
                        _, _, _, _, _, _, _, _, _ in \
                        data_utils.minibatcher(
                            data1=d[0],
                            data2=d[1],
                            batch_size=100,
                            score_func=None,
                            shuffle=False,
                            mega_batch=0,
                            p_scramble=0.):
                    scores = getattr(self.model, func_name)(s1, m1, s2, m2)
                    preds.extend(scores.tolist())
                assert len(golds) == len(preds)
                n_pred += len(preds)
                tot_domain += 1
                p_score.append(pearsonr(preds, golds)[0])
                s_score.append(spearmanr(preds, golds)[0])
                if verbose:
                    self.expe.log.info(
                        "YEAR: {}, #Preds: {}, Domain: {}, "
                        "Pearson: {:.4f}, Spearman: {:.4f}"
                        .format(y, len(preds), name, p_score[-1], s_score[-1]))
            stats[y] = (len(s_score), sum(p_score) / len(p_score),
                        sum(s_score) / len(s_score))
            if y == "STSBenchmark":
                assert len(p_score) == 1
                tot_domain -= 1
                stsbm = sum(p_score)
            else:
                tot_pearson += sum(p_score)
                tot_spear += sum(s_score)
            self.expe.log.info(
                "YEAR: {}, #Preds: {}, #Domain: {}, "
                "Pearson: {:.4f}, Spearman: {:.4f}"
                .format(y, n_pred, *stats[y]))
        return stats, stsbm, tot_pearson / tot_domain, tot_spear / tot_domain
