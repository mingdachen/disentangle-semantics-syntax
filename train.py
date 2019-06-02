import torch
import config
import train_helper
import data_utils

from models import vgvae
from tensorboardX import SummaryWriter
from config import EVAL_YEAR

best_dev_res = test_bm_res = test_avg_res = 0


def run(e):
    global best_dev_res, test_bm_res, test_avg_res

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    dp = data_utils.data_processor(
        train_path=e.config.train_file,
        eval_path=e.config.eval_file,
        experiment=e)
    data, W = dp.process()

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model = vgvae(
        vocab_size=len(data.vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)

    start_epoch = true_it = 0
    if e.config.resume:
        start_epoch, _, best_dev_res, test_avg_res = \
            model.load(name="latest")
        if e.config.use_cuda:
            model.cuda()
            e.log.info("transferred model to gpu")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}, best dev res: {:.3f}, test avg res: {:.3f}"
            .format(start_epoch, true_it, best_dev_res, test_avg_res))

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.summarize:
        writer = SummaryWriter(e.experiment_dir)

    if e.config.decoder_type.startswith("bag"):
        minibatcher = data_utils.bow_minibatcher
        e.log.info("using BOW batcher")
    else:
        minibatcher = data_utils.minibatcher
        e.log.info("using sequential batcher")

    train_batch = minibatcher(
        data1=data.train_data[0],
        data2=data.train_data[1],
        vocab_size=len(data.vocab),
        batch_size=e.config.batch_size,
        score_func=model.score,
        shuffle=True,
        mega_batch=0 if not e.config.resume else e.config.mb,
        p_scramble=e.config.ps)

    evaluator = train_helper.evaluator(model, e)

    e.log.info("Training start ...")
    train_stats = train_helper.tracker(["loss", "vmf_kl", "gauss_kl",
                                        "rec_logloss", "para_logloss",
                                        "wploss", "dp_loss"])

    for epoch in range(start_epoch, e.config.n_epoch):
        if epoch:
            train_batch.mega_batch = e.config.mb
        e.log.info("current mega batch: {}".format(train_batch.mega_batch))
        for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                 n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in \
                enumerate(train_batch):
            true_it = it + 1 + epoch * len(train_batch)

            loss, vkl, gkl, rec_logloss, para_logloss, wploss, dploss = \
                model(s1, m1, s2, m2, t1, tm1, t2, tm2,
                      n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2,
                      e.config.vmkl, e.config.gmkl,
                      epoch > 1 and e.config.dratio and e.config.mb > 1)
            model.optimize(loss)

            train_stats.update(
                {"loss": loss, "vmf_kl": vkl, "gauss_kl": gkl,
                 "para_logloss": para_logloss, "rec_logloss": rec_logloss,
                 "wploss": wploss, "dp_loss": dploss},
                len(s1))

            if (true_it + 1) % e.config.print_every == 0 or \
                    (true_it + 1) % len(train_batch) == 0:
                summarization = train_stats.summarize(
                    "epoch: {}, it: {} (max: {}), kl_temp: {:.2E}|{:.2E}"
                    .format(epoch, it, len(train_batch),
                            e.config.vmkl, e.config.gmkl))
                e.log.info(summarization)
                if e.config.summarize:
                    for name, value in train_stats.stats.items():
                        writer.add_scalar(
                            "train/" + name, value, true_it)
                train_stats.reset()

            if (true_it + 1) % e.config.eval_every == 0 or \
                    (true_it + 1) % len(train_batch) == 0:

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                dev_stats, _, dev_res, _ = evaluator.evaluate(
                    data.dev_data, 'pred')

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                if e.config.summarize:
                    writer.add_scalar(
                        "dev/pearson", dev_stats[EVAL_YEAR][1], true_it)
                    writer.add_scalar(
                        "dev/spearman", dev_stats[EVAL_YEAR][2], true_it)

                if best_dev_res < dev_res:
                    best_dev_res = dev_res

                    e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)

                    test_stats, test_bm_res, test_avg_res, test_avg_s = \
                        evaluator.evaluate(data.test_data, 'pred')

                    e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)
                    e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)

                    tz_stats, tz_bm_res, tz_avg_res, tz_avg_s = \
                        evaluator.evaluate(data.test_data, 'predz')
                    e.log.info("Summary - benchmark: {:.4f}, test avg: {:.4f}"
                               .format(tz_bm_res, tz_avg_res))
                    e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)

                    model.save(
                        dev_avg=best_dev_res,
                        dev_perf=dev_stats,
                        test_avg=test_avg_res,
                        test_perf=test_stats,
                        iteration=true_it,
                        epoch=epoch)

                    if e.config.summarize:
                        for year, stats in test_stats.items():
                            writer.add_scalar(
                                "test/{}_pearson".format(year),
                                stats[1], true_it)
                            writer.add_scalar(
                                "test/{}_spearman".format(year),
                                stats[2], true_it)

                        writer.add_scalar(
                            "test/avg_pearson", test_avg_res, true_it)
                        writer.add_scalar(
                            "test/avg_spearman", test_avg_s, true_it)
                        writer.add_scalar(
                            "test/STSBenchmark_pearson", test_bm_res, true_it)
                        writer.add_scalar(
                            "dev/best_pearson", best_dev_res, true_it)

                        writer.add_scalar(
                            "testz/avg_pearson", tz_avg_res, true_it)
                        writer.add_scalar(
                            "testz/avg_spearman", tz_avg_s, true_it)
                        writer.add_scalar(
                            "testz/STSBenchmark_pearson", tz_bm_res, true_it)
                train_stats.reset()
                e.log.info("best dev result: {:.4f}, "
                           "STSBenchmark result: {:.4f}, "
                           "test average result: {:.4f}"
                           .format(best_dev_res, test_bm_res, test_avg_res))

        model.save(
            dev_avg=best_dev_res,
            dev_perf=dev_stats,
            test_avg=test_avg_res,
            test_perf=test_stats,
            iteration=true_it,
            epoch=epoch + 1,
            name="latest")

    e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)

    test_stats, test_bm_res, test_avg_res, test_avg_s = \
        evaluator.evaluate(data.test_data, 'pred')

    e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)
    e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)

    tz_stats, tz_bm_res, tz_avg_res, tz_avg_s = \
        evaluator.evaluate(data.test_data, 'predz')
    e.log.info("Summary - benchmark: {:.4f}, test avg: {:.4f}"
               .format(tz_bm_res, tz_avg_res))
    e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)


if __name__ == '__main__':

    args = config.get_base_parser().parse_args()
    args.use_cuda = torch.cuda.is_available()

    def exit_handler(*args):
        print(args)
        print("best dev result: {:.4f}, "
              "STSBenchmark result: {:.4f}, "
              "test average result: {:.4f}"
              .format(best_dev_res, test_bm_res, test_avg_res))
        exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.experiment(args, args.save_prefix) as e:

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)
