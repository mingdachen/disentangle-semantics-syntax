# Disentangling Syntax and Semantics in Sentence Representations
A PyTorch implementation of "[A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations](https://ttic.uchicago.edu/~mchen/papers/mchen+etal.naacl19.pdf)" (NAACL 2019).

#### 2019/06/02 Script for evaluating tree edit distance will be released soon

## Dependencies

- Python 3.5
- PyTorch 0.3
- NumPy
- NLTK (for syntactic evaluation)
- [zss](https://github.com/timtadh/zhang-shasha) (for computing tree edit distance)


## Download Data

[Training and semantic evaluation data (processed)](https://drive.google.com/drive/folders/1i8cMh7E0TnbrDEw_s9W8LDP_TYNkp4ti?usp=sharing)

[Syntactic evaluataion (based on ParaNMT)](https://drive.google.com/drive/folders/1oVjn_3xIDZbkRm50fSHDZ5nKZtJ_BFyD?usp=sharing)

## Run

``run_vgvae.sh`` is provided as an example for training new models

## Evaluation

#### Labeled F1 and Tagging accuracy
``python eval_f1_acc.py -s PATH_TO_MODEL_PICKLE -v VOCAB_PICKLE -d SYNTACTIC_EVAL_DIR``

## Reference

```
@inproceedings{mchen-multitask-19,
  author    = {Mingda Chen and Qingming Tang and Sam Wiseman and Kevin Gimpel},
  title     = {A Multi-Task Approach for Disentangling Syntax and Semantics in Sentence Representations},
  booktitle = {Proc. of {NAACL}},
  year      = {2019}
}
```
