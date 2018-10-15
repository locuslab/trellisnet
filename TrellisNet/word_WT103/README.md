## Word-level Language Modeling on WikiText-103 (WT103)

This folder contains the code for processing and training a (TrellisNet-based) language model on the word-level WikiText-103
corpus. Compared to PTB, this is a much larger-scale dataset (>100 times larger) with more vocabularies as well. While PTB
tests the regularization and generalization of a model on a relatively constrained dataset, WT103 is more representative
and realistic (e.g., all numbers in PTB are replaced with `N`, which is not the case for WT103). Meanwhile, as we have
a large embedding now, many techniques that work well on small datasets are intractable due to memory problem.

#### Usage
You can download the dataset [here](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip). After unzipping the dataset, you should find three files: `wiki.{train, valid, test}.tokens`, which you should rename to `train.txt`, `valid.txt` and `test.txt`. Make sure these three files are in `data/wikitext103/`.

After downloading the dataset, you can start training the model by
```python
python word_wt103.py
```
You can also use the `-h` flag to set the arguments or change the hyperparameters. The training will not fit easily to one GPU (due to memory issue), so a multigpu environment is highly recommended.