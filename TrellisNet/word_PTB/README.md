## Word-level Language Modeling on Penn Treebank (PTB)

This folder contains the code for processing and training a (TrellisNet-based) language model on the word-level PTB
corpus. Besides the typical PTB training setting (softmax, as in [AWD-LSTM-LM](https://github.com/salesforce/awd-lstm-lm)),
our implementation also includes a mixture of softmax (Yang et al. 2018) version (to run MoS, you will need to specify parameters like `dropoutl` and `n_expert`; see our hyperparameter table in the paper for reference).

#### Usage
You can download the data using [observations](https://github.com/edwardlib/observations) package (or directly get PTB from the source). Then, move the training, validation and testing data to `data/penn`, with names `train.txt`, `valid.txt`, and `test.txt` respectivelly.

After downloading the dataset, you can start training the model by
```python
python word_ptb.py
```
You can also use the `-h` flag to set the arguments or change the hyperparameters.