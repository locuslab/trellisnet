## Character-level Language Modeling on Penn Treebank (PTB)

This folder contains the code for processing and training a (TrellisNet-based) language model on the character-level PTB
corpus. When used for character-level language modeling, PTB is a medium-sized corpus with longer dependency (than in
word-level) and an alphabet size of 50. Due to the small alphabet size, techniques such as mixture of softmax (Yang et al. 2018)
does not work on character-level language modeling tasks.

In addition, note that the `<eos>` tag in PTB (which marks the end of a sentence) is now considered as **one** character.


#### Usage
If you don't have the PTB dataset, simply running the program (see below) will automatically download the dataset for you using the [observations](https://github.com/edwardlib/observations) package (and then process it for you).

After downloading the dataset, you can start training the model by
```python
python char_ptb.py
```
You can also use the `-h` flag to set the arguments or change the hyperparameters.