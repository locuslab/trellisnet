## Synthetic Long-Term Dependency Tests - Sequential MNIST, Permuted MNIST, Sequential CIFAR-10

This folder contains the code for processing and training a (TrellisNet-based) sequence model on three synthetic
tests on evaluating the longer-term dependencies of a sequence model. These tasks include sequential MNIST,
sequential CIFAR-10, and permuted MNIST.

#### Usage
If you have not downloaded the datasets before, we use PyTorch to download them for you (and perform the necessary processing, such as normalization).

After downloading the dataset, you can start training the model by
```python
python seq_mnist_cifar.py
```
You can use the `-h` flag to set the arguments.