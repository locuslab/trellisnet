import torch.nn.functional as F
import sys
from torch import nn

sys.path.append('../')
from trellisnet import TrellisNet
from optimizations import WeightDrop


class TrellisNetModel(nn.Module):
    def __init__(self, ninp, nhid, nout, nlevels, kernel_size=2, dilation=[1],
                 dropout=0.0, dropouti=0.0, dropouth=0.0, wdrop=0.0,
                 temporalwdrop=True, wnorm=True, aux=False, aux_frequency=1e4):
        """
        A sequence model using TrellisNet (on sequential MNIST & CIFAR-10). Note that this is different from
        the models in other tasks (e.g. word-level PTB) because: 1) there is no more embedding; 2) we only need
        one output at the end for classification of the pixel stream; and 3) the input and output features are
        very low-dimensional (e.g., 3 channels).

        :param ninp: The number of input channels of the pixels
        :param nhid: The number of hidden units in TrellisNet (excluding the output size)
        :param nout: The number of output channels (which should agree with the number of classes)
        :param nlevels: The number of TrellisNet layers
        :param kernel_size: Kernel size of the TrellisNet
        :param dilation: Dilation size of the TrellisNet
        :param dropout: Output dropout
        :param dropouti: Input dropout
        :param dropouth: Hidden-to-hidden (VD-based) dropout
        :param wdrop: Weight dropout
        :param temporalwdrop: Whether we drop only the temporal parts of the weight (only valid if wdrop > 0)
        :param wnorm: Whether to apply weight normalization
        :param aux: Whether to use auxiliary loss (deep supervision)
        :param aux_frequency: The frequency of the auxiliary loss (only valid if aux == True)
        """
        super(TrellisNetModel, self).__init__()
        self.nout = nout    # Should be the number of classes
        self.nhid = nhid
        self.dropout = dropout
        self.dropouti = dropouti
        self.aux = aux

        network = TrellisNet
        self.network = network(ninp, nhid, nout=nout, nlevels=nlevels, kernel_size=kernel_size,
                               dropouth=dropouth, wnorm=wnorm, aux_frequency=aux_frequency, dilation=dilation)

        reg_term = '_v' if wnorm else ''
        self.network = WeightDrop(self.network,
                                  [['full_conv', 'weight1' + reg_term],
                                   ['full_conv', 'weight2' + reg_term]],
                                  dropout=wdrop, temporal=temporalwdrop)

        # If weight normalization is used, we apply the weight dropout to its "direction", instead of "scale"
        self.linear = nn.Linear(nout, nout)
        self.network = nn.ModuleList([self.network])

    def forward(self, inputs, hidden):
        inputs = F.dropout(inputs, self.dropouti)   # Inputs are very low-dimensional, so just use F.dropout
        raw_output, hidden, all_raw_outputs = self.network[0](inputs, hidden, aux=self.aux)
        raw_output = raw_output.transpose(1, 2)                           # Dimension (N, C, L)

        # Note: although we process it here, in the current implementation we don't use the auxiliary loss.
        if all_raw_outputs is not None:
            all_raw_outputs = all_raw_outputs.transpose(2, 3)

        out = F.dropout(self.linear(raw_output[:, :, -1]), self.dropout)    # Dimension (N, n_classes)
        return F.log_softmax(out, dim=1), hidden

    def init_hidden(self, bsz):
        h_size = self.nhid + self.nout
        weight = next(self.parameters()).data
        return (weight.new(bsz, h_size, 1).zero_(),
                weight.new(bsz, h_size, 1).zero_())