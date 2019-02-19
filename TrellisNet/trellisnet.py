import torch
from torch import nn
import torch.nn.functional as F
from optimizations import weight_norm, VariationalDropout, VariationalHidDropout

__author__ = 'shaojieb'


class WeightShareConv1d(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, kernel_size, dropouth=0.0):
        """
        The weight-tied 1D convolution used in TrellisNet.

        :param input_dim: The dim of original input
        :param hidden_dim: The dim of hidden input
        :param n_out: The dim of the pre-activation (i.e. convolutional) output
        :param kernel_size: The size of the convolutional kernel
        :param dropouth: Hidden-to-hidden dropout
        """
        super(WeightShareConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.n_inp1 = input_dim

        conv1 = nn.Conv1d(input_dim, n_out, kernel_size)
        self.weight1 = conv1.weight

        conv2 = nn.Conv1d(hidden_dim, n_out, kernel_size)
        self.weight2 = conv2.weight
        self.bias2 = conv2.bias
        self.init_weights()

        self.dict = dict()
        self.drop = VariationalHidDropout(dropout=dropouth)

    def init_weights(self):
        bound = 0.01
        self.weight1.data.normal_(0, bound)
        self.weight2.data.normal_(0, bound)
        self.bias2.data.normal_(0, bound)

    def forward(self, input, dilation, hid=None):
        k = self.kernel_size
        padding = (k - 1) * dilation    # To maintain causality constraint
        x = F.pad(input, (padding, 0))

        # Input part
        x_1 = x[:, :self.n_inp1]

        # Hidden part
        z_1 = x[:, self.n_inp1:]
        z_1[:, :, :padding] = hid.repeat(1, 1, padding)  # Note: we only pad the hidden part :-)
        device = x_1.get_device()

        # A linear transformation of the input sequence (and pre-computed once)
        if (dilation, device) not in self.dict or self.dict[(dilation, device)] is None:
            self.dict[(dilation, device)] = F.conv1d(x_1, self.weight1, dilation=dilation)

        # Input injection
        return self.dict[(dilation, device)] + F.conv1d(self.drop(z_1), self.weight2, self.bias2, dilation=dilation)


class TrellisNet(nn.Module):
    def __init__(self, ninp, nhid, nout, nlevels=40, kernel_size=2, dropouth=0.0,
                 wnorm=True, aux_frequency=20, dilation=[1]):
        """
        Build a trellis network with LSTM-style gated activations

        :param ninp: The input (e.g., embedding) dimension
        :param nhid: The hidden unit dimension (excluding the output dimension). In other words, if you want to build
                     a TrellisNet with hidden size 1000 and output size 400, you should set nhid = 1000-400 = 600.
                     (The reason we want to separate this is from Theorem 1.)
        :param nout: The output dimension
        :param nlevels: Number of layers
        :param kernel_size: Kernel size of the TrellisNet
        :param dropouth: Hidden-to-hidden (VD-based) dropout rate
        :param wnorm: A boolean indicating whether to use weight normalization
        :param aux_frequency: Frequency of taking the auxiliary loss; (-1 means no auxiliary loss)
        :param dilation: The dilation of the convolution operation in TrellisNet
        """
        super(TrellisNet, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.h_size = h_size = nhid + nout
        self.dilation = dilation
        self.nlevels = nlevels
        self.fn = None
        self.last_output = None
        self.aux_frequency = aux_frequency

        self.kernel_size = ker = kernel_size

        if wnorm:
            print("Weight normalization applied")
            self.full_conv, self.fn = weight_norm(
                WeightShareConv1d(ninp, h_size, 4 * h_size, kernel_size=kernel_size, dropouth=dropouth),
                names=['weight1', 'weight2'],
                dim=0)           # The weights to be normalized
        else:
            self.full_conv = WeightShareConv1d(ninp, h_size, 4 * h_size, kernel_size=ker, dropouth=dropouth)

    def transform_input(self, X):
        # X has dimension (N, ninp, L)
        batch_size = X.size(0)
        seq_len = X.size(2)
        h_size = self.h_size

        self.ht = torch.zeros(batch_size, h_size, seq_len).cuda()
        self.ct = torch.zeros(batch_size, h_size, seq_len).cuda()
        return torch.cat([X] + [self.ht], dim=1)     # "Injecting" input sequence at layer 1

    def step(self, Z, dilation=1, hc=None):
        ninp = self.ninp
        h_size = self.h_size
        (hid, cell) = hc

        # Apply convolution
        out = self.full_conv(Z, dilation=dilation, hid=hid)

        # Gated activations among channel groups
        ct_1 = F.pad(self.ct, (dilation, 0))[:, :, :-dilation]  # Dimension (N, h_size, L)
        ct_1[:, :, :dilation] = cell.repeat(1, 1, dilation)

        it = torch.sigmoid(out[:, :h_size])
        ot = torch.sigmoid(out[:, h_size: 2 * h_size])
        gt = torch.tanh(out[:, 2 * h_size: 3 * h_size])
        ft = torch.sigmoid(out[:, 3 * h_size: 4 * h_size])
        ct = ft * ct_1 + it * gt
        ht = ot * torch.tanh(ct)

        # Put everything back to form Z (i.e., injecting input to hidden unit)
        Z = torch.cat([Z[:, :ninp], ht], dim=1)
        self.ct = ct
        return Z

    def forward(self, X, hc, aux=True):
        ninp = self.ninp
        nout = self.nout
        Z = self.transform_input(X)
        aux_outs = []
        dilation_cycle = self.dilation

        if self.fn is not None:
            # Recompute weight normalization weights
            self.fn.reset(self.full_conv)
        for key in self.full_conv.dict:
            # Clear the pre-computed computations
            if key[1] == X.get_device():
                self.full_conv.dict[key] = None
        self.full_conv.drop.reset_mask(Z[:, ninp:])

        # Feed-forward layers
        for i in range(0, self.nlevels):
            d = dilation_cycle[i % len(dilation_cycle)]
            Z = self.step(Z, dilation=d, hc=hc)
            if aux and i % self.aux_frequency == (self.aux_frequency-1):
                aux_outs.append(Z[:, -nout:].unsqueeze(0))

        out = Z[:, -nout:].transpose(1, 2)              # Dimension (N, L, nout)
        hc = (Z[:, ninp:, -1:], self.ct[:, :, -1:])     # Dimension (N, h_size, L)
        if aux:
            aux_outs = torch.cat(aux_outs, dim=0).transpose(0, 1).transpose(2, 3)
        else:
            aux_outs = None
        return out, hc, aux_outs
