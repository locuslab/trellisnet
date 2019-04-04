import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle

from trellisnet import TrellisNet
from optimizations import WeightDrop, embedded_dropout, VariationalDropout


class MixSoftmax(nn.Module):
    def __init__(self, n_components, n_classes, nlasthid, ninp, decoder, dropoutl):
        """
        Apply mixture of softmax on the last layer of the network

        :param n_components: The number of softmaxes to use
        :param n_classes: The number of classes to predict
        :param nlasthid: The dimension of the last hidden layer from the deep network
        :param ninp: The embedding size
        :param decoder: The decoder layer
        :param dropoutl: The dropout to be applied on the pre-softmax output
        """
        super(MixSoftmax, self).__init__()
        self.n_components = n_components
        self.n_classes = n_classes
        self.prior = nn.Linear(nlasthid, n_components)  # C ---> m
        self.latent = nn.Linear(nlasthid, n_components * ninp)  # C ---> m*C
        self.decoder = decoder
        self.var_drop = VariationalDropout()
        self.ninp = ninp
        self.nlasthid = nlasthid
        self.dropoutl = dropoutl

    def init_weights(self):
        initrange = 0.1
        self.prior.weight.data.uniform_(-initrange, initrange)
        self.latent.weight.data.uniform_(-initrange, initrange)

    def forward(self, context):
        n_components = self.n_components
        n_classes = self.n_classes
        decoder = self.decoder
        ninp = self.ninp
        dim = len(context.size())

        if dim == 4:
            # context: (N, M, L, C)  (used for the auxiliary outputs)
            batch_size = context.size(0)
            aux_size = context.size(1)
            seq_len = context.size(2)
            priors = F.softmax(self.prior(context), dim=3).view(-1, n_components)  # (M*N*L, m)
            latent = self.var_drop(self.latent(context), self.dropoutl, dim=4)
            latent = F.softmax(decoder(F.tanh(latent.view(-1, n_components, ninp))), dim=2)
            return (priors.unsqueeze(2).expand_as(latent) * latent).sum(1).view(batch_size, aux_size, seq_len,
                                                                                n_classes)
        else:
            batch_size = context.size(0)
            seq_len = context.size(1)
            priors = F.softmax(self.prior(context), dim=2).view(-1, n_components)  # (N*L, m)
            latent = self.var_drop(self.latent(context), self.dropoutl)
            latent = F.softmax(decoder(F.tanh(latent.view(-1, n_components, ninp))), dim=2)  # (N*L, m, n_classes)
            return (priors.unsqueeze(2).expand_as(latent) * latent).sum(1).view(batch_size, seq_len, n_classes)


class TrellisNetModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nout, nlevels, kernel_size=2, dilation=[1],
                 dropout=0.0, dropouti=0.0, dropouth=0.0, dropoutl=0.0, emb_dropout=0.0, wdrop=0.0,
                 temporalwdrop=True, tie_weights=True, repack=False, wnorm=True, aux=True, aux_frequency=20, n_experts=0,
                 load=""):
        """
        A deep sequence model based on TrellisNet

        :param ntoken: The number of unique tokens
        :param ninp: The input dimension
        :param nhid: The hidden unit dimension (excluding the output dimension). In other words, if you want to build
                     a TrellisNet with hidden size 1000 and output size 400, you should set nhid = 1000-400 = 600.
                     (The reason we want to separate this is from Theorem 1.)
        :param nout: The output dimension
        :param nlevels: The number of TrellisNet layers
        :param kernel_size: Kernel size of the TrellisNet
        :param dilation: Dilation size of the TrellisNet
        :param dropout: Output (variational) dropout
        :param dropouti: Input (variational) dropout
        :param dropouth: Hidden-to-hidden (VD-based) dropout
        :param dropoutl: Mixture-of-Softmax dropout (only valid if MoS is used)
        :param emb_dropout: Embedding dropout
        :param wdrop: Weight dropout
        :param temporalwdrop: Whether we drop only the temporal parts of the weight (only valid if wdrop > 0)
        :param tie_weights: Whether to tie the encoder and decoder weights
        :param repack: Whether to use history repackaging for TrellisNet
        :param wnorm: Whether to apply weight normalization
        :param aux: Whether to use auxiliary loss (deep supervision)
        :param aux_frequency: The frequency of the auxiliary loss (only valid if aux == True)
        :param n_experts: The number of softmax "experts" (i.e., whether MoS is used)
        :param load: The path to the pickled weight file (the weights/biases should be in numpy format)
        """
        super(TrellisNetModel, self).__init__()
        self.emb_dropout = emb_dropout
        self.dropout = dropout  # Rate for dropping eventual output
        self.dropouti = dropouti  # Rate for dropping embedding output
        self.dropoutl = dropoutl
        self.var_drop = VariationalDropout()
        
        self.repack = repack
        self.nout = nout
        self.nhid = nhid
        self.ninp = ninp
        self.aux = aux
        self.n_experts = n_experts
        self.tie_weights = tie_weights
        self.wnorm = wnorm
        
        # 1) Set up encoder and decoder (embeddings)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        if tie_weights:
            if nout != ninp and self.n_experts == 0:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        # 2) Set up TrellisNet
        tnet = TrellisNet
        self.tnet = tnet(ninp, nhid, nout=nout, nlevels=nlevels, kernel_size=kernel_size,
                         dropouth=dropouth, wnorm=wnorm, aux_frequency=aux_frequency, dilation=dilation)
        
        # 3) Set up MoS, if needed
        if n_experts > 0:
            print("Applied Mixture of Softmax")
            self.mixsoft = MixSoftmax(n_experts, ntoken, nlasthid=nout, ninp=ninp, decoder=self.decoder,
                                      dropoutl=dropoutl)
            
        # 4) Apply weight drop connect. If weightnorm is used, we apply the dropout to its "direction", instead of "scale"
        reg_term = '_v' if wnorm else ''
        self.tnet = WeightDrop(self.tnet,
                               [['full_conv', 'weight1' + reg_term],
                                ['full_conv', 'weight2' + reg_term]],
                                dropout=wdrop,
                                temporal=temporalwdrop)
        self.network = nn.ModuleList([self.tnet])
        if n_experts > 0: self.network.append(self.mixsoft)
            
            
        # 5) Load model, if path specified
        if len(load) > 0:
            params_dict = torch.load(open(load, 'rb'))
            self.load_weights(params_dict)
            print("Model loaded successfully from {0}".format(load))

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def load_weights(self, params_dict):
        self.load_state_dict(params_dict)
   
    def save_weights(self, name):
        with open(name, 'wb') as f:
            d = self.state_dict()
            torch.save(d, f)

    def forward(self, input, hidden, decode=True):
        """
        Execute the forward pass of the deep network

        :param input: The input sequence, with dimesion (N, L)
        :param hidden: The initial hidden state (h, c)
        :param decode: Whether to use decoder
        :return: The predicted sequence
        """
        emb = embedded_dropout(self.encoder, input, self.emb_dropout if self.training else 0)
        emb = self.var_drop(emb, self.dropouti)
        emb = emb.transpose(1, 2)

        trellisnet = self.network[0]
        raw_output, hidden, all_raw_outputs = trellisnet(emb, hidden, aux=self.aux)
        output = self.var_drop(raw_output, self.dropout)
        all_outputs = self.var_drop(all_raw_outputs, self.dropout, dim=4) if self.aux else None  # N x M x L x C
        decoded, all_decoded = None, None

        if self.n_experts > 0 and not decode:
            raise ValueError("Mixture of softmax involves decoding phase. Must set decode=True")

        if self.n_experts > 0:
            decoded = torch.log(self.mixsoft(output).add_(1e-8))
            all_decoded = torch.log(self.mixsoft(all_outputs).add_(1e-8)) if self.aux else None

        if decode:
            decoded = decoded if self.n_experts > 0 else self.decoder(output)
            if self.aux: all_decoded = all_decoded if self.n_experts > 0 else self.decoder(all_outputs)  # N x M x L x C
            return (raw_output, output, decoded), hidden, all_decoded

        return (raw_output, output, output), hidden, all_outputs

    def init_hidden(self, bsz):
        h_size = self.nhid + self.nout
        weight = next(self.parameters()).data
        return (weight.new(bsz, h_size, 1).zero_(),
                weight.new(bsz, h_size, 1).zero_())
