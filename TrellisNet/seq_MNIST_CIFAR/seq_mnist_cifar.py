import torch
import torch.optim as optim
import torch.nn.functional as F
from model import TrellisNetModel
from utils import *
import numpy as np
import argparse
from setproctitle import setproctitle

parser = argparse.ArgumentParser(description='PyTorch TrellisNet Sequence Model - Sequential/Permuted MNIST & CIFAR-10')
parser.add_argument('--name', type=str, default='Trellis_seqMNIST_CIFAR',
                    help='name of the process')

parser.add_argument('--nhid', type=int, default=120,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')

parser.add_argument('--nlevels', type=int, default=11,
                    help='steps unrolled')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='output locked dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='input locked dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.1,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.2,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--anneal', type=int, default=10,
                    help='learning rate annealing criteria (default: 10)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use')
parser.add_argument('--aux', type=float, default=0,
                    help='use auxiliary loss (default: 0), -1 means w/o')
parser.add_argument('--aux_freq', type=float, default=1e4,
                    help='auxiliary loss frequency (default: 1e4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--when', nargs='+', type=int, default=[50, 75, 90],
                    help='When to decay the learning rate')
parser.add_argument('--ksize', type=int, default=2,
                    help='conv kernel size (default: 2)')
parser.add_argument('--dilation', nargs='+', type=int, default=[1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    help='dilation rate (default: [1,1,2,4,8,16,32,64,128,256,512])')
parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--permute', action='store_false',
                    help='use permuted dataset (default: True)')
parser.add_argument('--cifar', action='store_true',
                    help='use CIFAR (default: false)')
args = parser.parse_args()


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
setproctitle(args.name)
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

if args.cifar:
    print("Loading CIFAR-10")
    root = "./data/cifar10"
    input_channels = 3
    seq_length = 32 * 32
else:
    print("Loading PMNIST" if args.permute else "Loading MNIST")
    root = './data/mnist'
    input_channels = 1
    seq_length = 28 * 28

batch_size = args.batch_size
n_classes = 10
epochs = args.epochs
steps = 0
train_loader, test_loader = data_generator(root, batch_size)
permute = torch.Tensor(np.random.permutation(seq_length).astype(np.float64)).long()   # Use only if args.permute is True

import sys


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logs/" + args.name + ".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()
        self.terminal.flush()
        pass


sys.stdout = Logger()


###############################################################################
# Build the model
###############################################################################

if len(args.load) > 0:
    print("Loaded model\n")
    model = torch.load(args.load)
else:
    model = TrellisNetModel(ninp=input_channels,
                            nhid=args.nhid,
                            nout=n_classes,
                            nlevels=args.nlevels,
                            kernel_size=args.ksize,
                            dilation=args.dilation,
                            dropout=args.dropout,
                            dropouti=args.dropouti,
                            dropouth=args.dropouth,
                            wdrop=args.wdrop,
                            temporalwdrop=args.temporalwdrop,
                            wnorm=args.wnorm,
                            aux=(args.aux > 0),
                            aux_frequency=args.aux_freq)

if args.cuda:
    model.cuda()
    permute = permute.cuda()


lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

###############################################################################
# Training code
###############################################################################


def train(epoch):
    global steps
    train_loss = 0
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]

        hidden = model.init_hidden(data.size(0))

        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.6f}\tLoss: {:.6f}\tSteps: {}'.format(
                   epoch, batch_idx * batch_size, len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), lr, train_loss.data.item() / args.log_interval, steps))
            train_loss = 0

            sys.stdout.flush()


def test():
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]

        hidden = model.init_hidden(data.size(0))

        output, hidden = model(data, hidden)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    sys.stdout.flush()
    return test_loss


if __name__ == "__main__":
    all_test_losses = []
    for epoch in range(1, epochs + 1):
        train(epoch)
        test_loss = test()
        if epoch in args.when:
            # Scheduled learning rate decay
            lr /= 10.
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        all_test_losses.append(test_loss)
