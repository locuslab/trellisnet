# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch import optim
import data
import sys
from utils import *
from setproctitle import setproctitle

sys.path.append("../")
from model import TrellisNetModel


parser = argparse.ArgumentParser(description='PyTorch TrellisNet Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--name', type=str, default='Trellis_wordPTB',
                    help='name of the process')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1000,
                    help='number of hidden units per layer')
parser.add_argument('--nout', type=int, default=400,
                    help='number of output units')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.225,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit (default: 500)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')

# For most of the time, you should change these two together
parser.add_argument('--nlevels', type=int, default=58,
                    help='levels of the network')
parser.add_argument('--horizon', type=int, default=58,
                    help='The effective history size')

parser.add_argument('--dropout', type=float, default=0.45,
                    help='output dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.45,
                    help='input dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.28,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=0.29,
                    help='dropout applied to latent layer in MoS (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights (default: True)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--anneal', type=int, default=10,
                    help='learning rate annealing criteria (default: 10)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--wnorm', action='store_true',
                    help='use weight normalization (default: False)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use')
parser.add_argument('--asgd', action='store_true',
                    help='use ASGD when learning plateaus (follows from Merity et al. 2017) (default: False)')
parser.add_argument('--repack', action='store_true',
                    help='use repackaging (default: False)')
parser.add_argument('--eval', action='store_true',
                    help='evaluation only mode')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on outputs (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on outputs (beta = 0 means no regularization)')
parser.add_argument('--aux', type=float, default=0.05,
                    help='use auxiliary loss (default: 0.05), -1 means no auxiliary loss used')
parser.add_argument('--aux_freq', type=float, default=12,
                    help='auxiliary loss frequency (default: 12)')
parser.add_argument('--seq_len', type=int, default=110,
                    help='total sequence length, including effective history (default: 110)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--when', nargs='+', type=int, default=[-1],
                    help='When to decay the learning rate')
parser.add_argument('--ksize', type=int, default=2,
                    help='conv kernel size (default: 2)')
parser.add_argument('--dilation', nargs='+', type=int, default=[1],
                    help='dilation rate (default: [1])')
parser.add_argument('--n_experts', type=int, default=0,
                    help='number of softmax experts (default: 0)')
parser.add_argument('--load', type=str, default='',
                    help='path to load the model')
parser.add_argument('--load_weight', type=str, default='',
                    help='path to load the model weights (please only use --load or --load_weight)')

args = parser.parse_args()
args.save = args.name + ".pt"

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

import os
import hashlib

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    corpus = torch.load(fn)
else:
    print('Processing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 12
test_batch_size = 12
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)


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

ntokens = len(corpus.dictionary)

if len(args.load) > 0:
    print("Loaded model\n")
    model = torch.load(args.load)
else:
    model = TrellisNetModel(ntoken=ntokens,
                            ninp=args.emsize,
                            nhid=args.nhid,
                            nout=args.nout,
                            nlevels=args.nlevels,
                            kernel_size=args.ksize,
                            dilation=args.dilation,
                            dropout=args.dropout,
                            dropouti=args.dropouti,
                            dropouth=args.dropouth,
                            dropoutl=args.dropoutl,
                            emb_dropout=args.emb_dropout,
                            wdrop=args.wdrop,
                            temporalwdrop=args.temporalwdrop,
                            tie_weights=args.tied,
                            repack=args.repack,
                            wnorm=args.wnorm,
                            aux=(args.aux > 0),
                            aux_frequency=args.aux_freq,
                            n_experts=args.n_experts,
                            load=args.load_weight)

if args.cuda:
    model.cuda()

criterion = nn.NLLLoss() if args.n_experts > 0 else nn.CrossEntropyLoss()
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wdecay)


###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
        ntokens = len(corpus.dictionary)
        batch_size = data_source.size(1)
        hidden = model.init_hidden(batch_size)
        eff_history_mode = (args.seq_len > args.horizon and not args.repack)

        if eff_history_mode:
            validseqlen = args.seq_len - args.horizon
            seq_len = args.seq_len
        else:
            validseqlen = args.horizon
            seq_len = args.horizon

        processed_data_size = 0
        for i in range(0, data_source.size(0) - 1, validseqlen):
            eff_history = args.horizon if eff_history_mode else 0
            if i + eff_history >= data_source.size(0) - 1: continue
            data, targets = get_batch(data_source, i, seq_len, evaluation=True)

            if args.repack:
                hidden = repackage_hidden(hidden)
            else:
                hidden = model.init_hidden(data.size(1))

            data = data.t()
            net = nn.DataParallel(model) if batch_size > 10 else model
            (_, output, decoded), hidden, _ = net(data, hidden)
            decoded = decoded.transpose(0, 1)
            targets = targets[eff_history:].contiguous().view(-1)
            final_decoded = decoded[eff_history:].contiguous().view(-1, ntokens)

            loss = criterion(final_decoded, targets)
            loss = loss.data

            total_loss += (data.size(1) - eff_history) * loss
            processed_data_size += data.size(1) - eff_history

        output = None
        decoded = None
        targets = None
        final_output = None
        final_decoded = None

        return total_loss.item() / processed_data_size


def train(epoch):
    model.train()
    total_loss = 0
    total_aux_losses = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    eff_history_mode = (args.seq_len > 0 or not args.repack)

    if eff_history_mode:
        validseqlen = args.seq_len - args.horizon
        seq_len = args.seq_len
    else:
        validseqlen = args.horizon
        seq_len = args.horizon

    for batch, i in enumerate(range(0, train_data.size(0) - 1, validseqlen)):
        # When not using repackaging mode, we DISCARD the first arg.horizon outputs in backprop (which are
        # the "effective history".
        eff_history = args.horizon if eff_history_mode else 0
        if i + eff_history >= train_data.size(0) - 1: continue
        data, targets = get_batch(train_data, i, seq_len)

        if args.repack:
            hidden = repackage_hidden(hidden)
        else:
            hidden = model.init_hidden(args.batch_size)

        optimizer.zero_grad()
        data = data.t()
        net = nn.DataParallel(model) if data.size(0) > 10 else model
        (raw_output, output, decoded), hidden, all_decoded = net(data, hidden)
        decoded = decoded.transpose(0, 1)

        targets = targets[eff_history:].contiguous().view(-1)
        final_decoded = decoded[eff_history:].contiguous().view(-1, ntokens)

        # Loss 1: CE loss
        raw_loss = criterion(final_decoded, targets)

        # Loss 2: Aux loss
        aux_losses = 0
        if args.aux > 0:
            all_decoded = all_decoded[:, :, eff_history:].permute(1, 2, 0, 3).contiguous()  # (N, M, L, C) --> (M, L, N, C)
            aux_size = all_decoded.size(0)
            all_decoded = all_decoded.view(aux_size, -1, ntokens)
            aux_losses = args.aux * sum([criterion(all_decoded[i], targets) for i in range(aux_size)])

        # Loss 3: AR & TAR
        alpha_loss = 0
        beta_loss = 0
        if args.alpha > 0:
            output = output.transpose(0, 1)
            final_output = output[eff_history:]
            alpha_loss = args.alpha * final_output.pow(2).mean()
        if args.beta > 0:
            raw_output = raw_output.transpose(0, 1)
            final_raw_output = raw_output[eff_history:]
            beta_loss = args.beta * (final_raw_output[1:] - final_raw_output[:-1]).pow(2).mean()

        # Combine losses
        loss = raw_loss + aux_losses + alpha_loss + beta_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        if args.aux:
            total_aux_losses += aux_losses.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_aux_loss = total_aux_losses.item() / args.log_interval if args.aux else 0
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'raw_loss {:5.2f} | aux_loss {:5.2f} | ppl {:8.2f}'.format(
                   epoch, batch, len(train_data) // validseqlen, lr,
                   elapsed * 1000 / args.log_interval, cur_loss, cur_aux_loss, math.exp(cur_loss)))
            total_loss = 0
            total_aux_losses = 0
            start_time = time.time()

            sys.stdout.flush()

    raw_output = None
    output = None
    decoded = None
    targets = None
    final_output = None
    final_decoded = None
    all_decoded = None
    all_outputs = None
    final_raw_output = None


def inference(epoch, epoch_start_time):
    val_loss = evaluate(val_data)
    test_loss = evaluate(test_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                    test_loss, math.exp(test_loss)))
    print('-' * 89)
    return val_loss, test_loss

if args.eval:
    print("Eval only mode")
    inference(-1, time.time())
    sys.exit(0)

lr = args.lr
best_val_loss = None
all_val_losses = []
all_test_losses = []
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(epoch)

        if 't0' in optimizer.param_groups[0]:
            # Average SGD, see (Merity et al. 2017).
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss, test_loss = inference(epoch, epoch_start_time)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    # model.save_weights('weights/pretrained.pkl')
                    print('ASGD Saving model (new best validation) in ' + args.save)
                best_val_loss = val_loss
            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss, test_loss = inference(epoch, epoch_start_time)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    # model.save_weights('weights/pretrained.pkl')
                    print('Saving model (new best validation) in ' + args.save)
                best_val_loss = val_loss

            if len(all_val_losses) > args.anneal and val_loss > min(all_val_losses[:-args.anneal]):
                print("\n" + "*" * 89)
                if args.asgd and 't0' not in optimizer.param_groups[0]:
                    print('Switching to ASGD')
                    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    args.save = args.name + "_asgd.pt"
                elif lr > 0.02:
                    print('Annealing learning rate')
                    lr /= 4.0
                    optimizer.param_groups[0]['lr'] = lr
                print("*" * 89 + "\n")

        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        sys.stdout.flush()
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    delete_cmd = input('DO YOU WANT TO DELETE THIS RUN [YES/NO]:')
    if delete_cmd == "YES":
        import os
        os.remove('logs/' + args.name + ".log")
        print("Removed log file")
        os.remove('logs/' + args.name + ".pt")
        print("Removed pt file")

# Load the best saved model
with open(args.save, 'rb') as f:
    model = torch.load(f)
    print("Saving the pre-trained weights of the best saved model")
    model.save_weights('weights/pretrained_wordptb.pkl')

# Run on test data
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
