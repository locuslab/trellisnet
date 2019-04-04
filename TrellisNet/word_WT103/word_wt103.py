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
from splitcross import *

sys.path.append("../")
from model import TrellisNetModel

parser = argparse.ArgumentParser(description='PyTorch TrellisNet Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--name', type=str, default='Trellis_wordWT103',
                    help='name of the process')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1600,
                    help='number of hidden units per layer')
parser.add_argument('--nout', type=int, default=512,
                    help='number of output units')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--clip', type=float, default=0.07,
                    help='gradient clipping (default: 0.07)')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit (default: 25)')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size (default: 40)')

# For most of the time, you should change these two together
parser.add_argument('--nlevels', type=int, default=75,
                    help='levels of the network')
parser.add_argument('--horizon', type=int, default=75,
                    help='The effective history size')

parser.add_argument('--dropout', type=float, default=0.1,
                    help='output dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='input dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.0,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--emb_dropout', type=float, default=0.0,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.1,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights (default: True)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--anneal', type=int, default=5,
                    help='learning rate annealing criteria (default: 5)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--repack', action='store_false',
                    help='use repackaging (default: True)')
parser.add_argument('--aux', type=float, default=0.1,
                    help='use auxiliary loss (default: 0.1), -1 means no auxiliary loss used')
parser.add_argument('--aux_freq', type=float, default=25,
                    help='auxiliary loss frequency (default: 25)')
parser.add_argument('--seq_len', type=int, default=0,
                    help='total sequence length; if this is 0 then it defaults to args.horizon (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--when', nargs='+', type=int, default=[15, 20],
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

eval_batch_size = 1
test_batch_size = 1
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

splits = []
if ntokens > 75000:
    splits = [2800, 20000, 76000]    # This can be tuned.

criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)   # Use an adaptive softmax

if args.cuda:
    criterion = criterion.cuda()
params = list(model.parameters()) + list(criterion.parameters())

lr = args.lr
optimizer = getattr(optim, args.optim)(params, lr=lr, weight_decay=args.wdecay)


###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        total_loss = 0
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
            (_, _, output), hidden, _ = net(data, hidden, decode=False)
            output = output.transpose(0, 1)
            targets = targets[eff_history:].contiguous().view(-1)
            final_output = output[eff_history:].contiguous().view(-1, output.size(2))

            loss = criterion(model.decoder.weight, model.decoder.bias, final_output, targets)
            loss = loss.data

            total_loss += (data.size(1) - eff_history) * loss
            processed_data_size += data.size(1) - eff_history

        data = None
        output = None
        targets = None
        final_output = None

        return total_loss.item() / processed_data_size


def train(epoch):
    model.train()
    total_loss = 0
    total_aux_losses = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    eff_history_mode = (args.seq_len > args.horizon and not args.repack)

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
        (raw_output, _, output), hidden, all_outputs = net(data, hidden, decode=False)
        raw_output = raw_output.transpose(0, 1)
        output = output.transpose(0, 1)
        targets = targets[eff_history:].contiguous().view(-1)
        final_output = output[eff_history:].contiguous().view(-1, output.size(2))
        dec_weight, dec_bias = model.decoder.weight, model.decoder.bias

        # Loss 1: CE loss
        raw_loss = criterion(dec_weight, dec_bias, final_output, targets)

        # Loss 2: Aux loss
        aux_losses = 0
        if args.aux > 0:
            all_outputs = all_outputs[:, :, eff_history:].permute(1, 2, 0, 3).contiguous()
            aux_size = all_outputs.size(0)   # The number of auxiliary losses
            all_outputs = all_outputs.view(aux_size, -1, all_outputs.size(3))
            aux_losses = args.aux * sum([criterion(dec_weight, dec_bias, all_outputs[i], targets) for i in range(aux_size)])

        # Combine losses
        loss = raw_loss + aux_losses
        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        if args.aux:
            total_aux_losses += aux_losses.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_aux_loss = total_aux_losses.item() / args.log_interval if args.aux else 0
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'raw_loss {:5.2f} | aux_loss {:5.2f} | ppl {:8.2f}'.format(
                   epoch, batch, len(train_data) // validseqlen, lr,
                   elapsed * 1000 / args.log_interval, cur_loss, cur_aux_loss, math.exp(cur_loss)))
            total_loss = 0
            total_aux_losses = 0
            start_time = time.time()

        sys.stdout.flush()

    data = None
    raw_output = None
    output = None
    targets = None
    final_output = None
    all_outputs = None


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
    
# Loop over epochs
lr = args.lr
best_val_loss = None
all_val_losses = []
all_test_losses = []
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train(epoch)

        val_loss, test_loss = inference(epoch, epoch_start_time)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
                print('Saving model (new best validation) in ' + args.save)
            best_val_loss = val_loss

        if (len(all_val_losses) > args.anneal and val_loss > min(all_val_losses[:-args.anneal])) \
                or epoch in args.when:
            print("\n" + "*" * 89)
            if lr > 1e-5:
                print('Annealing learning rate')
                lr /= 10.0
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
    model.save_weights('weights/pretrained_wt103.pkl')

# Run on test data
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)
