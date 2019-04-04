import argparse
import torch.optim as optim
import sys
from utils import *
from data import data_generator
import time
import math
from setproctitle import setproctitle

import warnings

sys.path.append("../")
from model import TrellisNetModel

warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser(description='PyTorch TrellisNet Language Model')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use')
parser.add_argument('--name', type=str, default='Trellis_charPTB',
                    help='name of the process')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1050,
                    help='number of hidden units per layer')
parser.add_argument('--nout', type=int, default=200,
                    help='number of output units')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=400,
                    help='upper epoch limit (default: 400)')
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size')

# For most of the time, you should change these two together
parser.add_argument('--nlevels', type=int, default=140,
                    help='levels of the network')
parser.add_argument('--horizon', type=int, default=140,
                    help='The effective history size')

parser.add_argument('--dropout', type=float, default=0.1,
                    help='output dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='input dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.26,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--emb_dropout', type=float, default=0.02,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.29,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights (default: True)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--anneal', type=int, default=5,
                    help='learning rate annealing criteria (default: 5)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--repack', action='store_false',
                    help='use repackaging (default: True)')
parser.add_argument('--eval', action='store_true',
                    help='evaluation only mode')
parser.add_argument('--aux', type=float, default=0.3,
                    help='use auxiliary loss (default: 0.3), -1 means no auxiliary loss used')
parser.add_argument('--aux_freq', type=float, default=80,
                    help='auxiliary loss frequency (default: 80)')
parser.add_argument('--seq_len', type=int, default=0,
                    help='total sequence length; if this is 0 then it defaults to args.horizon (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--when', nargs='+', type=int, default=[220, 350],
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

file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(args)

ntokens = len(corpus.dictionary)
eval_batch_size = 10
test_batch_size = 10
train_data = batchify(char_tensor(corpus, file), args.batch_size, args)
val_data = batchify(char_tensor(corpus, valfile), eval_batch_size, args)
test_data = batchify(char_tensor(corpus, testfile), eval_batch_size, args)
print(train_data.size(), val_data.size())


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
                            load=args.load_weight)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        hidden = model.init_hidden(eval_batch_size)
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
                hidden = model.init_hidden(eval_batch_size)

            data = data.t()
            net = nn.DataParallel(model) if data.size(0) > 10 else model
            (_, _, decoded), hidden, all_decoded = net(data, hidden)
            decoded = decoded.transpose(0, 1)
            targets = targets[eff_history:].contiguous().view(-1)
            final_decoded = decoded[eff_history:].contiguous().view(-1, ntokens)

            loss = criterion(final_decoded, targets)
            loss = loss.data

            total_loss += (data.size(1) - eff_history) * loss
            processed_data_size += data.size(1) - eff_history

        decoded = None
        final_decoded = None
        targets = None
        all_decoded = None   # This is for auxiliary losses; not used in evaluation

        return total_loss.item() / processed_data_size


def train(epoch):
    model.train()
    total_loss = 0
    total_aux_losses = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
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
        (_, _, decoded), hidden, all_decoded = net(data, hidden)
        decoded = decoded.transpose(0, 1)

        targets = targets[eff_history:].contiguous().view(-1)
        final_decoded = decoded[eff_history:].contiguous().view(-1, ntokens)

        # Loss 1: CE loss
        raw_loss = criterion(final_decoded, targets)

        # Loss 2: Aux loss
        aux_losses = 0
        if args.aux > 0:
            all_decoded = all_decoded[:, :, eff_history:].permute(1, 2, 0, 3).contiguous()
            aux_size = all_decoded.size(0)
            all_decoded = all_decoded.view(aux_size, -1, ntokens)
            aux_losses = args.aux * sum([criterion(all_decoded[i], targets) for i in range(aux_size)])

        # Combine losses
        loss = raw_loss + aux_losses
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        if args.aux:
            total_aux_losses += aux_losses.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_aux_loss = total_aux_losses.item() / args.log_interval if args.aux else 0
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'raw_loss {:5.3f} | aux_loss {:5.2f} | bpc {:5.3f}'.format(
                   epoch, batch, len(train_data) // validseqlen, lr,
                   elapsed * 1000 / args.log_interval, cur_loss, cur_aux_loss, cur_loss / math.log(2)))
            total_loss = 0
            total_aux_losses = 0
            start_time = time.time()

            sys.stdout.flush()

    decoded = None
    targets = None
    final_decoded = None
    all_decoded = None


def inference(epoch):
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.format(
        epoch, val_loss, val_loss / math.log(2)))
    test_loss = evaluate(test_data)
    print('| End of epoch {:3d} | test loss {:5.3f} | test bpc {:8.3f}'.format(
        epoch, test_loss, test_loss / math.log(2)))
    print('-' * 89)
    return val_loss, test_loss


if args.eval:
    print("Eval only mode")
    inference(-1)
    sys.exit(0)
    
lr = args.lr
best_val_loss = None
all_val_losses = []
all_test_losses = []
try:
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)

        val_loss, test_loss = inference(epoch)

        if not best_val_loss or val_loss < best_val_loss:
            print("Saving model (new best validation) in " + args.save)
            save(model, args)
            best_val_loss = val_loss

        if epoch in args.when:
            print("\n" + "*" * 89)
            if lr > 1e-5:
                print("Annealing learning rate")
                lr = lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        all_val_losses.append(val_loss)
        all_test_losses.append(test_loss)
        sys.stdout.flush()
except KeyboardInterrupt:
    print('-' * 89)
    print("Saving before quit...")
    save(model, args)

# Load the best saved model
with open(args.save, 'rb') as f:
    model = torch.load(f)
    model.save_weights('weights/pretrained_charptb.pkl')
    
# Run on test data
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
    test_loss, test_loss / math.log(2)))
print('=' * 89)

