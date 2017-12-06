import argparse
import time
import math
from functools import reduce
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from tqdm import tqdm

import data
import reversible

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--log', type=str, default='./log.txt')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = reversible.RevRNNModel(ntokens, args.emsize, args.nhid, args.tied)
if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
reg_criterion = nn.L1Loss()
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def attach(h):
    if type(h) == Variable:
        h.requires_grad=True
        return h
    else:
        return tuple(attach(v) for v in h)

def detach(h):
    if type(h) == Variable:
        h.volatile=True
        h.requires_grad=False
        return h
    else:
        return tuple(attach(v) for v in h)

def flatten(input_list):
    output = ()
    for item in input_list:
        output += flatten(item) if type(item) == list or type(item) == tuple else (item,)
    return output

def group(input_list):
    return tuple(zip(*[iter(L)]*2))

def get_batch(source, i, evaluation=False):
    #seq_len = min(args.bptt, len(source) - 1 - i)
    seq_len = 1
    data = Variable(source[i], volatile=evaluation)
    target = Variable(source[i+1].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_loss_error = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    cell_states = model.init_hidden(args.batch_size)
    #optimizer = optim.Adam(model.parameters(), lr = lr)
    with open(args.log, 'w') as f:
        f.write("")
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        cell_states = repackage_hidden(cell_states)
        #cell_states = model.init_hidden(args.batch_size)
        state_grad = model.init_hidden(args.batch_size)
        #optimizer.zero_grad()
        model.zero_grad()
        forward_loss = 0
        backward_loss = 0
        
        for j in range(0, args.bptt):
            data, targets = get_batch(train_data, i+j)
            
            output, cell_states = model(data, cell_states)
            #import pdb; pdb.set_trace()
            loss = criterion(output.view(-1, ntokens), targets)/args.bptt
            forward_loss += loss.data
            del output, data, targets
            cell_states = repackage_hidden(cell_states)    
        new_cell_states = repackage_hidden(cell_states)
        cell_states = repackage_hidden(cell_states)
        
        for j in range(args.bptt-1, -1, -1):
            data, targets = get_batch(train_data, i+j)
            output, old_cell_states = model.reconstruct(data, new_cell_states)
            
            old_cell_states = attach(repackage_hidden(old_cell_states))
            output, new_cell_states_recon = model(data, old_cell_states)
            loss = criterion(output.view(-1, ntokens), targets)/args.bptt
            backward_loss += loss.data
                #import pdb; pdb.set_trace()
            #reg_loss = reduce((lambda x,y: x+y),(5*torch.mean(torch.pow(p, 2)) for p in model.parameters() if p.ndimension() == 2))
            #loss += reg_loss + reduce((lambda x,y: x+y),(torch.mean(torch.pow((x-y),2)) for x,y in zip(flatten(new_cell_states_recon), flatten(detach(new_cell_states)))))
            
            state_grad = grad((loss,)+flatten(new_cell_states_recon), flatten(old_cell_states), (None,)+flatten(state_grad), only_inputs=False)
            del loss, data, targets
            new_cell_states = repackage_hidden(old_cell_states)
            state_grad = repackage_hidden(state_grad)
            del old_cell_states
            
            
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # print(i, forward_loss, backward_loss)
        total_loss_error = (forward_loss - backward_loss)**2
        parameter_norm = 0
        # print(total_loss_error)
        # import pdb; pdb.set_trace()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            #import pdb; pdb.set_trace()
            #print(p, p.grad)
            if p.ndimension() == 2:
                parameter_norm += torch.mean(torch.pow(p, 2))
            p.data.add_(-lr, p.grad.data)
        #optimizer.step()
        total_loss += backward_loss
        #cur_loss = total_loss[0] / args.log_interval
        #print(parameter_norm.data.max(), total_loss_error.max(), math.exp(cur_loss))
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            string = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '\
                    'loss {:5.2f} | ppl {:8.2f} | norm {} | error {} |'.format(
                    epoch, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), parameter_norm.data.max(), total_loss_error.max())
            print(string)
            with open(args.log, 'a') as f:
                f.write(string+'\n')
            total_loss = 0
            start_time = time.time()
            #print(parameter_norm.data.max(), total_loss_error.max())

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
