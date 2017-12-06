import torch
import torch.nn as nn
from torch.autograd import Variable

from time import time


class RevRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, tie_weights=False):
        super(RevRNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn1 = RevLSTMCell(ninp, nhid)
        self.rnn2 = RevLSTMCell(nhid, nhid)
        self.rnn = MultiRNNCell([self.rnn1, self.rnn2])
        #self.rnn = RevLSTMCell(ninp, nhid)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output)
        return decoded, hidden

    def reconstruct(self, input, states):
        emb = self.encoder(input)
        output, old_cell_states = self.rnn.reconstruct(emb, states)
        return output, old_cell_states

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)
        
class MultiRNNCell(nn.Module):
    """Defining multi layer rnn"""
    def __init__(self, cells):
        super(MultiRNNCell, self).__init__()
        self._cells = cells
    def forward(self, input, cell_states):
        current_input = input
        new_cell_states = []
        for i, cell in enumerate(self._cells):
            #import pdb; pdb.set_trace()
            current_input, state = cell(current_input, cell_states[i])
            new_cell_states.append(state)
        return current_input, new_cell_states
    def reconstruct(self, input, new_cell_states):
        current_input = input
        cell_states = []
        for i, cell in enumerate(self._cells):
            current_input, state = cell.reconstruct(current_input, new_cell_states[i])
            cell_states.append(state)
        return current_input, cell_states
    def init_hidden(self, bsz):
        return tuple(x.init_hidden(bsz) for x in self._cells)
         
class RevLSTMCell(nn.Module):
    """ Defining Network Completely along with gradients to Variables """
    def __init__(self, input_size, hidden_size):
        super(RevLSTMCell, self).__init__()
        self.f_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.i_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.uc_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.u_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.r_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.uh_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size
        self.init_weights()
        
    def forward(self, x, state):
        c, h = state
        #import pdb; pdb.set_trace()
        concat1 = torch.cat((x, h), dim=-1)
        f = self.sigmoid(self.f_layer(concat1))
        i = self.sigmoid(self.i_layer(concat1))
        c_ = self.tanh(self.uc_layer(concat1))
        new_c = (f*c + i*c_)/2
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.sigmoid(self.u_layer(concat2))
        r = self.sigmoid(self.r_layer(concat2))
        h_ = self.tanh(self.uh_layer(concat2))
        new_h = (r*h + u*h_)/2
        return new_h, (new_c, new_h)
    
    def reconstruct(self, x, state):
        new_c, new_h = state
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.sigmoid(self.u_layer(concat2))
        r = self.sigmoid(self.r_layer(concat2))
        h_ = self.tanh(self.uh_layer(concat2))
        h = (2*new_h - u*h_)/(r+1e-64)
        
        
        concat1 = torch.cat((x, h), dim=-1)
        f = self.sigmoid(self.f_layer(concat1))
        i = self.sigmoid(self.i_layer(concat1))
        c_ = self.tanh(self.uc_layer(concat1))
        c = (2*new_c - i*c_)/(f+1e-64)
        
        return h, (c, h)
    
    def init_weights(self):
        for parameter in self.parameters():
            if parameter.ndimension() == 2:
                nn.init.xavier_uniform(parameter, gain=0.01)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.hidden_size).zero_()),
                    Variable(weight.new(bsz, self.hidden_size).zero_()))

class RevLSTMCell2(nn.Module):
    """ Defining Network Completely along with gradients to Variables """
    def __init__(self, input_size, hidden_size):
        super(RevLSTMCell2, self).__init__()
        self.f_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.i_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.u_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.r_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.init_weights()
        
    def forward(self, x, state):
        c, h = state
        #import pdb; pdb.set_trace()
        concat1 = torch.cat((x, h), dim=-1)
        f = self.relu(self.f_layer(concat1))
        i = self.relu(self.i_layer(concat1))
        new_c = (c - f + i)/2
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.relu(self.u_layer(concat2))
        r = self.relu(self.r_layer(concat2))
        new_h = (h - r + u)/2
        return new_h, (new_c, new_h)
    
    def reconstruct(self, x, state):
        new_c, new_h = state
        
        concat2 = torch.cat((x, new_c), dim=-1)
        u = self.relu(self.u_layer(concat2))
        r = self.relu(self.r_layer(concat2))
        h = (2*new_h - u + r)
                
        concat1 = torch.cat((x, h), dim=-1)
        f = self.relu(self.f_layer(concat1))
        i = self.relu(self.i_layer(concat1))
        c = (2*new_c - i + f)
        
        return h, (c, h)
    
    def init_weights(self):
        for parameter in self.parameters():
            if parameter.ndimension() == 2:
                nn.init.xavier_uniform(parameter, gain=1)
    
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.hidden_size).zero_()),
                    Variable(weight.new(bsz, self.hidden_size).zero_()))

class RevLSTMCell3(nn.Module):
    """ Defining Network Completely along with gradients to Variables """
    def __init__(self, input_size, hidden_size):
        super(RevLSTMCell3, self).__init__()
        self.c1_layer= nn.Linear(input_size + hidden_size, hidden_size) 
        self.c2_layer= nn.Linear(hidden_size, hidden_size, bias=False)
        self.h1_layer= nn.Linear(input_size + hidden_size, hidden_size)
        self.h2_layer= nn.Linear(hidden_size, hidden_size, bias=False) 
        #self.c_batchnorm = nn.BatchNorm1d(hidden_size)
        #self.h_batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.init_weights()
        
    def forward(self, x, state):
        c, h = state
        #import pdb; pdb.set_trace()
        concat1 = torch.cat((x, h), dim=-1)
        c_ = self.relu(self.c1_layer(concat1))
        c_ = self.c2_layer(c_)
        new_c = (c + c_)
        
        concat2 = torch.cat((x, new_c), dim=-1)
        h_ = self.relu(self.h1_layer(concat2))
        h_ = self.h2_layer(h_)
        new_h = (h - h_)
        return new_h, (new_c, new_h)
    
    def reconstruct(self, x, state):
        new_c, new_h = state
        
        concat2 = torch.cat((x, new_c), dim=-1)
        h_ = self.relu(self.h1_layer(concat2))
        h_ = self.h2_layer(h_)
        h = (new_h + h_)
                
        concat1 = torch.cat((x, h), dim=-1)
        c_ = self.relu(self.c1_layer(concat1))
        c_ = self.c2_layer(c_)
        c = (new_c - c_)
        
        return h, (c, h)
    
    def init_weights(self):
        for parameter in self.parameters():
            if parameter.ndimension() == 2:
                nn.init.xavier_uniform(parameter, gain=1)
    
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.hidden_size).zero_()),
                    Variable(weight.new(bsz, self.hidden_size).zero_()))

