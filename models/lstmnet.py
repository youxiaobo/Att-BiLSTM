# coding:utf8
import torch
from torch import nn
from .basic_module import BasicModule

class LSTMNet(BasicModule):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, device):
        super(LSTMNet, self).__init__()
        self.n_class = n_class
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=True)
        #self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.out = nn.Linear(hidden_dim*2, self.n_class)
        #self.out = nn.Linear(hidden_dim, self.n_class)
        self.device = device

    def forward(self, x):
#        import ipdb
#        ipdb.set_trace()
        h0 = torch.zeros(self.n_layer*2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layer*2, x.size(0), self.hidden_dim).to(self.device)
        
        lstm_out, _ = self.lstm(x,(h0,c0))
        #lstm_out, _ = self.lstm(x)
        out = self.out(lstm_out)
	# pay attention to the dim of tensor,or there are some error of criterion 
        out = out.view(-1, self.n_class)
        #return out
        return out,lstm_out


