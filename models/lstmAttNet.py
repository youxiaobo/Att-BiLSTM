# coding:utf8
import torch
from torch import nn
from .basic_module import BasicModule

class LSTMATTNet(BasicModule):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, device , D):
        super(LSTMATTNet, self).__init__()
        self.n_class = n_class
        self.n_layer = n_layer
        self.device = device
        # local shift step
        self.D = D
        # w is a learnable parameter
        self.w = nn.Parameter(torch.randn(hidden_dim).to(self.device))
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=True)
        #self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_dim, self.n_class)

        # pay attention parameter dim
        self.softmax = nn.Softmax(dim=1)

    def attlayer(self,H):
        batchsize = H.shape[0]
        seq_len = H.shape[1]
        shift = self.D

        # [batchsize,seq_len,hidden_dim]
        context = torch.zeros(H.shape).to(self.device)
        
        # [batchsize,seq_len,hidden_dim]
        M = self.tanh(H)
       
        # [batchsize,seq_len,seq_len]
        att_matrix = torch.zeros([batchsize,seq_len,seq_len]).to(self.device)

        #calculate local attention 
        for t in range(seq_len):
            startIdx = t - shift
            endIdx = t + shift
            if startIdx < 0:
                startIdx = 0
            if endIdx > seq_len - 1:
                endIdx = seq_len - 1
            
           

            
            # [batchsize,fixStep,hidden_dim]*[batchsize,hidden_dim,1] = [batchsize,fixStep,1] fixStep=2D+1
            locala = self.softmax(torch.bmm(M[:,startIdx:endIdx+1,:],self.w.repeat(batchsize,1,1).transpose(1,2)))
            # create a attention matrix, fill it with 0 out of the window
            att_weight = torch.cat((torch.zeros([batchsize,startIdx-0,1]).to(self.device),locala,torch.zeros([batchsize,seq_len-1-endIdx,1]).to(self.device)),1)
            

            localH = H[:,startIdx:endIdx+1,:]
            # [batchsize,1,fixStep] * [batchsize,fixStep,hidden_dim] = [batchsize,1,hidden_dim]
            localCt = self.tanh(torch.bmm(locala.transpose(1,2),localH))
            # [batchsize,hidden_dim]
            context[:,t,:] = localCt.squeeze(-2)
            
            #record attention matrix
            #[batchsize,seq,seq]
            att_matrix[:,t,:] = att_weight.transpose(1,2).squeeze(-2)
            

        return context,att_matrix
        

    def forward(self, x):
#        import ipdb
#        ipdb.set_trace()
        h0 = torch.zeros(self.n_layer*2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layer*2, x.size(0), self.hidden_dim).to(self.device)
        
        lstm_out, _ = self.lstm(x,(h0,c0))
        # get sum of hidden state
        lstm_out = lstm_out[:,:,:self.hidden_dim] + lstm_out[:,:,self.hidden_dim:]
        # use attention
        atth,att_matrix = self.attlayer(lstm_out)
        out = self.out(atth)
        
	# pay attention to the dim of tensor,or there are some error of criterion 
        out = out.view(-1, self.n_class)
        return out,lstm_out


