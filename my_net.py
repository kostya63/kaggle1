import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class nlp_nn(nn.Module):
	def __init__(self, embedding_dim, L1, L2, L3, L4, L5):
		super(nlp_nn, self).__init__()
		self.Flatten = nn.Flatten()
		self.Linear1 = nn.Linear(embedding_dim, L1)
		self.Linear2 = nn.Linear(L1, L2)
		self.Linear3 = nn.Linear(L2, L3)
		self.Linear4 = nn.Linear(L3, L4)
		self.Linear5 = nn.Linear(L4, L5)
		self.LinearFinal = nn.Linear(L1, 2)
		self.Drop0 = nn.Dropout(0.4)
		self.Drop1 = nn.Dropout(0.4)
		self.Drop2 = nn.Dropout(0.4)
		self.Drop3 = nn.Dropout(0.4)
		self.Drop4 = nn.Dropout(0.4)
		self.Drop5 = nn.Dropout(0.4)
		
		
	def forward(self, x):
		act = nn.ReLU()
		soft = nn.Softmax(dim=0)

		x = self.Flatten(x)
		
		x = self.Drop0(x)

		x = act(self.Linear1(x))
		x = self.Drop1(x)
		
		#x = act(self.Linear2(x))
		#x = self.Drop2(x)
		
		#x = act(self.Linear3(x))
		#x = self.Drop3(x)

		#x = act(self.Linear4(x))
		#x = self.Drop4(x)

		#x = act(self.Linear5(x))
		#x = self.Drop5(x)

		x = self.LinearFinal(x)
				
		return x
