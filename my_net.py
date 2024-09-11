import torch
from torch import nn
import torch.nn.functional as F


class nlp_nn(nn.Module):
	def __init__(self, embedding_dim=2048, L1=1024, L2=512, L3=256, L4=128, L5=64, L6=32, L7=16, L8=8):
		super(nlp_nn, self).__init__()
		self.Flatten = nn.Flatten()
		self.Linear1 = nn.Linear(embedding_dim, L1)
		self.Linear2 = nn.Linear(L1, L2)
		self.Linear3 = nn.Linear(L2, L3)
		self.Linear4 = nn.Linear(L3, L4)
		self.Linear5 = nn.Linear(L4, L5)
		self.Linear6 = nn.Linear(L5, L6)
		self.Linear7 = nn.Linear(L6, L7)
		self.Linear8 = nn.Linear(L7, L8)
		self.LinearFinal = nn.Linear(L8, 2)
		self.Drop0 = nn.Dropout(0.25)
		self.Drop1 = nn.Dropout(0.25)
		self.Drop2 = nn.Dropout(0.25)
		self.Drop3 = nn.Dropout(0.25)
		self.Drop4 = nn.Dropout(0.25)
		self.Drop5 = nn.Dropout(0.25)
		self.Drop6 = nn.Dropout(0.25)
		self.Drop7 = nn.Dropout(0.25)
		self.Drop8 = nn.Dropout(0.25)
		
		
	def forward(self, x):
		act = nn.ReLU()
		soft = nn.Softmax(dim=0)

		if x.ndim > 1: x = self.Flatten(x)
		
		x = self.Drop0(x)

		x = act(self.Linear1(x))
		x = self.Drop1(x)
		
		x = act(self.Linear2(x))
		x = self.Drop2(x)
		
		x = act(self.Linear3(x))
		x = self.Drop3(x)

		x = act(self.Linear4(x))
		x = self.Drop4(x)

		x = act(self.Linear5(x))
		x = self.Drop5(x)

		x = act(self.Linear6(x))
		x = self.Drop6(x)

		x = act(self.Linear7(x))
		x = self.Drop7(x)

		x = act(self.Linear8(x))
		x = self.Drop8(x)

		x = self.LinearFinal(x)
				
		return x