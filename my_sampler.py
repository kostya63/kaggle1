import torch
from torch import nn
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import pandas as pd
import random

# 'paraphrase-MiniLM-L6-v2'
# Alibaba-NLP/gte-large-en-v1.5

class NLPDataset(Dataset):
	def __init__(self, data_file, embedder_on, transform = None):
		self.embedder_on = embedder_on
		
		if self.embedder_on:
			self.input_data = pd.read_json(data_file, orient='split')
		else:
			self.input_data = pd.read_csv(data_file)
		
		self.transform = transform
												
	def __len__(self):
		return len(self.input_data)
	
	def __getitem__(self, idx):
		label = self.input_data.iloc[idx, 1]
		
		if self.embedder_on:
			key_word = torch.Tensor(self.input_data.iloc[idx, 2])
			embedding = torch.Tensor(self.input_data.iloc[idx, 3])
		else:
			key_word = self.model.encode(self.input_data.iloc[idx, 1])
			embedding = self.model.encode(self.input_data.iloc[idx, 3])
		
		embedding = torch.cat((key_word, embedding), dim=0)
		        
		return embedding, label
