import torch
from torch import nn
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import pandas as pd

# 'paraphrase-MiniLM-L6-v2'

class NLPDataset(Dataset):
	def __init__(self, data_file, transformer = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', transform=None):
		self.input_data = pd.read_csv(data_file, keep_default_na=False)

		self.model = SentenceTransformer(transformer, trust_remote_code=True)
		self.transform = transform
										
	def __len__(self):
		return len(self.input_data)
	
	def __getitem__(self, idx):
		label = self.input_data.iloc[idx, 4]
		key_word = torch.from_numpy(self.model.encode(self.input_data.iloc[idx, 1]))
		embedding = torch.from_numpy(self.model.encode(self.input_data.iloc[idx, 3]))
		#embedding = self.model.encode(self.input_data.iloc[idx, 3])
		embedding = torch.cat((key_word, embedding), dim=0)
		        
		return embedding, label
