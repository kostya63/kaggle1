import torch

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda0 = torch.device('cuda:0')
x = torch.ones([10000, 10000], dtype=torch.float64, device=cuda0)