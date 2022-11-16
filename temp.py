import torch
import os

class Temp(torch.nn.Module):
	def __init__(self):
		super().__init__()

temp = Temp()
dir_path = "ckpts/single_context_pairwise/"
if not os.path.isdir(dir_path):
	os.mkdir(dir_path)
torch.save(temp, os.path.join(dir_path, "model.pt"))