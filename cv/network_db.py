
import torch
import torch.nn as nn
from torchvision.models import vgg16
from network import MLP
import torch.nn.functional as F

class Vgg16(torch.nn.Module):
	"""
	Network using VGG-16 for feature extraction

	"""
	def __init__(self):
		"""
		Construct copy of VGG-16
		See https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html for details
		"""
		super(Vgg16, self).__init__()
		v = vgg16(pretrained= True)
		# Copy modules of vgg16
		features = list(v.features)
		avgpool = v.avgpool
		sequentials = list(v.classifier)
		self.features = nn.ModuleList(features).eval() 
		self.avgpool = avgpool
		self.sequentials = nn.ModuleList(sequentials).eval() 

	def forward(self, x):
		"""
		Parameters
		----------
		x	: torch.Tensor
			input to the network
			batchsize * channel * height * width

		Returns
		-------
		x	: torch.Tensor
			output from the network
			batchsize * outout_dimension
		"""
		for i,model in enumerate(self.features):
			x = model(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		for i,model in enumerate(self.sequentials):
			x = model(x)
			# Return output of 1st layer in sequentials
			if i == 1:
				return x

class MnistMLP(MLP):
	"""
	Network using MLP for feature extraction

	"""
	def __init__(self, checkpoint_path, n_units, n_in, n_out):
		super().__init__(n_units, n_in, n_out)
		
		self.load_state_dict(torch.load(checkpoint_path))

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
