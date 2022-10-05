from torchfusion.datasets.datasets import *
from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator,StandardProjectionDiscriminator
from torch.optim import Adam
from torchfusion.datasets import mnist_loader
import torchvision
import torch.cuda as cuda
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from torch.distributions import Normal

G = StandardGenerator(output_size=(3,64,64),latent_size=100,num_classes=2)
D = StandardProjectionDiscriminator(input_size=(3,64,64),apply_sigmoid=False,num_classes=2)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())
    D = nn.DataParallel(D.cuda())

g_optim = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optim = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = imagefolder_loader(root="data/TB",transform=train_transforms,batch_size=16,shuffle=True)

learner = RAvgStandardGanLearner(G,D)

if __name__ == "__main__":
    learner.train(dataset,num_classes=2,gen_optimizer=g_optim,disc_optimizer=d_optim,save_outputs_interval=500,model_dir="./TB-gan-v16-20-085-099",latent_size=100,num_epochs=20,batch_log=False,display_metrics=True,save_metrics=True)
