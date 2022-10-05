from torchfusion.gan.learners import *
from torchfusion.gan.applications import StandardGenerator
import torch.cuda as cuda
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

import os


G = StandardGenerator(output_size=(3,64,64),latent_size=100,num_classes=4)

if cuda.is_available():
    G = nn.DataParallel(G.cuda())

learner = RAvgHingeGanLearner(G,None)
learner.load_generator("covid4-gan-v7-10/gen_models/gen_model_10.pth")
# learner.load_generator("TB-gan-v4-50/gen_models/gen_model_50.pth")

result_dir = "demo/covid4/COVID"
# result_dir = "demo/covid4/Lung Opacity"
# result_dir = "demo/covid4/Normal"
# result_dir = "demo/covid4/Viral Pneumonia"

# result_dir = "demo/tb/normal"
# result_dir = "demo/tb/tuberculosis"

def generate_img(i):
    "Define an instance of the normal distribution"
    dist = Normal(0,1)

    #Get a sample latent vector from the distribution
    latent_vector = dist.sample((1,100))

    #Define the class of the image you want to generate
    label = torch.LongTensor(1).fill_(0)

    #Run inference
    image = learner.predict([latent_vector,label])
    # images = make_grid(image.cpu().data, normalize=True)
    # images = np.transpose(images.numpy(), (1, 2, 0))
    save_image(image, os.path.join(result_dir, f"{i:04d}.png"))
    # plt.axis("off")
    # plt.imshow(images)
    # plt.show()


def main():
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    for i in range(50):
        generate_img(i)


if __name__=="__main__":
    main()