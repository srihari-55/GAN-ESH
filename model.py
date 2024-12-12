import torch
import os
from dotenv import load_dotenv
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


load_dotenv()

# Assign variables
root_dir = os.getenv("ROOT_DIR")
workers = int(os.getenv("WORKERS"))
batch_size = int(os.getenv("BATCH_SIZE"))
image_size = int(os.getenv("IMAGE_SIZE"))
nc = int(os.getenv("NC"))
nz = int(os.getenv("NZ"))
ngf = int(os.getenv("NGF"))
ngc = int(os.getenv("NGC"))
ndf = int(os.getenv("NDF"))
num_epochs = int(os.getenv("NUM_EPOCHS"))
lr = float(os.getenv("LR"))
beta1 = float(os.getenv("BETA1"))


# Encoder
class StegEncoder(nn.Module):
    def __init__(self):
        super(StegEncoder, self).__init__()

        #cover image
        self.cover = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ngc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc) x 32 x 32
            nn.Conv2d(ngc, ngc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*2) x 16 x 16
            nn.Conv2d(ngc * 2, ngc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*4) x 8 x 8
            nn.Conv2d(ngc * 4, ngc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #state size. (ngc*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(ngc*8*4*4, nz),
        )

        #secret image
        self.secret = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ngc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc) x 32 x 32
            nn.Conv2d(ngc, ngc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*2) x 16 x 16
            nn.Conv2d(ngc * 2, ngc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*4) x 8 x 8
            nn.Conv2d(ngc * 4, ngc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #state size. (ngc*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(ngc*8*4*4, nz),
        )

        #generator for stego image
        self.stego = nn.Sequential(
            # input is 2*nz (latent vectors of cover and secret images concatenated)
            nn.ConvTranspose2d(2*nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )


    def forward(self, secret, cover):
        cover = self.cover(cover)
        secret = self.secret(secret)
        stego = torch.cat((cover, secret), 1)
        stego = stego.view(-1, 2*nz, 1, 1)
        stego = self.stego(stego)
        return stego
    
# Decoder
class StegDecoder(nn.Module):
    def __init__(self):
        super(StegDecoder, self).__init__()

        self.reverseStego = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ngc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc) x 32 x 32
            nn.Conv2d(ngc, ngc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*2) x 16 x 16
            nn.Conv2d(ngc * 2, ngc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*4) x 8 x 8
            nn.Conv2d(ngc * 4, ngc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #state size. (ngc*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(ngc*8*4*4, nz),
        )

        self.regenerate = nn.Sequential(
            # input is (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, stego):
        stego = self.reverseStego(stego)
        stego = stego.view(-1, nz, 1, 1)
        stego = self.regenerate(stego)
        return stego
    
# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.compress = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ngc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc) x 32 x 32
            nn.Conv2d(ngc, ngc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*2) x 16 x 16
            nn.Conv2d(ngc * 2, ngc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ngc*4) x 8 x 8
            nn.Conv2d(ngc * 4, ngc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            #state size. (ngc*8) x 4 x 4
            nn.Flatten(),
            nn.Linear(ngc*8*4*4, nz),
        )

        self.classify = nn.Sequential(
            nn.Linear(nz, nz / 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz / 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        image = self.compress(image)
        image = self.classify(image)
        return image
            
            

#Loss functions
a = int(os.getenv("a"))
b = int(os.getenv("b"))

class CoverMSELoss(nn.Module):
    def __init__(self):
        super(CoverMSELoss, self).__init__()

    def forward(self, cover, stego):
        return torch.mean((cover - stego) ** 2)
    
class SecretMSELoss(nn.Module):
    def __init__(self):
        super(SecretMSELoss, self).__init__()

    def forward(self, secret, stego_decoded):
        return torch.mean((secret - stego_decoded) ** 2)
    
class EncDecLoss(nn.Module):
    def __init__(self):
        super(EncDecLoss, self).__init__()

    def forward(self, cover, secret, stego, stego_decoded):
        return a * CoverMSELoss(cover, stego) + b * SecretMSELoss(secret, stego_decoded)
