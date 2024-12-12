import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model
import torchvision.transforms as transforms


device = 'cpu'

def train(img_dataloader, enc, dec, disc, encDecOptim, discOptim, encDecCriterion, criterion, num_epochs, mix_coeff, batch_size):
    #update discriminator 4 times for every generator training

    #put in training mode
    enc.train()
    dec.train()
    disc.train()

    for epoch in range(num_epochs):
        for i, data in enumerate(img_dataloader, 0):
            if i % 5 == 0:
                #update generator
                encDecOptim.zero_grad()
                cover, secret = data
                cover = cover.to(device)
                secret = secret.to(device)
                stego = enc(secret, cover)
                stego_decoded = dec(stego)
                gen_loss = disc(stego)
                loss = encDecCriterion(cover, secret, stego, stego_decoded) + mix_coeff * gen_loss
                loss.backward()
                encDecOptim.step()
            else:
                #update discriminator
                discOptim.zero_grad()
                cover, secret = data
                cover = cover.to(device)
                secret = secret.to(device)
                stego = enc(secret, cover)
                real = torch.zeros((batch_size, 1), device=device)
                fake = torch.ones((batch_size, 1), device=device)
                real_loss = criterion(disc(cover), real)
                fake_loss = criterion(disc(stego), fake)
                loss = real_loss + fake_loss
                loss.backward()
                discOptim.step()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.pairs = []
        self.createPairs()
        self.transform = transform
    
    def createPairs(self):
        #create pairs of images for cover and secret
        for i in range(len(self.images)):
            for j in range(len(self.images)):
                self.pairs.append((self.images[i], self.images[j]))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        cover = self.images[self.pairs[idx][0]]
        secret = self.images[self.pairs[idx][1]]
        self.transform(cover)
        self.transform(secret)
        return cover, secret
    
transform = transforms.Compose([
    transforms.Resize(image_size),
    # transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

from PIL import Image
import os

img_dir = model.root_dir

images = [Image.open(img_dir + img) for img in os.listdir(img_dir)]


train_dataloader = torch.utils.data.DataLoader(Dataset(images), batch_size=model.batch_size, shuffle=True, num_workers=model.workers)

enc = model.StegEncoder().to(device)
dec = model.StegDecoder().to(device)
disc = model.Discriminator().to(device)

encDecOptim = torch.optim.Adam(enc.parameters(), lr=model.lr, betas=(model.beta1, 0.999))
discOptim = torch.optim.Adam(disc.parameters(), lr=model.lr, betas=(model.beta1, 0.999))

enc.apply(model.weights_init)
dec.apply(model.weights_init)
disc.apply(model.weights_init)

train(train_dataloader, enc, dec, disc, encDecOptim, discOptim, model.encDecCriterion, model.criterion, model.num_epochs, 0.5)

#save the model
torch.save(enc.state_dict(), "encoder.pth")
torch.save(dec.state_dict(), "decoder.pth")
torch.save(disc.state_dict(), "discriminator.pth")