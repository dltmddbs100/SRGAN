import os
import glob
import time
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import math
from math import log10

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, ToPILImage

from loss import GeneratorLoss, psnr
from dataset import TrainDataset, TestDataset
from model import Generator, Discriminator


# args
args={}
args['max_epochs'] = 50
args['batch_size'] = 4
args['data_path'] = "/content/drive/MyDrive/Dacon/LG_Denoising/"
args['weight_path'] = "/content/drive/MyDrive/코드정리(블로그)/논문정리/GAN/SRGAN/weights/"

# dataset
train_data=TrainDataset(args['data_path'])
test_data=TestDataset(args['data_path'])

# dataloader
train_loader=DataLoader(train_data, num_workers=4, batch_size=4, shuffle=True,pin_memory=True)
test_loader=DataLoader(test_data, num_workers=4, batch_size=4, shuffle=False,pin_memory=True)

# train
netG = Generator(2)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
  netG.cuda()
  netD.cuda()
  generator_criterion.cuda()
    
optimizerG = optim.Adam(netG.parameters(),lr=1e-4)
optimizerD = optim.Adam(netD.parameters(),lr=1e-4)

netG = Generator(2)
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
  netG.cuda()
  netD.cuda()
  generator_criterion.cuda()
    
optimizerG = optim.Adam(netG.parameters(),lr=1e-4, weight_decay=1e-5)
optimizerD = optim.Adam(netD.parameters(),lr=1e-4, weight_decay=1e-5)

for epoch_i in range(0, args['max_epochs']):

  print("")
  print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args['max_epochs']))

  total_train_gloss = 0 
  total_train_dloss = 0
  total_g_score=0
  total_d_score=0
  
  t0 = time.time()
  total_batch=len(train_loader)


  netG.train()
  netD.train()

  for i, batch in enumerate(train_loader):
    train, target = batch[0].cuda(), batch[1].cuda()

    ############################
    # (1) Update D network: maximize D(x)-1-D(G(z))
    ###########################
    fake_img = netG(train)

    netD.zero_grad()
    real_out = netD(target).mean()
    fake_out = netD(fake_img).mean()

    d_loss = 1 - real_out + fake_out    
    d_loss.backward(retain_graph=True)

    optimizerD.step()


    ############################
    # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
    ###########################
    netG.zero_grad()
    
    fake_img = netG(train)
    fake_out = netD(fake_img).mean()

    g_loss = generator_criterion(fake_out, fake_img, target)    
    g_loss.backward()

    optimizerG.step()

    training_time = time.time() - t0

    total_train_dloss += d_loss.item() * args['batch_size']
    total_train_gloss += g_loss.item() * args['batch_size']

    print(f"\rTotal Batch {i+1}/{total_batch} , elapsed time : {training_time/(i+1):.1f}s , g_loss : {total_train_gloss/(i+1):.6f}, d_loss : {total_train_dloss/(i+1):.6f}" , end='')
  print('')

  
  ############################
  # (3) Validating
  ###########################
  netG.eval()

  total_valid_psnr = 0
  total_valid_mse = 0
  total_batch=len(test_loader)

  for i,batch in enumerate(test_loader):
    lr, hr = batch[0].cuda(), batch[1].cuda()

    with torch.no_grad():
      sr=netG(lr)

    batch_mse = ((sr - hr) ** 2).data.mean()
    total_valid_mse += batch_mse * args['batch_size']
    total_valid_psnr += psnr(hr,sr)

    print(f"\rTest Batch {i+1}/{total_batch} , mse : {total_valid_mse/(i+1):.6f}, psnr : {total_valid_psnr/(i+1):.6f}", end='')

  # show the result among test datasets samples
  lr = test_data.__getitem__(13)[0].cuda()
  sr=netG(lr.unsqueeze(0))
  display(ToPILImage()(torch.clamp(sr[0].data.cpu(),0,1)))

  if (epoch_i+1)%10==0:
    torch.save(netG.state_dict(), args['weight_path']+f'netG_epoch_{epoch_i+1}_upscale_2.pth')
