from __future__ import print_function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from utils import TextureDataset, setNoise
#, learnedWN
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys
from network import weights_init,Discriminator,calc_gradient_penalty,NetG
from config import opt,bMirror,nz,nDep,criterion
import time
import numpy as np
import csv 
import torch.nn as nn
  

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

######## Data is expected to be in range [0,1] for R,G,B and in range [0,100] for depth
######## normalization is adapted to normalize depth values (that are in the [0,100] range) to [-1,1] as required by the genereator network.
canonicT=[transforms.RandomCrop(opt.imageSize),transforms.Normalize((0.5, 0.5, 0.5, 50), (0.5, 0.5, 0.5, 50))]#,transforms.ToTensor()
mirrorT= []
if bMirror:
    mirrorT += [transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()]
transformTex=transforms.Compose(mirrorT+canonicT)
dataset = TextureDataset(opt.texturePath,transformTex,opt.textureScale)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

N=0
ngf = int(opt.ngf)
ndf = int(opt.ndf)

desc="fc"+str(opt.fContent)+"_ngf"+str(ngf)+"_ndf"+str(ndf)+"_dep"+str(nDep)+"-"+str(opt.nDepD)

if opt.WGAN:
    desc +='_WGAN'
if opt.LS:
        desc += '_LS'
if bMirror:
    desc += '_mirror'
if opt.textureScale !=1:
    desc +="_scale"+str(opt.textureScale)
    

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print ("device",device)    

##### parallel netD
#netD = Discriminator(ndf, opt.nDepD, bSigm=not opt.LS and not opt.WGAN, ncIn=4)
#netD = nn.parallel.DistributedDataParallel(Discriminator(ndf, opt.nDepD, bSigm=not opt.LS and not opt.WGAN, ncIn=4))
#netD.to(device)
netD = Discriminator(ndf, opt.nDepD, bSigm=False, ncIn=4)

################################## CHANGED DEVICES
netG =NetG(ngf, nDep, nz, 4)
#####moved up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device",device)
################this is new
#netG = nn.parallel.DistributedDataParallel(NetG(ngf, nDep, nz, 4))
#netG.to(device)

Gnets=[netG]
#if opt.z:
#    Gnets += [learnedWN]

for net in [netD] + Gnets:
    try:
        net.apply(weights_init)
    except Exception as e:
        print (e,"weightinit")
    pass
    net=net.to(device)
    print(net)

NZ = opt.imageSize//2**nDep
noise = torch.FloatTensor(opt.batchSize, nz, NZ,NZ)
fixnoise = torch.FloatTensor(opt.batchSize, nz, NZ*4,NZ*4)

real_label = 1
fake_label = 0

noise=noise.to(device)
fixnoise=fixnoise.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))#netD.parameters()
optimizerU = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lr, betas=(opt.beta1, 0.999))

out_csv = '%s/losses.csv' % (opt.outputFolder)
fieldnames = ['epoch', 'D_x', 'D_G_z1', 'D_G_z2']
with open(out_csv, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fieldnames)
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            t0 = time.time()
            sys.stdout.flush()
            # train with real
            netD.zero_grad()
            text, _ = data
            text=text.to(device)
            output = netD(text)
            errD_real = criterion(output, output.detach()*0+real_label)
            errD_real.backward()
            D_x = output.mean()

            # train with fake
            noise=setNoise(noise)
            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, output.detach()*0+fake_label)
            errD_fake.backward()

            D_G_z1 = output.mean()
            errD = errD_real + errD_fake
            if opt.WGAN:
                gradient_penalty = calc_gradient_penalty(netD, text, fake[:text.shape[0]])##for case fewer text images
                gradient_penalty.backward()

            optimizerD.step()
            if i >0 and opt.WGAN and i%opt.dIter!=0:
                continue ##critic steps to 1 GEN steps

            for net in Gnets:
                net.zero_grad()

            noise=setNoise(noise)
            fake = netG(noise)
            output = netD(fake)
            loss_adv = criterion(output, output.detach()*0+real_label)
            D_G_z2 = output.mean()
            errG = loss_adv
            errG.backward()
            optimizerU.step()

            print('[%d/%d][%d/%d] D(x): %.4f D(G(z)): %.4f / %.4f time %.4f'
                  % (epoch, opt.niter, i, len(dataloader),D_x, D_G_z1, D_G_z2,time.time()-t0))

            ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
            if i % 100 == 0:
                csvwriter.writerow([epoch, f'{D_x}', f'{D_G_z1}', f'{D_G_z2}'])
                vutils.save_image(text,    '%s/real_textures.png' % opt.outputFolder,  normalize=True)
                vutils.save_image(fake,'%s/generated_textures_%03d_%s.png' % (opt.outputFolder, epoch,desc),normalize=False)
                np.save('%s/generated_textures_%03d_%s.npy' % (opt.outputFolder, epoch,desc) ,fake.cpu().detach().numpy())

                fixnoise=setNoise(fixnoise)

                vutils.save_image(fixnoise.view(-1,1,fixnoise.shape[2],fixnoise.shape[3]), '%s/noiseBig_epoch_%03d_%s.png' % (opt.outputFolder, epoch, desc),normalize=False)

                netG.eval()
                with torch.no_grad():
                    fakeBig=netG(fixnoise)

                vutils.save_image(fakeBig,'%s/big_texture_%03d_%s.png' % (opt.outputFolder, epoch,desc),normalize=False)
                netG.train()

                ##OPTIONAL
                ##save/load model for later use if desired
                outModelName_G = '%s/netG_epoch_%d_%s.pth' % (opt.outputFolder, epoch*0,desc)
                outModelName_D = '%s/netD_epoch_%d_%s.pth' % (opt.outputFolder, epoch*0,desc)
                torch.save(netG.state_dict(),outModelName_G)
                torch.save(netD.state_dict(),outModelName_D)
                netG.load_state_dict(torch.load(outModelName_G))
