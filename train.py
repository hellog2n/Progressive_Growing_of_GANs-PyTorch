import argparse
import os
import copy
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import random

from model import *
from progressBar import printProgressBar
from utils import *
from mmdLoss import MMD_loss

from time import time



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='dataset/CT', help='directory containing the data')
parser.add_argument('--outd', default='Results', help='directory to save results')
parser.add_argument('--outf', default='Images', help='folder to save synthetic images')
parser.add_argument('--evalDir', type=str, default='EvalDir', help='folder to save Generated images')
parser.add_argument('--pggan', type=str, default='PGGAN', help='folder to save Generated images-PGGAN')
parser.add_argument('--outl', default='Losses', help='folder to save Losses')
parser.add_argument('--outm', default='Models', help='folder to save models')

parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batchSizes', type=list, default=[16, 16, 16, 16], help='list of batch sizes during the training')
parser.add_argument('--nch', type=int, default=4, help='base number of channel for networks')
parser.add_argument('--saveImageGroup', type=list, default=[1000, 2560], help='number of Saving Image Groups')
parser.add_argument('--seed', type=int, default=1212, help='number of the fixed random seed')
parser.add_argument('--BN', action='store_true', help='use BatchNorm in G and D')
parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')
parser.add_argument('--checkVal', type=int, default=9, help='Checking the Number Saving Images')

parser.add_argument('--n_iter', type=int, default=1, help='number of epochs to train before changing the progress')
parser.add_argument('--lambdaGP', type=float, default=10, help='lambda for gradient penalty')
parser.add_argument('--gamma', type=float, default=1, help='gamma for gradient penalty')
parser.add_argument('--e_drift', type=float, default=0.001, help='epsilon drift for discriminator loss')
parser.add_argument('--saveimages', type=int, default=1, help='number of epochs between saving image examples')
parser.add_argument('--savenum', type=int, default=64, help='number of examples images to save')
parser.add_argument('--savemodel', type=int, default=10, help='number of epochs between saving models')
parser.add_argument('--savemaxsize', action='store_true', help='save sample images at max resolution instead of real resolution')
parser.add_argument('--size', type=int, default=4, help='The name of the main folder')

opt = parser.parse_args()
print(opt)

# Set random seed for reproducibility
manualSeed = opt.seed   # default: 1212
print("Random Seed: ", manualSeed, "...")
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)  # if use multi-GPU
cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(manualSeed)
print("Check the Current Random Seed: ", torch.seed(), "...")



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_RES = opt.size # for 32x32 output - 3 / 64x64 output - 4 / 128x128 output - 5
image_size  = 2**(2+MAX_RES)

z1 = hypersphere(torch.randn(10, opt.nch * 32, 1, 1, device=DEVICE))
z2 = hypersphere(torch.randn(10, opt.nch * 32, 1, 1, device=DEVICE))
print('The z1 is ', z1)
print('The z2 is ', z2)


from torchvision import datasets

transform = transforms.Compose(
    [
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Resize((image_size, image_size)),
     transforms.Normalize((0.5, ), (0.5, ))
])



dataset = datasets.ImageFolder(root = opt.data,
                transform = transform)



# dataset = MNIST(opt.data, download=True, train=True, transform=transform)


# creating output folders
if not os.path.exists(os.path.join(opt.outd)):
    os.makedirs(os.path.join(opt.outd))
for f in [opt.outf, opt.outl, opt.outm, 'MMD_Loss']:
    if not os.path.exists(os.path.join(opt.outd, f)):
        os.makedirs(os.path.join(opt.outd, f))
for i in ['Normalized-[-1,1]', 'Normalized-[0,1]']:
    if not os.path.exists(os.path.join(opt.outf, i)):
        os.makedirs(os.path.join(opt.outf, i))

# Model creation and init
G = Generator(max_res=MAX_RES, nch=opt.nch, nc=1, bn=opt.BN, ws=opt.WS, pn=opt.PN).to(DEVICE)
D = Discriminator(max_res=MAX_RES, nch=opt.nch, nc=1, bn=opt.BN, ws=opt.WS).to(DEVICE)
mmd_loss = MMD_loss().to(DEVICE)
if not opt.WS:
    # weights are initialized by WScale layers to normal if WS is used
    G.apply(weights_init)
    D.apply(weights_init)
Gs = copy.deepcopy(G)

optimizerG = Adam(G.parameters(), lr=1e-3, betas=(0, 0.99))
optimizerD = Adam(D.parameters(), lr=1e-3, betas=(0, 0.99))

GP = GradientPenalty(opt.batchSizes[0], opt.lambdaGP, opt.gamma, device=DEVICE)

epoch = 0
global_step = 0
total = 2
d_losses = np.array([])
d_losses_W = np.array([])
g_losses = np.array([])
mmdL = []
P = Progress(opt.n_iter, MAX_RES, opt.batchSizes)

z_save = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))

P.progress(epoch, 1, total)
GP.batchSize = P.batchSize
# Creation of DataLoader
data_loader = DataLoader(dataset,
                         batch_size=P.batchSize,
                         shuffle=True,
                         num_workers=opt.workers,
                         drop_last=True,
                         pin_memory=True)
dataarr=[]
checkingLoss_D = []
checkingLoss_G = []
while True:
    t0 = time()

    lossEpochG = []
    lossEpochD = []
    lossEpochD_W = []

    G.train()
    # cudnn.benchmark = True

    P.progress(epoch, 1, total)

    if P.batchSize != data_loader.batch_size:
        # update batch-size in gradient penalty
        GP.batchSize = P.batchSize
        # modify DataLoader at each change in resolution to vary the batch-size as the resolution increases
        data_loader = DataLoader(dataset,
                                 batch_size=P.batchSize,
                                 shuffle=True,
                                 num_workers=opt.workers,
                                 drop_last=True,
                                 pin_memory=True)

    total = len(data_loader)

    for i, (images, _) in enumerate(data_loader):
        P.progress(epoch, i + 1, total + 1)
        global_step += 1

        # Build mini-batch
        images = images.to(DEVICE)
        images = P.resize(images)

        # ============= Train the discriminator =============#

        # zeroing gradients in D
        D.zero_grad()
        # compute fake images with G
        z = hypersphere(torch.randn(P.batchSize, opt.nch * 32, 1, 1, device=DEVICE))
        with torch.no_grad():
            fake_images = G(z, P.p)

        # compute scores for real images
        D_real = D(images, P.p)
        D_realm = D_real.mean()

        # compute scores for fake images
        D_fake = D(fake_images, P.p)
        D_fakem = D_fake.mean()

        # compute gradient penalty for WGAN-GP as defined in the article
        gradient_penalty = GP(D, images.data, fake_images.data, P.p)

        # prevent D_real from drifting too much from 0
        drift = (D_real ** 2).mean() * opt.e_drift

        # Backprop + Optimize
        d_loss = D_fakem - D_realm
        d_loss_W = d_loss + gradient_penalty + drift
        d_loss_W.backward()
        optimizerD.step()

        lossEpochD.append(d_loss.item())
        lossEpochD_W.append(d_loss_W.item())


        # =============== Train the generator ===============#

        G.zero_grad()

        z = hypersphere(torch.randn(P.batchSize, opt.nch * 32, 1, 1, device=DEVICE))
        fake_images = G(z, P.p)
        # compute scores with new fake images
        G_fake = D(fake_images, P.p)
        G_fakem = G_fake.mean()
        # no need to compute D_real as it does not affect G
        g_loss = -G_fakem

        # Optimize
        g_loss.backward()
        optimizerG.step()

        lossEpochG.append(g_loss.item())

        # Checking the MMD Loss
        dataarr = images.to(DEVICE)
        fakearr = fake_images
        mmd = mmd_loss(fakearr, dataarr)
        mmdL.append(mmd.item())

        # update Gs with exponential moving average
        exp_mov_avg(Gs, G, alpha=0.999, global_step=global_step)

        printProgressBar(i + 1, total + 1,
                         length=20,
                         prefix=f'Epoch {epoch} ',
                         suffix=f', d_loss: {d_loss.item():.3f}'
                                f', d_loss_W: {d_loss_W.item():.3f}'
                                f', GP: {gradient_penalty.item():.3f}'
                                f', progress: {P.p:.2f}'
                                f', MMD_Loss: {mmd.item()} ')
        checkingLoss_G.append(d_loss.item())
        checkingLoss_D.append(g_loss.item())
    printProgressBar(total, total,
                     done=f'Epoch [{epoch:>3d}]  d_loss: {np.mean(lossEpochD):.4f}'
                          f', d_loss_W: {np.mean(lossEpochD_W):.3f}'
                          f', progress: {P.p:.2f}, time: {time() - t0:.2f}s'
                            f', Mean MMD_Loss: {np.mean(mmd.item())}'
                     )

    d_losses = np.append(d_losses, lossEpochD)
    d_losses_W = np.append(d_losses_W, lossEpochD_W)
    g_losses = np.append(g_losses, lossEpochG)

    np.save(os.path.join(opt.outd, opt.outl, 'd_losses.npy'), d_losses)
    np.save(os.path.join( opt.outd, opt.outl, 'd_losses_W.npy'), d_losses_W)
    np.save(os.path.join( opt.outd, opt.outl, 'g_losses.npy'), g_losses)

    # cudnn.benchmark = False


    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(checkingLoss_G, label="G")
    plt.plot(checkingLoss_D, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(opt.outd, opt.outl, f'Gen Dis Loss Epoch_{epoch}.png'), dpi=200, bbox_inches='tight')
    plt.clf()

    if not (epoch + 1) % opt.saveimages:
        # plotting loss values, g_losses is not plotted as it does not represent anything in the WGAN-GP
        ax = plt.subplot()
        ax.plot(np.linspace(0, epoch + 1, len(d_losses)), d_losses, '-b', label='d_loss', linewidth=0.1)
        ax.legend(loc=1)
        ax.set_xlabel('epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Progress: {P.p:.2f}')
        plt.savefig(os.path.join( opt.outd, opt.outl, f'Discriminator Epoch_{epoch}.png'), dpi=200, bbox_inches='tight')
        plt.clf()

        plt.figure(figsize=(10, 5))
        plt.title("Fake and Real MMD Loss During Training")
        plt.plot(mmdL, label="MMD Loss")
        plt.xlabel("iterations")
        plt.ylabel("MMD Loss")
        plt.legend()
        plt.savefig(os.path.join( opt.outd, 'MMD_Loss', f'MMD Loss Epoch_{epoch}.png'), dpi=200, bbox_inches='tight')
        plt.clf()

        # Save sampled images with Gs
        Gs.eval()
        #z_evalsave = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))

        # 매 에폭마다 예시 이미지 생성
        with torch.no_grad():
            fake_images = Gs(z_save, P.p)

        save_image(fake_images,
                   os.path.join( opt.outd, opt.outf, f'fake_images-{epoch:04d}-p{P.p:.2f}.png'),
                    nrow=8, pad_value=0,
                   normalize=False, range=(-1, 1))
        save_image(fake_images,
                   os.path.join(opt.outd, opt.outf, f'fake_images-Normalize-{epoch:04d}-p{P.p:.2f}.png'),
                   nrow=8, pad_value=0,
                   normalize=True, range=(-1, 1))


    if P.p >= P.pmax and not epoch % opt.savemodel:
        torch.save(G, os.path.join( opt.outd, opt.outm, f'G_nch-{opt.nch}_epoch-{epoch}.pth'))
        torch.save(D, os.path.join( opt.outd, opt.outm, f'D_nch-{opt.nch}_epoch-{epoch}.pth'))
        torch.save(Gs, os.path.join( opt.outd, opt.outm, f'Gs_nch-{opt.nch}_epoch-{epoch}.pth'))

    if (epoch > 100 and (epoch-100) % 20 == 0) or epoch == checkVal:
        # 테스트  에폭마다 예시 이미지 생성
        with torch.no_grad():
            for num in opt.saveImageGroup:
                z = hypersphere(torch.randn(num, opt.nch * 32, 1, 1, device=DEVICE))
                fake_images = Gs(z, P.p)
                for i, img in enumerate(fake_images):
                    # Normalize [0, 1]
                    save_image(img,
                   os.path.join(opt.outd, opt.outf,  'Normalized-[0,1]', f'fake_images-{epoch:04d}-p{P.p:.2f}_{i}.png'),
                   nrow=8, pad_value=0,
                   normalize=True)
                    # Normalize [-1, 1]
                    save_image(img,
                   os.path.join(opt.outd, opt.outf, 'Normalized-[-1,1]', f'fake_images-{epoch:04d}-p{P.p:.2f}_{i}.png'),
                   nrow=8, pad_value=0,
                   normalize=False, range=(-1, 1))

    epoch += 1

