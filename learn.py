import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
#from torch.utils.data.dataset import Dataset
#from torch.utils.data import DataLoader

from IPython import embed

from models import MMVAE

cuda = True

def main():
    # net configuration
    image_dim = 784
    image_units = 512
    image_layers = 2
    label_dim = 10
    embed_dim = 10
    label_units = 512
    label_layers = 2
    n_latent = 64

    # training configuration
    n_epochs = 500
    save_interval = 100
    visualize_interval = 10
    train_batchsize = 100
    test_batchsize = 500

    # data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=train_batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.ToTensor()),
        batch_size=test_batchsize, shuffle=True)

    # model, loss, optimizer
    model = MMVAE(image_dim, image_units, image_layers, 
                  label_dim, embed_dim, label_units,
                  label_layers, n_latent)
    cel = nn.CrossEntropyLoss(size_average=False)
    if cuda:
        model.cuda()
        cel = cel.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # training
    errorlog = []
    if not os.path.exists("results"):
        os.mkdir("results")
    start = time.time()
    for epoch in range(n_epochs):
        beta = min(1.0, float(epoch)/200.0)
        for i, (ix, lx) in enumerate(train_loader):
            mode = i % 3
            if cuda:
                ix = ix.cuda()
                lx = lx.cuda()
            ix = Variable(ix)
            ix = ix.view(-1, image_dim)
            lx = Variable(lx)
            optimizer.zero_grad()
            iy, ly, mu, logvar, z = model(ix, lx, mode)

            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = beta * kld
            if mode < 2:
                i_loss = F.binary_cross_entropy(iy, ix, size_average=False) 
                loss += i_loss
            if mode % 2 == 0:
                l_loss = cel(ly, lx)
                loss += l_loss
            loss.backward()
            optimizer.step()
            #if mode == 0:
            #    print "Total:{}, Image:{}, Label:{}, KL:{}".format(float(loss.data)/train_batchsize,
            #                                                       float(i_loss.data)/train_batchsize,
            #                                                       float(l_loss.data)/train_batchsize,
            #                                                       float(kld.data)/train_batchsize)

        ix, lx = next(iter(test_loader))
        if cuda:
            ix = ix.cuda()
            lx = lx.cuda()
        ix = Variable(ix)
        ix = ix.view(-1, image_dim)
        lx = Variable(lx)
        iy, ly, mu, logvar, z = model(ix, lx, 0)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / test_batchsize
        i_loss = F.binary_cross_entropy(iy, ix, size_average=False) / test_batchsize
        l_loss = cel(ly, lx) / test_batchsize
        loss = beta*kld + i_loss + l_loss

        end = time.time()
        print "================="
        print "epoch:{}, time:{}".format(epoch, end-start)
        print "Total:{}, Image:{}, Label:{}, KL:{}".format(float(loss.data),
                                                           float(i_loss.data),
                                                           float(l_loss.data),
                                                           float(kld.data))
        errorlog.append([loss.data, i_loss.data, l_loss.data, kld.data])
        start = end

        if (epoch+1) % visualize_interval == 0:
            for i, (ix, lx) in enumerate(test_loader):
                if i > 10:
                    break
                elif i == 10:
                    mode = 3
                else:
                    mode = 2
                    lx = np.zeros(test_batchsize, dtype=np.int64)
                    lx = lx+i
                    lx = torch.from_numpy(lx)
                if cuda:
                    ix = ix.cuda()
                    lx = lx.cuda()
                ix = Variable(ix)
                ix = ix.view(-1, image_dim)
                lx = Variable(lx)
                iy, ly, mu, logvar, z = model(ix, lx, mode)
                #np.save("results/z%.4d_%.2d.npy" %(epoch, i+1), np.array(z.data))
                iy = iy.view(-1,1,28,28)
                save_image(iy[:20].data.cpu(), "results/reconstructed%.4d_sample%.2d.png" %(epoch+1, i), nrow=10)

        if (epoch+1) % save_interval == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, "results/model%.8d.pt"%(epoch+1))
    np.savetxt("results/error.log", np.array(errorlog))

if __name__ == "__main__":
    main()
