import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

cuda = True


class Encoder(nn.Module):
    def __init__(self, io_size, n_units, n_layers):
        super(Encoder, self).__init__()
        layers = []

        pre_size = io_size
        for i in range(n_layers):
            post_size = n_units
            layer = nn.Linear(pre_size, post_size)
            activation = nn.Tanh()
            if cuda:
                layer = layer.cuda()
            layers.append(layer)
            layers.append(activation)
            pre_size = post_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        return self.layers(x)
            
class Decoder(nn.Module):
    def __init__(self, n_units, n_layers, n_latent):
        super(Decoder, self).__init__()
        layers = []
        activation = nn.Tanh()

        pre_size = n_latent
        for i in range(n_layers):
            post_size = n_units
            layer = nn.Linear(pre_size, post_size)
            activation = nn.Tanh()
            if cuda:
                layer = layer.cuda()
            layers.append(layer)
            layers.append(activation)
            pre_size = post_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous()
        return self.layers(x)

class MMVAE(nn.Module):
    def __init__(self, i_dim, i_units, i_layers,
                 l_dim, embed_dim, l_units, l_layers, n_latent):
        super(MMVAE, self).__init__()
        self.embed_dim = embed_dim
        self.n_latent = n_latent

        #encoding
        self.i_encode = Encoder(i_dim, i_units, i_layers)
        self.l_embed = nn.Embedding(l_dim, embed_dim)
        if cuda:
            self.l_embed = self.l_embed.cuda()
        self.l_encode = Encoder(l_dim, l_units, l_layers)

        #sampler
        self.i_mean = nn.Linear(i_units, n_latent)
        self.i_logvar = nn.Linear(i_units, n_latent)
        self.l_mean = nn.Linear(l_units, n_latent)
        self.l_logvar = nn.Linear(l_units, n_latent)
        if cuda:
            self.i_mean = self.i_mean.cuda()
            self.i_logvar = self.i_logvar.cuda()
            self.l_mean = self.l_mean.cuda()
            self.l_logvar = self.l_logvar.cuda()

        #decoder
        self.i_decode = Decoder(i_units, i_layers, n_latent) 
        self.l_decode = Decoder(l_units, l_layers, n_latent)
        self.i_project = nn.Linear(i_units, i_dim)
        self.l_project = nn.Linear(l_units, l_dim)
        if cuda:
            self.i_project = self.i_project.cuda()
            self.l_project = self.l_project.cuda()

    def mix_gauss(self, gaussians):
        m_div_v = None
        div_v = None
        for m, lv in gaussians:
            v = lv.exp()
            if m_div_v is None:
                m_div_v = m / v
                div_v = 1 / v
            else:
                m_div_v += m / v
                div_v += 1 / v
        mu = m_div_v * (1 / div_v)
        v = 1 / div_v
        logvar = v.log()
        return mu, logvar
            
    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.data.size()))
        if cuda:
            eps = eps.cuda()
        z = mu + eps * logvar.mul(0.5).exp()
        return z
        
    def forward(self, ix, lx, mode=0):
        """
        mode// 0:both, 1:image, 2:label
        """
        batchsize = int(ix.shape[0])
        io = None
        lo = None
        mu = np.zeros((batchsize, self.n_latent), dtype=np.float32)
        mu = Variable(torch.from_numpy(mu))
        logvar = np.zeros((batchsize, self.n_latent), dtype=np.float32)
        logvar = Variable(torch.from_numpy(logvar))
        if cuda:
            mu = mu.cuda()
            logvar = logvar.cuda()
        gaussians = [(mu, logvar)]

        """encode"""
        if mode < 2:
            ih = self.i_encode(ix)
            i_mu = self.i_mean(ih)
            i_logvar = self.i_logvar(ih)
            gaussians.append((i_mu, i_logvar))
        if mode % 2 == 0:
            le = self.l_embed(lx.view(-1,1))
            le = le.view(batchsize, self.embed_dim)
            lh = self.l_encode(le)
            l_mu = self.l_mean(lh)
            l_logvar = self.l_logvar(lh)
            gaussians.append((l_mu, l_logvar))

        """sampling"""
        mu, logvar = self.mix_gauss(gaussians)
        z = self.reparameterize(mu, logvar)

        """decoding"""
        iy = self.i_decode(z)
        io = self.i_project(iy)
        sigmoid = nn.Sigmoid()
        io = sigmoid(io)
        ly = self.l_decode(z)
        lo = self.l_project(ly)

        return io, lo, mu, logvar, z
