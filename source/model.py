import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import train_test_split

class Encoder(nn.Module):
    
    def __init__(self, seq_len, aa_var, zdim, alpha):
        super(Encoder, self).__init__()
        
        self.q  = seq_len * aa_var
        self.hsize=int(1.5*self.q)
        #self.en_mu = nn.Linear(self.hsize, d)
        #self.en_std = nn.Linear(self.hsize, d) 
        
        self.model = nn.Sequential(
            #encoder layer 1
            nn.Linear(self.q, self.hsize),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(p=0.3),
            
            #encoder layer 2
            nn.Linear(self.hsize, self.hsize),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm1d(self.hsize), # BN1
            
            #encoder layer 3
            nn.Linear(self.hsize, self.hsize),
            nn.LeakyReLU(alpha, inplace=True),
            #nn.BatchNorm1d(self.hsize), # BN1
            
            nn.Linear(self.hsize, zdim)
        )
    def forward(self, x):
        x = x.view(x.size(0), self.q)
        return self.model(x)

    
class Decoder(nn.Module):
    
    def __init__(self, seq_len, aa_var, zdim, alpha):
        super(Decoder, self).__init__()
        
        self.q = seq_len * aa_var
        self.zdim = zdim
        self.hsize=int(1.5*self.q)
        
        self.model = nn.Sequential(
            #decoder layer 1
            nn.Linear(zdim, self.hsize),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm1d(self.hsize), #BN2
            
            #decoder layer 2
            nn.Linear(self.hsize, self.hsize),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(p=0.3),
            
            #decoder layer 3
            nn.Linear(self.hsize, self.hsize),
            nn.LeakyReLU(alpha, inplace=True),
            nn.BatchNorm1d(self.hsize), 
            
            nn.Linear(self.hsize, self.q),
            #nn.BatchNorm1d(self.q), #BNfinal
        )
    def forward(self, z):
        outputs = self.model(z)
        outputs = outputs.view(z.size(0), seq_len, aa_var)
        outputs = nn.Softmax(dim = 2)(outputs)
        return outputs


class MMD_VAE(nn.Module):
    def __init__(self, zdims, seq_len, aa_var, alpha):
        super(MMD_VAE, self).__init__()
        self.zdims = zdims
        self.seq_len = seq_len
        self.aa_var = aa_var
        self.alpha = alpha
        self.encoder = Encoder(self.seq_len, self.aa_var, self.zdims, self.alpha)
        self.decoder = Decoder(self.seq_len, self.aa_var, self.zdims, self.alpha)
    
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return z, recon_x
    

def loss_function(recon_x, x, z):
    batch_size = x.size(0)
    zdim = z.size(1)
    true_samples = torch.randn(batch_size, zdim, requires_grad = False).to(device)

    loss_MMD = compute_mmd(true_samples, z)
    loss_REC = (recon_x - x).pow(2).mean()

    return loss_REC + 2*loss_MMD, loss_REC, loss_MMD

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd