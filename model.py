import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super(VariationalAutoencoder, self).__init__()
        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        
        # Decoder
        self.z_2hid = nn.Linear(z_dim,h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x) 
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        x = self.hid_2img(h)
        return torch.sigmoid(x)
        

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x_reconst = self.decode(z)
        return x_reconst, mu, sigma

if __name__ == '__main__':
    x = torch.randn(4, 28*28)
    vae = VariationalAutoencoder(input_dim=28*28)
    x_reconst, mu, sigma = vae(x)
    print(x_reconst.shape)