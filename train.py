import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from model import VariationalAutoencoder

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28*28
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

# dataset loading
dataset = torchvision.datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoencoder(input_dim=INPUT_DIM, h_dim=H_DIM, z_dim=Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss(reduction='sum')

# training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # forward
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconst, mu, sigma = model(x)

        # loss
        reconst_loss = loss_fn(x_reconst, x)
        kl_div = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        loss = reconst_loss + kl_div

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        


model = model.to("cpu")

def inference(digit, num_examples = 1):
    images = []
    idx = 0
    for x,y in dataset:
        if y==idx:
            images.append(x)
            idx+=1
        if idx == 10:
            break
    encodings_digit = []
    for d in range (10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1,784))
        encodings_digit.append((mu,sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1,1,28,28)
        save_image(out, f"generated {digit} ex_{example}/png")

for idx in range(10):
    inference(idx, num_examples=1)
    