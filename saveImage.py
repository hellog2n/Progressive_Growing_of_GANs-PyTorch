import torch
from model import *
from progressBar import printProgressBar
from utils import *
from torchvision.utils import save_image

size = 128
model = 'PGGAN'
nch = 4
MAX_RES = 5
savenum = 64
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
G = torch.load('Result-PGGAN-128/Models/Gs_nch-4_epoch-140.pth')

G.eval()
size = 0
size  = 2**(2+MAX_RES)


with torch.no_grad():
  z_noise = hypersphere(torch.randn(savenum, nch * 32, 1, 1, device=DEVICE))
  output = G(z_noise)

  # Normalize Settings
  save_image(output,
                   f'fake_images-{model}-{size}.png',
                    nrow=8, pad_value=0,
                   normalize=False, range=(-1, 1))

def set_seed(self, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)