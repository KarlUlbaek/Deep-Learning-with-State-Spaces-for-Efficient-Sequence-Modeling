import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

ROOT = "../data/cifar10/"

class Cifar10seq(Dataset):
   def __init__(self, train=True, d="cuda"):
      data = torchvision.datasets.CIFAR10(root=ROOT, train=train, download=True)
      x = torch.from_numpy(data.data)
      x = torch.flatten(x, 1) / x.max()
      self.x = torch.unsqueeze(x, -1).to(d)
      self.y = torch.tensor(data.targets).to(d)


   def __len__(self):
      return self.x.shape[0]

   def __getitem__(self, idx):
      return self.x[idx], self.y[idx]



b = 128
cifar_dataloader_train = DataLoader(dataset=Cifar10seq(train=True),
                                    shuffle=True,
                                    batch_size=b)
cifar_dataloader_train = DataLoader(dataset=Cifar10seq(train=True),
                                    shuffle=True,
                                    batch_size=b)
lr = 1e-4
n_layers = 4
d_data = 1
d_model = 128
d_state = 16
L = next(iter(cifar_dataloader_train))[0].shape[1]
d = "cuda"
classes = 10

from mamba_fork.mamba_ssm.models.mixer_seq_simple import MixerModel as MambaNN
s6NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, 
               d_out = classes, discrete=False, fused_add_norm=False).to(d)

s4NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, 
               d_out = classes, discrete=False, fused_add_norm=False,
               s4={"mode":"dplr", "hippo_init":"legs"}).to(d)

s4dNN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, 
               d_out = classes, discrete=False, fused_add_norm=False,
               s4={"mode":"diag", "hippo_init":"legs"}).to(d)

model = s6NN
#loss = CrossEntropyLoss()
opt = AdamW(model.parameters(), lr=lr)


n_epochs = 100

p_bar = tqdm.tqdm(range(n_epochs))
for epoch_dix in p_bar:
   for batch_idx, (x, y) in (enumerate(cifar_dataloader_train)):
      pred = model(x)
      last_pred = pred[:,-1,:]
      #print(last_pred.shape)
      loss = torch.nn.functional.cross_entropy(last_pred, y)
      loss.backward()
      opt.step()
      opt.zero_grad()
      acc = torch.mean((torch.argmax(last_pred, dim=-1) == y).to(float))
      p_bar.set_postfix({'loss': loss.cpu().item(), "acc":acc.cpu().item()})





































