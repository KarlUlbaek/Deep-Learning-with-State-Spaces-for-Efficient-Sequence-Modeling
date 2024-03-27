import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from einops import rearrange

import os
import sys
sys.path.append(os.getcwd())
ROOT = "../../data/cifar10/"

class Cifar10seq(Dataset):
   def __init__(self, train=True, d="cuda"):
      data = torchvision.datasets.CIFAR10(root=ROOT, train=train, download=True)
      x = torch.from_numpy(data.data).to(torch.float)
      x = rearrange(x, "a b c d -> a (b c) d")
      x = x / x.max()
      self.x = x.to(d)
      self.y = torch.tensor(data.targets).to(d).to(torch.long)

   def __len__(self):
      return self.x.shape[0]

   def __getitem__(self, idx):
      return self.x[idx], self.y[idx]



b = 64
num_workers = 0
cifar_dataloader_train = DataLoader(dataset=Cifar10seq(train=True),
                                    shuffle=True,
                                    batch_size=b, num_workers=num_workers)
cifar_dataloader_test = DataLoader(dataset=Cifar10seq(train=False),
                                    shuffle=False,
                                    batch_size=b,
                                    num_workers=num_workers)

n_layers = 4
d_data = 3
d_model = 128#1028//2
d_state = 16
dropout = 0.1
L = next(iter(cifar_dataloader_train))[0].shape[1]
d = "cuda"
classes = 10
classification = True

if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
   fast = False
else:
   fast = True

#sys.path.append(os.path.join(os.getcwd(), "src/s4_fork"))
#sys.path.append(os.path.join(os.getcwd(), "src/s4_playground"))
#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11",sys.path)
from mamba_fork.mamba_ssm.models.mixer_seq_simple import MixerModel as MambaNN
s6NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, dropout=dropout,
               d_out = classes, discrete=False, fused_add_norm=fast, rms_norm=fast,
               classification=classification).to(d)

s4NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, dropout=dropout,
               d_out = classes, discrete=False, fused_add_norm=fast, rms_norm=fast,
               s4={"mode":"dplr", "hippo_init":"legs"}, classification=classification).to(d)

s4dNN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, dropout=dropout,
                d_out = classes, discrete=False, fused_add_norm=fast, rms_norm=fast,
                s4={"mode":"diag", "hippo_init":"legs"}, classification=classification).to(d)

model = s6NN

#from s4_fork.example import model
print(model)
#print(model)
#opt = AdamW(model.parameters(), lr=lr, foreach=True)
n_epochs = 100

from misc import setup_optimizer
lr = 3e-3
lr_scale = 0.1
opt, sched = setup_optimizer(model, opt=AdamW, lr=lr, lr_scale=lr_scale, epochs=n_epochs)


print("trainable params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))
p_bar = tqdm.tqdm(enumerate(cifar_dataloader_train), unit="batch", total=len(cifar_dataloader_train))
inf_dict = {}
for epoch_dix in range(n_epochs):
   correct, tot = 0, 0
   for batch_idx, (x, y) in p_bar:
      x, y = x.to(d), y.to(d)
      last_pred = model(x)
      loss = torch.nn.functional.cross_entropy(last_pred, y)
      loss.backward()
      opt.step()
      opt.zero_grad()
      correct += torch.sum((torch.argmax(last_pred, dim=-1) == y).to(float))
      tot += y.shape[0]
      inf_dict["train acc"] = (correct / tot).item()
      p_bar.set_postfix(inf_dict)

      #if batch_idx % 20 == 0:
   with torch.no_grad():
      model.eval()
      all_preds = []
      ys = []
      for x, y in cifar_dataloader_test:
         x, y = x.to(d), y.to(d)
         all_preds.append(model(x))
         ys.append(y)

      all_preds = torch.cat(all_preds)
      ys = torch.cat(ys)
      test_acc = torch.mean((torch.argmax(all_preds, dim=-1) == ys).to(float))
      p_bar = tqdm.tqdm(enumerate(cifar_dataloader_train), unit="batch", total=len(cifar_dataloader_train))
      inf_dict["test acc"] = test_acc.cpu().item()
      #p_bar.set_postfix(inf_dict)
      model.train()

   sched.step()
   print(f"Epoch {epoch_dix} learning rate: {sched.get_last_lr()}")


















