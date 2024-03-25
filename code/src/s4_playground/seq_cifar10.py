import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from einops import rearrange

ROOT = "../data/cifar10/"

class Cifar10seq(Dataset):
   def __init__(self, train=True, d="cuda"):
      data = torchvision.datasets.CIFAR10(root=ROOT, train=train, download=True)
      x = torch.from_numpy(data.data).to(torch.float)
      # x = x.mean(dim=-1)
      # x = rearrange(x, "b c d -> b (c d) 1")
      x = rearrange(x, "a b c d -> a (b c) d")
      x = x / x.max()
      self.x = x.to(d)
      self.y = torch.tensor(data.targets).to(d).to(torch.long)


   def __len__(self):
      return self.x.shape[0]

   def __getitem__(self, idx):
      return self.x[idx], self.y[idx]



b = 64
num_workers = 6
cifar_dataloader_train = DataLoader(dataset=Cifar10seq(train=True, d="cpu"),
                                    shuffle=True,
                                    batch_size=b, num_workers=num_workers)
cifar_dataloader_test = DataLoader(dataset=Cifar10seq(train=False, d="cpu"),
                                    shuffle=False,
                                    batch_size=b*5,
                                    num_workers=num_workers)
lr = 1e-3
n_layers = 4
d_data = 3
d_model = 128
d_state = 64
dropout = None# 0.1
L = next(iter(cifar_dataloader_train))[0].shape[1]
d = "cuda"
classes = 10
classification = True

if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
   fast = False
else:
   fast = True

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
print(model)
opt = AdamW(model.parameters(), lr=lr, foreach=True)
n_epochs = 100

print("trainable params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))
p_bar = tqdm.tqdm(enumerate(cifar_dataloader_train), unit="batch", total=len(cifar_dataloader_train))
for epoch_dix in range(n_epochs):
   for batch_idx, (x, y) in p_bar:
      x, y = x.to(d), y.to(d)
      last_pred = model(x)
      #print(last_pred.argmax(dim=1))
      #last_pred = pred[:,-1000,:].mean(dim=1)
      #print(last_pred.shape)
      loss = torch.nn.functional.cross_entropy(last_pred, y)
      loss.backward()
      opt.step()
      opt.zero_grad()

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
      acc = torch.mean((torch.argmax(all_preds, dim=-1) == ys).to(float))
      p_bar = tqdm.tqdm(enumerate(cifar_dataloader_train), unit="batch", total=len(cifar_dataloader_train))
      p_bar.set_postfix({'test loss': loss.cpu().item(), "test acc":acc.cpu().item()})
      model.train()





































