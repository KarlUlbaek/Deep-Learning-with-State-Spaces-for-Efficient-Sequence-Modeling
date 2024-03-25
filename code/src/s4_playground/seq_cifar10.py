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
cifar_dataloader_train = DataLoader(dataset=Cifar10seq(train=True),
                                    shuffle=True,
                                    batch_size=b)
cifar_dataloader_test = DataLoader(dataset=Cifar10seq(train=False),
                                    shuffle=False,
                                    batch_size=b*5)
lr = 1e-3
n_layers = 4
d_data = 3
d_model = 128
d_state = 64
L = next(iter(cifar_dataloader_train))[0].shape[1]
d = "cuda"
classes = 10
classification = True

from mamba_fork.mamba_ssm.models.mixer_seq_simple import MixerModel as MambaNN
s6NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, 
               d_out = classes, discrete=False, fused_add_norm=True, classification=classification).to(d)

s4NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, 
               d_out = classes, discrete=False, fused_add_norm=True,
               s4={"mode":"dplr", "hippo_init":"legs"},classification=classification).to(d)

s4dNN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, 
               d_out = classes, discrete=False, fused_add_norm=True,
               s4={"mode":"diag", "hippo_init":"legs"}, classification=classification).to(d)

model = s6NN
opt = AdamW(model.parameters(), lr=lr, foreach=True)
n_epochs = 100

p_bar = tqdm.tqdm(range(n_epochs))
print("trainable params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))
for epoch_dix in p_bar:
   for batch_idx, (x, y) in (enumerate(cifar_dataloader_train)):
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
         all_preds.append(model(x))
         ys.append(y)

      all_preds = torch.cat(all_preds)
      ys = torch.cat(ys)
      acc = torch.mean((torch.argmax(all_preds, dim=-1) == ys).to(float))
      p_bar.set_postfix({'test loss': loss.cpu().item(), "test acc":acc.cpu().item()})
      model.train()





































