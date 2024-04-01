import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as CosSched
import time
def setup_optimizer(model, opt=AdamW, Sched=CosSched, lr=1e-3, lr_scale = 0.1, weight_decay=0.01, epochs=100):

   # All parameters in the model
   all_parameters = list(model.parameters())

   # General parameters don't contain the special _optim key
   params = [p for p in all_parameters if not hasattr(p, "_optim")]
   params_ABC = [p for p in all_parameters if hasattr(p, "_optim")]

   optimizer = opt([
      {"params": params},
      {"params": params_ABC, "lr": lr * lr_scale, "weight_decay": 0.0}],
      lr=lr, weight_decay=weight_decay)

   scheduler = Sched(optimizer, epochs)

   return optimizer, scheduler

#
# optimizer, scheduler = setup_optimizer(
#    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
# )

from s4_playground.rope_fork import RotaryEmbedding

class RotaryEmbeddingCustom(torch.nn.Module):
   def __init__(self, d_model , loc="all", BDL_shape=True, theta=10, seq_norm=None, learned_freq=False):
      super().__init__()

      self.pos_emb_layer = RotaryEmbedding(dim=d_model, theta=theta, seq_norm=seq_norm, learned_freq=learned_freq)
      assert loc in ["all", "first", "everyother"], "los is {} of type {}".format(loc, type(loc))
      self.loc = loc
      self.BDL_shape = BDL_shape

   def forward(self, x, layer_idx):
      if self.BDL_shape:
         if self.loc == "all":
            return self.pos_emb_layer.rotate_queries_or_keys(x.transpose(-1, -2)).transpose(-1, -2)

         if self.loc == "first" and layer_idx == 0:
            return self.pos_emb_layer.rotate_queries_or_keys(x.transpose(-1, -2)).transpose(-1, -2)

         if self.loc == "everyother" and ((layer_idx) % 2 == 0):
            return self.pos_emb_layer.rotate_queries_or_keys(x.transpose(-1, -2)).transpose(-1, -2)
         return x

      else:
         if self.loc == "all":
            return self.pos_emb_layer.rotate_queries_or_keys(x)

         if self.loc == "first" and layer_idx == 0:
            return self.pos_emb_layer.rotate_queries_or_keys(x)

         if self.loc == "everyother" and ((layer_idx) % 2 == 0):
            return self.pos_emb_layer.rotate_queries_or_keys(x)
         return x

def print_model_stats(model):
   name = model.__class__.__name__ + " " + model.s4
   n_layer, d_state, d_model = model.n_layer, model.d_state, model.d_model
   trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
   nontrainable_params = sum([param.numel() for param in model.parameters() if not param.requires_grad])
   print("####################################################################################")
   print(f"{name}: trainable {trainable_params/1e6:.3f}m, \n"
         f"n_layers: {n_layer}, d_model: {d_model}, d_state: {d_state}")
   if nontrainable_params != 0: print("Non-trainbale params:", nontrainable_params)
   #print("####################################################################################")
   return name, trainable_params

# assumes tokesn atm
def model_throughput(model, vocab_size, d_input, b=64, L=1000, reps=10):
   opt = AdamW(model.parameters(), lr=1e-9)
   if vocab_size is not None:
      batch = (torch.rand((b, L))*(vocab_size-1)).to(torch.long).abs()
   else:
      batch = torch.randn((b, L, d_input))

   batch = batch.to("cuda")
   #warm up
   model = model.eval()
   for _ in range(3):
      model(batch)
   torch.cuda.synchronize()

   # farward
   torch.cuda.reset_peak_memory_stats()
   t0 = time.perf_counter()
   for _ in range(reps):
      model(batch)
   torch.cuda.synchronize()
   t1 = (time.perf_counter() - t0) / reps
   mem1 = torch.cuda.max_memory_allocated()

   # backward
   model = model.train()
   torch.cuda.reset_peak_memory_stats()
   t0 = time.perf_counter()
   for _ in range(int(reps)):
      (model(batch)).sum().backward()
      opt.step()
      opt.zero_grad()
   torch.cuda.synchronize()
   t2 = (time.perf_counter() - t0) / reps
   mem2 = torch.cuda.max_memory_allocated()

   print(f"far/back mem GB: {mem1/1e9:.1f}, {mem2/1e9:.1f}")
   print(f"far/back speed b/s: {1/t1:.1f}, {1/t2:.1f}")
   model = model.to("cpu")

def data_throughput(data_loader, name, warmup=10, actualrun=50):
   for i, _ in enumerate(data_loader):
      if i == warmup:
         break

   t0 = time.perf_counter()
   for i, _ in enumerate(data_loader):
      if i == actualrun:
         break
   t1 = actualrun / (time.perf_counter() - t0)

   print(f"{name} batch per sec: {t1:.1f}" )



# ROOT_data = "../data/cifar10/"
# class Cifar10seq(Dataset):
#    def __init__(self, train=True, d="cpu"):
#       data = torchvision.datasets.CIFAR10(root=ROOT_data, train=train, download=False)
#       x = torch.from_numpy(data.data).to(torch.float)
#       x = rearrange(x, "a b c d -> a (b c) d")
#       x = x / x.max()
#       self.x = x.to(d)
#       self.y = torch.tensor(data.targets).to(d).to(torch.long)
#
#    def __len__(self):
#       return self.x.shape[0]
#
#    def __getitem__(self, idx):
#       return self.x[idx], self.y[idx]






