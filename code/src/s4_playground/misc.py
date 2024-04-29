import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as CosSched
import time
import tqdm
from copy import deepcopy

def trainer(model, train_loader, val_loader, test_loader, test_mode, criterion, optimizer, scheduler, n_epochs, wandb_run,
            improvement_demand=0.01, classification=True, bi=False, d="cuda"):

   info_dict, train_acc = {}, 0
   for epoch_idx in range(n_epochs):
      info_dict_last_iter = deepcopy(info_dict) # make copy, it is only usued for logging the first iter
      p_bar = tqdm.tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc="E: {}/{}".format(epoch_idx+1, n_epochs),
                        unit="b")
      p_bar.set_postfix(info_dict)
      correct, tot, tot_loss = .0, .0, .0
      model.train()
      random_guess = 1 / model.d_output

      for batch_idx, xy_ in p_bar:
         if classification or bi:
            x, y = xy_[0].to(d), xy_[1].to(d)
            pred = model(x)

            if not classification and bi: # masked language modeling
               loss = criterion(pred.transpose(-1,-2), y)
            else: # regular classification
               loss = criterion(pred, y)

         else: # causual/next word prediction
            x = xy_[0].to(d)
            if model.reversed_pre:
               assert not bool(model.bi_s6) and not bool(model.bi_module)
               x = x.flip(1)
            pred = model(x)
            loss = criterion(pred[:,:-1,:].transpose(-1,-2), x[:,1:])

         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

         tot_loss += loss.item()
         info_dict["loss"] = tot_loss / (batch_idx + 1)

         if classification:
            correct += torch.sum((torch.argmax(pred, dim=-1) == y).to(float))
            tot += y.shape[0]
            train_acc = (correct / tot).item()
            info_dict["train acc"] = train_acc
         else:
            info_dict["train per"] = torch.exp(torch.tensor(info_dict["loss"])).item()



         p_bar.set_postfix(info_dict)

         # break early due to test mode
         if test_mode and batch_idx == 1:
            #Test that the testset can run also
            info_dict = eval(model, test_loader, info_dict, test_mode, criterion, classification, bi=bi, test_data=True)
            break

      info_dict = eval(model, val_loader, info_dict, test_mode, criterion, classification, bi=bi,d=d)
      info_dict["lr"] = scheduler.get_last_lr()[0]
      scheduler.step()

      # break early due to poor initializing. i.e. if we havent leared anything for the first epoch
      # we demand to be atleast 3% points better than random else we reinitialize
      if epoch_idx==0 and (train_acc - random_guess) < improvement_demand and not test_mode and classification:
         print(f"model failed since train acc is {train_acc:.3f} and random is {random_guess:.3f} "
               f"and tol is {improvement_demand:.2f}")
         print("Reinitializing and rerunning!!!")

         succes = False
         return succes

      if epoch_idx == 1 and test_mode:
         succes = True
         return succes

      # init the logging and log 0epoch and 1st epoch
      if epoch_idx == 1 and wandb_run is not None:
         wandb_run = wandb_run()
         wandb_run.log(info_dict_last_iter)
         wandb_run.log(info_dict)

      # log as usual for the rest of the epochs
      if epoch_idx > 1 and wandb_run is not None:
         wandb_run.log(info_dict)

   #get test-test performance
   info_dict = eval(model, test_loader, info_dict, test_mode, criterion, classification, bi=bi, test_data=True,d=d)
   test_perf = {key: val for key, val in info_dict.items() if key.startswith("val")}
   wandb_run.log(test_perf)

   if wandb_run is not None: wandb_run.finish()
   succes = True
   return succes

      #if batch_idx % 20 == 0:
@torch.no_grad()
def eval(model, eval_loader, info_dict, test_mode, criterion, classification=True, bi=False, test_data=False, d="cuda"):
   if test_data:
      split = "val"
   else:
      split= "test"

   model.eval()
   all_preds = []
   ys = []
   for idx, xy_ in enumerate(eval_loader):
      x = xy_[0].to(d)

      # reversed causal pretraining
      if model.reversed_pre:
         assert not bool(model.bi_s6) and not bool(model.bi_module)
         x = x.flip(1)

      y = xy_[1].to(d) if classification or bi else x
      pred = model(x)
      all_preds.append(pred)
      ys.append(y)

      if test_mode and idx == 1: break

   all_preds = torch.cat(all_preds)
   ys = torch.cat(ys)
   if classification:
      test_acc = (torch.argmax(all_preds, dim=-1) == ys).float().mean()
      info_dict[f"{split} acc"] = test_acc.cpu().item()

   elif bi: #bi pretraining
      per = torch.exp(criterion(all_preds.transpose(-1,-2), ys))
      info_dict[f"{split} per"] = per.cpu().item()

   else: # causal pretraining
      per = torch.exp(criterion(all_preds[:,:-1,:].transpose(-1,-2), ys[:,1:]))
      info_dict[f"{split} per"] = per.cpu().item()

   #p_bar.set_postfix(inf_dict)
   return info_dict


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

   scheduler = Sched(optimizer, epochs, eta_min=lr*0.1)

   return optimizer, scheduler

#
# optimizer, scheduler = setup_optimizer(
#    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
# )

from s4_playground.rope_fork import RotaryEmbedding

class RotaryEmbeddingCustom(torch.nn.Module):
   def __init__(self, d_model , loc="all", BDL_shape=True, theta=10_100, seq_norm=None, learned_freq=False, b_c_dt_x=None):
      super().__init__()
      # b_c_dt_x is only used by s6mamba and actually not even in here

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
   n_layer, d_state, d_model = model.n_layer, model.d_state, model.d_model
   trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
   nontrainable_params = sum([param.numel() for param in model.parameters() if not param.requires_grad])
   print(f"trainable: {trainable_params/1e6:.3f}m, \n"
         f"n_layers: {n_layer}, d_model: {d_model}, d_state: {d_state}")
   if nontrainable_params != 0: print("Non-trainbale params:", nontrainable_params)
   #print("####################################################################################")
   return trainable_params


def set_model_dropout(model, new_dropout):
   for param in model.parameters():
      if isinstance(param, torch.nn.Dropout1d):
         param.p = new_dropout

   model.dropout = new_dropout
   return model
def get_model_name(model, model_name_add, dataset=None):
   m_name = model.__class__.__name__ + "_" + model.s4
   if dataset is not None:
      m_name += "_" + str(dataset.max_length)

   if bool(model_name_add):
      m_name += "_" + model_name_add

   if bool(model.pos_emb):
      m_name += "_" + str(list(model.pos_emb.values()))

   m_name += model.s4_kwargs.get("bi", "")

   if hasattr(model, "bi_s6"):
      if model.bi_s6.get("bi", 0):
         m_name += "_bi"

   if hasattr(model, "bi_module"):
      if model.bi_module:
         m_name += "_BIMODULE"

      if model.bi_module.get("placebo", 0):
         m_name += "_placebo"

   return m_name


def get_data_dim(train_loader, dataset):
   xy_ = next(iter(train_loader))
   x = xy_[0]
   L = x.shape[1]
   if x.dtype in [torch.int64, torch.int32, torch.int16]:
      vocab_size = dataset.n_tokens
      d_input = 1
   else:
      vocab_size = None
      d_input = dataset.d_input

   d_output = dataset.d_output
   return d_input, d_output, vocab_size, L

def get_data_name(dataset, data_name_add, cons_or_token="NaN"):
   d_name = dataset.__class__.__name__

   if cons_or_token != "NaN":
      if cons_or_token is None:
         d_name += "cons" # short for continuous but misspelled woops. cant change because of wandb project names
      else:
         d_name += "token"

   d_name = d_name + data_name_add
   if not dataset.classification:
      d_name += "_pretrain"
   return d_name

# assumes tokesn atm
def model_throughput(model, vocab_size, d_input,
                     len_data_loader, e, b, L, reps=10):
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
   with torch.no_grad():
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
   est = int(((t2)*len_data_loader*e)/(60))
   print(f"estimated training time: {est}m")
   model = model.to("cpu")

def data_throughput(data_loader, warmup=5, actualrun=10):
   for i, xyz in enumerate(data_loader):
      if i == warmup:
         break

   t0 = time.perf_counter()
   for i, _ in enumerate(data_loader):
      if i == actualrun:
         break
   t1 = actualrun / (time.perf_counter() - t0)

   print(f"loader speed (size={xyz[0].shape[0]}): b/s: {t1:.1f}" )


from pathlib import Path
class AAN_tensor_dataset(torch.utils.data.Dataset):
   def __init__(self, path, split=None):
      super().__init__()
      self.n_tokens = 99
      self.classification = False
      self.dummy = torch.empty(1)
      self.d_output = self.n_tokens

      self.path = path
      if split is not None:
         self.d = torch.load(Path(path) / "aan/aan_tensor/{}.pt".format(split))

   def __setup__(self):
      pass

   def __len__(self):
      return self.d.shape[0]

   def __getitem__(self, idx):
      return self.d[idx].to(torch.int64), self.dummy

   def train_dataloader(self, batch_size=64, num_workers=0, shuffle=True):
      return torch.utils.data.DataLoader(AAN_tensor_dataset(self.path, split="train"),
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=shuffle)

   def test_dataloader(self, batch_size=64, num_workers=0):
      return torch.utils.data.DataLoader(AAN_tensor_dataset(self.path, split="test"),
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=True)

   def val_dataloader(self, batch_size=64, num_workers=0):
      return torch.utils.data.DataLoader(AAN_tensor_dataset(self.path, split="val"),
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=False)


if __name__ == "__main__":
   import os
   import tqdm

   print(os.getcwd())
   a = AAN_tensor_dataset("../../data")
   for val in tqdm.tqdm(a.train_dataloader()):
      pass

   for val in tqdm.tqdm(a.val_dataloader()):
      pass
   for val in tqdm.tqdm(a.test_dataloader()):
      pass

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






