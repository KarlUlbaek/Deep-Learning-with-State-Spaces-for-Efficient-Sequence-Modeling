import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.optim import AdamW
from einops import rearrange
from torch.nn import CrossEntropyLoss
import sys, os
sys.path.append(os.getcwd())

def trainer(model, train_loader, eval_loader, test_mode, criterion, optimizer, scheduler, n_epochs, wandb_object):
   info_dict = {}
   for epoch_idx in range(n_epochs):
      p_bar = tqdm.tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc="E: {}/{}".format(epoch_idx+1, n_epochs),
                        unit="b")
      p_bar.set_postfix(info_dict)
      correct, tot = 0, 0
      model.train()

      for batch_idx, xy_ in p_bar:
         x, y = xy_[0].to(d), xy_[1].to(d)
         pred = model(x)
         loss = criterion(pred, y)

         loss.backward()
         optimizer.step()
         optimizer.zero_grad()

         correct += torch.sum((torch.argmax(pred, dim=-1) == y).to(float))
         tot += y.shape[0]
         info_dict["train acc"] = (correct / tot).item()
         info_dict["loss"] = loss.item()
         p_bar.set_postfix(info_dict)

         if test_mode and batch_idx == 1: break

      info_dict = eval(model, eval_loader, info_dict, test_mode)
      info_dict["lr"] = scheduler.get_last_lr()[0]
      scheduler.step()

      if not test_mode and wandb_object is not None: wandb_object.log(info_dict)
      if test_mode and epoch_idx == 1: break
      #p_bar = tqdm.tqdm(enumerate(cifar_dataloader_train), unit="batch", total=len(cifar_dataloader_train))

      #if batch_idx % 20 == 0:
@torch.no_grad()
def eval(model, eval_loader, info_dict, test_mode):
   model.eval()
   all_preds = []
   ys = []
   for idx, xy_ in enumerate(eval_loader):
      x, y = xy_[0].to(d), xy_[1].to(d)
      pred = model(x)
      all_preds.append(pred)
      ys.append(y)

      if test_mode and idx == 1: break

   all_preds = torch.cat(all_preds)
   ys = torch.cat(ys)
   test_acc = torch.mean((torch.argmax(all_preds, dim=-1) == ys).to(float))
   info_dict["test acc"] = test_acc.cpu().item()
   #p_bar.set_postfix(inf_dict)
   return info_dict

#
# print(f"Epoch {epoch_dix} learning rate: {sched.get_last_lr()[0]}")



if __name__ == "__main__":
   import os
   import sys
   import wandb
   sys.path.append(os.getcwd())

   # todo
   # infer model specs
   # print model specs
   # make all training configs
   # make quick training run of all configs
   # wandb
   # clean up

   ROOT_data = "../data/cifar10/"
   class Cifar10seq(Dataset):
      def __init__(self, train=True, d="cpu"):
         data = torchvision.datasets.CIFAR10(root=ROOT_data, train=train, download=False)
         x = torch.from_numpy(data.data).to(torch.float)
         x = rearrange(x, "a b c d -> a (b c) d")
         x = x / x.max()
         self.x = x.to(d)
         self.y = torch.tensor(data.targets).to(d).to(torch.long)

      def __len__(self):
         return self.x.shape[0]

      def __getitem__(self, idx):
         return self.x[idx], self.y[idx]

   #d = Cifar10seq()


   if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
      fast = False
   else:
      fast = True

   from mamba_fork.mamba_ssm.models.mixer_seq_simple import MambaModel
   from s4_modules import S4ClassicModel, s4ClassicModule
   from functools import partial


   n_layer = 4
   d_model = 86
   d_state = 16
   dropout = 0.1
   s6Mamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast)

   s4Mamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast, s4={"mode": "dplr", "hippo_init": "legs"})

   s4dMamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                      fused_add_norm=fast, rms_norm=fast, s4={"mode": "diag", "hippo_init": "legs"})

   d_state = 64
   d_model = 128
   layernorm = True # = True means layernorm and not RMS
   prenorm = False # =
   s4Classic  = partial(S4ClassicModel, s4_or_s6=s4ClassicModule, n_layer=n_layer, d_model=d_model,
                        d_state=d_state, dropout=dropout, s4={"mode": "dplr", "hippo_init": "legs"},
                        layernorm=layernorm, prenorm=prenorm)
   s4dClassic = partial(S4ClassicModel, s4_or_s6=s4ClassicModule, n_layer=n_layer, d_model=d_model,
                        d_state=d_state, dropout=dropout, s4={"mode": "diag", "hippo_init": "legs"},
                        layernorm=layernorm, prenorm=prenorm)

   def print_model_stats(model):
      name = model.__class__.__name__ + " " + model.s4
      n_layer, d_state, d_model = model.n_layer, model.d_state, model.d_model
      trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
      nontrainable_params = sum([param.numel() for param in model.parameters() if not param.requires_grad])
      print("####################################################################################")
      print(f"{name}: trainable {trainable_params/1e6:.3f}m, \n"
            f"n_layers: {n_layer}, d_model: {d_model}, d_state: {d_state}")
      if nontrainable_params != 0: print("nontrainable params!!!!!!!!!1", nontrainable_params)
      #print("####################################################################################")
      return name, trainable_params

   # assumes tokesn atm
   def model_throughput(model, vocab_size, d_input, b=64, L=1000, reps=10):
      import time
      opt = AdamW(model.parameters(), lr=1e-9)
      if vocab_size is not None:
         batch = (torch.rand((b, L))*(vocab_size-1)).to(torch.long).abs()
      else:
         batch = torch.randn((b, L, d_input))

      batch = batch.to("cuda")
      #warm up
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




   Models = [s4Classic, s4dClassic, s6Mamba, s4Mamba, s4dMamba]


   from misc import setup_optimizer

   from lra_benchmarks_fork.lra_datasets import LRATensor
   #datasetnames = ["Cifar10DatasetToken", "ListOpsDataset", "ImdbDataset"]
   #datasetnames = ["Cifar10DatasetCont", "Cifar10DatasetToken", "ImdbDataset"]
   from lra import IMDB, PathFinder
   from s4_fork.src.dataloaders.basic import CIFAR10
   from copy import deepcopy

   cwd = os.getcwd()

   data = CIFAR10("cifar")
   data.setup("../data/cifar10")
   CIFAR10cont = deepcopy(data)
   data.tokenize = True
   data.grayscale = True
   data.setup("../data/cifar10")#split_train_val(val_split=0.1)
   CIFAR10token = deepcopy(data)

   data = IMDB("imdb")
   data.setup("../data")#split_train_val(val_split=0.1)
   IMDBtoken = deepcopy(data)

   data = PathFinder("pathfinder")
   data.setup("../data")#.split_train_val(val_split=0.1)
   Pathfindercont = deepcopy(data)
   data.tokenize = True
   data.setup("../data")#.split_train_val(val_split=0.1)
   Pathfindertoken = deepcopy(data)

   datasets = [IMDBtoken, CIFAR10token, Pathfindertoken, CIFAR10cont, Pathfindercont]
   #datasets = [IMDBtoken, Pathfindercont, Pathfindertoken]



   n_epochs = 30
   b = 64
   classification = True
   num_workers = 6
   d = "cuda"
   lr = 3e-3
   lr_scale = 0.1
   criterion = CrossEntropyLoss()
   test_throughput = True

   run_test_run = True
   #wandb.login(key="b797f78f8b0f6b430d646e95a505e747eef4315c")
   NOT_wandb_logging = True
   test_modes = [True, False] if run_test_run else [False]
   for test_mode in test_modes:
      for dataset in datasets:
         train_loader = dataset.train_dataloader(batch_size=b, num_workers=num_workers, shuffle=True)
         # toptier cursed. IMDB doesnt have a val loader and CIFAR and PAtherfinder returns a dictionary object with the key *None*
         try:
            eval_loader = dataset.val_dataloader(batch_size=b, num_workers=num_workers)[None]
         except TypeError as e:
            eval_loader = dataset.test_dataloader(batch_size=b, num_workers=num_workers)

         #eval_loader = DataLoader(dataset.dataset_test, batch_size=b, num_workers=0)
         if eval_loader is None:
            print("CAOS")

         xy_ = next(iter(train_loader))
         x, y = xy_[0], xy_[1]
         L = x.shape[1]
         if x.dtype in [torch.int64, torch.int32, torch.int16]:
            vocab_size = dataset.n_tokens
            d_input = 1
         else:
            vocab_size = None
            d_input = dataset.d_input

         d_output = dataset.d_output

         for Model in Models:
            d_name = dataset.__class__.__name__
            d_name = (d_name+"token") if vocab_size is not None else (d_name+"cons")
            print(f"\n Running on {d_name}")
            model = Model(d_input=d_input, d_output=d_output, vocab_size=vocab_size, classification=classification)
            m_name, n_params = print_model_stats(model)
            model = model.to(d)
            if test_throughput: model_throughput(model, model.vocab_size, d_input=d_input, b=b, L=L)
            optimizer, scheduler = setup_optimizer(model, lr=lr, epochs=n_epochs)

            if test_mode or NOT_wandb_logging:
               wandb_object = None
            else:
               wandb_object = wandb.init(project="LRA2", config={"model":m_name, "data":d_name, "lr":lr, "b": b,
                                                                 "n_layer":model.n_layer, "d_state":model.d_state,
                                                                 "d_model":model.d_model, "n_params": n_params})

            trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, test_mode=test_mode,
                    criterion=criterion, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs,
                    wandb_object=wandb_object)
            model = model.to("cpu")

















