import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.nn import CrossEntropyLoss
import sys, os
sys.path.append(os.getcwd())
from s4_playground.misc import setup_optimizer, print_model_stats, model_throughput, data_throughput



def trainer(model, train_loader, val_loader, test_loader, test_mode, criterion, optimizer, scheduler, n_epochs, wandb_run,
            improvement_demand=0.01, classification=True, bi=False):

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

      info_dict = eval(model, val_loader, info_dict, test_mode, criterion, classification, bi=bi)
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
   info_dict = eval(model, test_loader, info_dict, test_mode, criterion, classification, bi=bi, test_data=True)
   test_perf = {key: val for key, val in info_dict.items() if key.startswith("val")}
   wandb_run.log(test_perf)

   if wandb_run is not None: wandb_run.finish()
   succes = True
   return succes

      #if batch_idx % 20 == 0:
@torch.no_grad()
def eval(model, eval_loader, info_dict, test_mode, criterion, classification=True, bi=False, test_data=False):
   if test_data:
      split = "val"
   else:
      split= "test"

   model.eval()
   all_preds = []
   ys = []
   for idx, xy_ in enumerate(eval_loader):
      x = xy_[0].to(d)
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

#
# print(f"Epoch {epoch_dix} learning rate: {sched.get_last_lr()[0]}")


if __name__ == "__main__":
   from misc import get_model_name, get_data_name, get_data_dim
   import wandb
   from copy import deepcopy

   from mamba_fork.mamba_ssm.models.mixer_seq_simple import MambaModel
   from s4_modules import S4ClassicModel, s4ClassicModule
   from functools import partial

   # my own desktop is old and doesnt support some fused operations
   if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
      fast = False
   else:
      fast = True


   n_layer = 6
   d_model = 116
   d_state = 16
   dropout = 0.1
   s6Mamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast, bi_s6=False)

   s4dMamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                      fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs"})

   d_state = 64
   d_model = 170
   layernorm = True # = True means layernorm and not RMS
   prenorm = False # =
   s4dClassic = partial(S4ClassicModel, n_layer=n_layer, d_model=d_model,
                        d_state=d_state, dropout=dropout, s4_kwargs={"mode": "diag", "init": "legs"},
                        layernorm=layernorm, prenorm=prenorm)

   from lra import IMDB, PathFinder
   from s4_fork.src.dataloaders.basic import CIFAR10
   from s4_playground.misc import AAN_tensor_dataset

   AAN_dataset = AAN_tensor_dataset("../data")
   next(iter(AAN_dataset.train_dataloader()))

   data = CIFAR10("cifar")
   data.setup("../data/cifar10")
   CIFAR10cont = deepcopy(data)
   data.tokenize = True
   data.grayscale = True
   data.setup("../data/cifar10")
   CIFAR10token = deepcopy(data)

   data = IMDB("imdb")
   data.l_max = 1024
   data.setup("../data/imdb")
   IMDBtoken = deepcopy(data)
   #
   data = PathFinder("pathfinder")
   data.setup("../data")
   #data.setup("../data")
   Pathfindercont = deepcopy(data)
   data.tokenize = True
   data.setup("../data")
   #data.setup("../data")
   Pathfindertoken = deepcopy(data)


   d_model = 116
   d_state = 16
   #bi = ["paper_bi", "stacked_bi", "sequential_bi", "sequential_bi_tied", "half_dim_bi", "", "placebo"]
   #bi = ["sequential_bi", "half_dim_bi", ""]
   #bi_module = {"d_model_scale": 0.66, "d_state_scale":1.0, "placebo": False}
   # m1 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs"})
   m1 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                         fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs"})
                         #bi_module={"d_model_scale": 0.72, "d_state_scale":1.0, "placebo": False})
   # m2 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs", "bi":"sequential_bi"},
   #                       bi_module={"d_model_scale": 0.72, "d_state_scale":1.0, "placebo": False},
   #                       pos_emb = {"loc":"all"})
   # m3 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs", "bi":"sequential_bi"},
   #                       pos_emb = {"loc":"all"})
   #
   # m4 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast, bi_s6 = {"bi":True},
   #                       bi_module={"d_model_scale": 0.72, "d_state_scale":1.0, "placebo": False})
   #
   # m5 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast, bi_s6 = {"bi":True},
   #                       bi_module={"d_model_scale": 0.72, "d_state_scale":1.0, "placebo": False},
   #                       pos_emb = {"b_c_dt_x":"b_c_dt", "loc":"all"})
   #
   # m6 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast, bi_s6 = {"bi":True},
   #                       pos_emb = {"b_c_dt_x":"b_c_dt", "loc":"all"})
   # m4 =    partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
   #                       fused_add_norm=fast, rms_norm=fast,
   #                       bi_module={"d_model_scale": 0.72, "d_state_scale":1.0, "placebo": True})

   d_state = 64
   d_model = 170
   m2 =    partial(S4ClassicModel, n_layer=n_layer, d_model=d_model,
                        d_state=d_state, dropout=dropout, s4_kwargs={"mode": "diag", "init": "legs"},
                        layernorm=layernorm, prenorm=prenorm)
   #s4dMamba, s4dClassic]#, s4dMamba, s4Classic, s4dClassic, s6Mamba]
   #datasets = [IMDBtoken, CIFAR10token, CIFAR10cont, Pathfindertoken, Pathfindercont]

   models = [m2]#, m2, m3, m4, m5, m6]

   datasets = [CIFAR10cont, Pathfindercont, Pathfindertoken, IMDBtoken, CIFAR10token]#, IMDBtoken]#, CIFAR10cont] AAN_dataset
   #datasets = [Pathfindercont]

   n_epochs = 2
   b = 64
   num_workers = 0
   d = "cuda"
   lr = 3e-3
   lr_scale = 0.1 # 0.1
   weight_decay = 0.01 # 0.01
   #pos_emb = {"loc": "all", "theta": 10_000, "seq_norm": 1024, "learned_freq": False, "b_c_dt_x": "b_c_dt"}
   #loc must be ["all", "first", "everyother"]
   criterion = CrossEntropyLoss()

   test_throughput = True
   run_test_run = True
   wandb_logging = True
   wandb_name = "testval" #""
   data_name_add = ""
   model_name_add = ""

   test_modes = [True, False] if run_test_run else [False]
   print("datasets:", [dataset.__class__.__name__ for dataset in datasets])
   print("models:", [model.func.__name__ for model in models])
   for test_mode in test_modes:
      for dataset in datasets:
         # toptier cursed. IMDB doesnt have a val loader and CIFAR and PAtherfinder returns a dictionary object with the key *None*
         train_loader = dataset.train_dataloader(batch_size=b, num_workers=num_workers, shuffle=True)
         val_loader = dataset.val_dataloader(batch_size=b, num_workers=num_workers)
         test_loader = dataset.test_dataloader(batch_size=b, num_workers=num_workers)

         for model in models:

            d_input, d_output, vocab_size, L = get_data_dim(train_loader, dataset)
            model = model(d_input=d_input, d_output=d_output, vocab_size=vocab_size, classification=True).to(d)
            m_name = get_model_name(model, model_name_add)
            d_name = get_data_name(dataset, data_name_add)

            lr_L = lr * (2048 / L)
            optimizer, scheduler = setup_optimizer(model, lr=lr_L, epochs=n_epochs, weight_decay=weight_decay)

            print("####################################################################################")
            print("MODEL:", m_name)
            n_params = print_model_stats(model)
            if test_throughput: model_throughput(deepcopy(model), model.vocab_size, d_input=d_input, e=n_epochs,
                                                 len_data_loader=len(train_loader), b=b, L=L)
            print("DATA:", d_name)
            if test_throughput: data_throughput(train_loader)
            print(
               f"hparams: e:{n_epochs}, b:{b}, lr:{lr_L}, w_d:{weight_decay}, L:{L}, drop:{model.dropout}")

            if test_mode or not wandb_logging:
               wandb_run = None
            else:
               print("Logging with wandb! Happens after 2. epoch!")
               wandb_run = partial(wandb.init, project=d_name + wandb_name, name=m_name,
                                   config={"model": m_name, "data": d_name, "lr": lr_L, "b": b,
                                           "weight_decay": weight_decay,
                                           "n_layer": model.n_layer, "d_state": model.d_state, "dropout": model.dropout,
                                           "d_model": model.d_model, "n_params": n_params})

            _ = trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, test_mode=test_mode,
                        criterion=criterion, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs,
                        wandb_run=wandb_run, classification=dataset.classification, bi=bool(model.bi_module))
            model = model.to("cpu")













