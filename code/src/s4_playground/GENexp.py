import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.nn import CrossEntropyLoss
import sys, os
sys.path.append(os.getcwd())
from s4_playground.misc import setup_optimizer, print_model_stats, model_throughput, data_throughput



def trainer(model, train_loader, eval_loader, test_mode, criterion, optimizer, scheduler, n_epochs, wandb_run,
            improvement_demand=0.01, classification=True):

   info_dict, train_acc = {}, 0
   for epoch_idx in range(n_epochs):
      info_dict_last_iter = deepcopy(info_dict) # make copy, it is only usued for logging the first iter
      p_bar = tqdm.tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc="E: {}/{}".format(epoch_idx+1, n_epochs),
                        unit="b")
      p_bar.set_postfix(info_dict)
      correct, tot, tot_loss = 0, 0, 0
      model.train()
      random_guess = 1 / model.d_output

      for batch_idx, xy_ in p_bar:
         if classification:
            x, y = xy_[0].to(d), xy_[1].to(d)
            pred = model(x)
            loss = criterion(pred, y)
         else:
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
         if test_mode and batch_idx == 1: break

      info_dict = eval(model, eval_loader, info_dict, test_mode, criterion, classification)
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

   if wandb_run is not None: wandb_run.finish()
   succes = True
   return succes

      #if batch_idx % 20 == 0:
@torch.no_grad()
def eval(model, eval_loader, info_dict, test_mode, criterion, classification=True):
   model.eval()
   all_preds = []
   ys = []
   for idx, xy_ in enumerate(eval_loader):
      x = xy_[0].to(d)
      y = xy_[1].to(d) if classification else x
      pred = model(x)
      all_preds.append(pred)
      ys.append(y)

      if test_mode and idx == 1: break

   all_preds = torch.cat(all_preds)
   ys = torch.cat(ys)
   if classification:
      test_acc = torch.mean((torch.argmax(all_preds, dim=-1) == ys).to(float))
      info_dict["test acc"] = test_acc.cpu().item()
   else:
      per = torch.exp(criterion(all_preds[:,:-1,:].transpose(-1,-2), ys[:,1:]))
      info_dict["test per"] = per.cpu().item()

   #p_bar.set_postfix(inf_dict)
   return info_dict

#
# print(f"Epoch {epoch_dix} learning rate: {sched.get_last_lr()[0]}")


if __name__ == "__main__":
   import wandb
   from copy import deepcopy

   from mamba_fork.mamba_ssm.models.mixer_seq_simple import MambaModel
   from s4_modules import S4ClassicModel, s4ClassicModule
   from functools import partial

   if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
      fast = False
   else:
      fast = True


   n_layer = 6
   d_model = 116
   d_state = 16
   dropout = 0.0
   s6Mamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast, bi_s6={})

   s4dMamba = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                      fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs"})





   Models = [s6Mamba, s4dMamba]

   from genomics import Species




   n_epochs = 10
   sched_epochs = int(n_epochs * 1.5)
   b = 64
   num_workers = 2
   d = "cuda"
   lr = 1e-3
   lr_scale = 0.1 # 0.1
   weight_decay = 0.0 # 0.01
   criterion = CrossEntropyLoss()

   gen_clas = Species(["hippo", "human", "pig", "sheep", "lemur"], "../data/species", max_length=1024,
                tokenizer_name="char", total_size=50000, batch_size=b, classification=True, num_workers=num_workers)
   gen_clas.setup()

   datasets = [gen_clas]


   test_throughput = True
   run_test_run = True
   wandb_logging = False
   wandb_name = "_bi_test_v2" #""
   date_name_add = ""
   model_name_add = ""

   test_modes = [True, False] if run_test_run else [False]
   print("datasets:", [dataset.__class__.__name__ for dataset in datasets])
   print("models:", [model.func.__name__ for model in Models])
   for test_mode in test_modes:
      for dataset in datasets:
         train_loader = dataset.train_dataloader(batch_size=b, num_workers=num_workers, shuffle=True)
         eval_loader = dataset.val_dataloader(batch_size=b, num_workers=num_workers)
         assert eval_loader is not None, "EVAL LOADER NONE. CAOS"
         assert train_loader is not None, "TRAIN LOADER NONE. CAOS"

         xy_ = next(iter(train_loader))
         x = xy_[0]
         print(x.shape[0])
         L = x.shape[1]
         if x.dtype in [torch.int64, torch.int32, torch.int16]:
            vocab_size = dataset.n_tokens
            d_input = 1
         else:
            vocab_size = None
            d_input = dataset.d_input

         d_output = dataset.d_output

         for Model in Models:
            succes = False # we rerun the model till it actually learns
            while not succes:
               d_name = dataset.__class__.__name__
               d_name = (d_name+date_name_add+"token") if vocab_size is not None else (d_name+date_name_add+"cons")
               print(f"\n Running on {d_name}")
               model = Model(d_input=d_input, d_output=d_output, vocab_size=vocab_size,
                             classification=dataset.classification)

               m_name, n_params = print_model_stats(model)
               m_name += model_name_add + str(list(model.pos_emb.values())) + model.s4_kwargs.get("bi", "")
               if hasattr(model, "bi_s6"): m_name = m_name+"_bi" if model.bi_s6.get("bi", 0) else m_name
               if hasattr(model, "bi_module"):
                  m_name = (m_name + "_BIMODULE") if model.bi_module else m_name
                  m_name = m_name + "_placebo" if model.bi_module.get("placebo", 0) else m_name

               print(m_name)
               model = model.to(d)
               #print(model)
               optimizer, scheduler = setup_optimizer(model, lr=lr, epochs=sched_epochs, weight_decay=weight_decay)
               #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)

               if test_throughput:
                  data_throughput(train_loader, d_name)
                  model_throughput(deepcopy(model), model.vocab_size, d_input=d_input, b=b, L=L)

               if test_mode or not wandb_logging:
                  wandb_run = None
               else:
                  print("Logging with wandb! Happens after 2. epoch!")
                  wandb_run = partial(wandb.init, project=d_name+wandb_name, name=m_name,
                                      config={"model":m_name, "data":d_name, "lr":lr, "b": b, "weight_decay":weight_decay,
                                              "n_layer":model.n_layer, "d_state":model.d_state, "dropout": model.dropout,
                                              "d_model":model.d_model, "n_params": n_params})

               succes = trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, test_mode=test_mode,
                                criterion=criterion, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs,
                                wandb_run=wandb_run, classification=dataset.classification)

               model = model.to("cpu")

















