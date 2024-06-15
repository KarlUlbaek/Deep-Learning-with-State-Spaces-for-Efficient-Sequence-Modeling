import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
from torch.nn import CrossEntropyLoss
import sys, os
sys.path.append(os.getcwd())
from s4_playground.misc import setup_optimizer, print_model_stats, model_throughput, data_throughput
from s4_playground.misc import trainer


# def trainer(model, train_loader, eval_loader, test_mode, criterion, optimizer, scheduler, n_epochs, wandb_run,
#             improvement_demand=0.01, classification=True, bi=False):
#
#    info_dict, train_acc = {}, 0
#    for epoch_idx in range(n_epochs):
#       info_dict_last_iter = deepcopy(info_dict) # make copy, it is only usued for logging the first iter
#       p_bar = tqdm.tqdm(enumerate(train_loader),
#                         total=len(train_loader),
#                         desc="E: {}/{}".format(epoch_idx+1, n_epochs),
#                         unit="b")
#       p_bar.set_postfix(info_dict)
#       correct, tot, tot_loss = .0, .0, .0
#       model.train()
#       random_guess = 1 / model.d_output
#
#       for batch_idx, xy_ in p_bar:
#          if classification or bi:
#             x, y = xy_[0].to(d), xy_[1].to(d)
#             pred = model(x)
#
#             if not classification and bi: # masked language modeling
#                loss = criterion(pred.transpose(-1,-2), y)
#             else: # regular classification
#                loss = criterion(pred, y)
#
#          else: # causual/next word prediction
#             x = xy_[0].to(d)
#             pred = model(x)
#             loss = criterion(pred[:,:-1,:].transpose(-1,-2), x[:,1:])
#
#          loss.backward()
#          optimizer.step()
#          optimizer.zero_grad()
#
#          tot_loss += loss.item()
#          info_dict["loss"] = tot_loss / (batch_idx + 1)
#
#          if classification:
#             correct += torch.sum((torch.argmax(pred, dim=-1) == y).to(float))
#             tot += y.shape[0]
#             train_acc = (correct / tot).item()
#             info_dict["train acc"] = train_acc
#          else:
#             info_dict["train per"] = torch.exp(torch.tensor(info_dict["loss"])).item()
#
#
#
#          p_bar.set_postfix(info_dict)
#
#          # break early due to test mode
#          if test_mode and batch_idx == 1: break
#
#       info_dict = eval(model, eval_loader, info_dict, test_mode, criterion, classification, bi=bi)
#       info_dict["lr"] = scheduler.get_last_lr()[0]
#       scheduler.step()
#
#       # break early due to poor initializing. i.e. if we havent leared anything for the first epoch
#       # we demand to be atleast 3% points better than random else we reinitialize
#       if epoch_idx==0 and (train_acc - random_guess) < improvement_demand and not test_mode and classification:
#          print(f"model failed since train acc is {train_acc:.3f} and random is {random_guess:.3f} "
#                f"and tol is {improvement_demand:.2f}")
#          print("Reinitializing and rerunning!!!")
#
#          succes = False
#          return succes
#
#       if epoch_idx == 1 and test_mode:
#          succes = True
#          return succes
#
#       # init the logging and log 0epoch and 1st epoch
#       if epoch_idx == 1 and wandb_run is not None:
#          wandb_run = wandb_run()
#          wandb_run.log(info_dict_last_iter)
#          wandb_run.log(info_dict)
#
#       # log as usual for the rest of the epochs
#       if epoch_idx > 1 and wandb_run is not None:
#          wandb_run.log(info_dict)
#
#    if wandb_run is not None: wandb_run.finish()
#    succes = True
#    return succes
#
#       #if batch_idx % 20 == 0:
# @torch.no_grad()
# def eval(model, eval_loader, info_dict, test_mode, criterion, classification=True, bi=False):
#    model.eval()
#    all_preds = []
#    ys = []
#    for idx, xy_ in enumerate(eval_loader):
#       x = xy_[0].to(d)
#       y = xy_[1].to(d) if classification or bi else x
#       pred = model(x)
#       all_preds.append(pred)
#       ys.append(y)
#
#       if test_mode and idx == 1: break
#
#    all_preds = torch.cat(all_preds)
#    ys = torch.cat(ys)
#    if classification:
#       test_acc = (torch.argmax(all_preds, dim=-1) == ys).float().mean()
#       info_dict["test acc"] = test_acc.cpu().item()
#
#    elif bi: #bi pretraining
#       per = torch.exp(criterion(all_preds.transpose(-1,-2), ys))
#       info_dict["test per"] = per.cpu().item()
#
#    else: # causal pretraining
#       per = torch.exp(criterion(all_preds[:,:-1,:].transpose(-1,-2), ys[:,1:]))
#       info_dict["test per"] = per.cpu().item()
#
#    #p_bar.set_postfix(inf_dict)
#    return info_dict

#
# print(f"Epoch {epoch_dix} learning rate: {sched.get_last_lr()[0]}")
def set_model_dropout(model, new_dropout):
   for param in model.parameters():
      if isinstance(param, torch.nn.Dropout1d):
         param.p = new_dropout

   model.dropout = new_dropout
   return model


def get_model_name(model, model_name_add):
   m_name = model.__class__.__name__ + "_" + model.s4
   m_name += "_" + str(dataset.max_length)
   m_name += "_" + model_name_add + str(list(model.pos_emb.values())) + model.s4_kwargs.get("bi", "")
   if hasattr(model, "bi_s6"):
      if model.bi_s6.get("bi", 0):
         m_name += "_bi"

   if hasattr(model, "bi_module"):
      if model.bi_module:
         m_name += "_BIMODULE"

   if model.bi_module.get("placebo", 0):
      m_name += "_placebo"

   return m_name


def get_data_name(dataset, data_name_add):
   d_name = dataset.__class__.__name__
   d_name = d_name + data_name_add
   if not dataset.classification:
      d_name += "_pretrain"
   return d_name
from mamba_ssm.modules.mamba_simple import S6MambaModule
from mamba_ssm.ops.triton.layernorm import RMSNorm
def fuse_model(m1, m2, tmp):
   #m1_mamba = [mod for mod in m1.layers.modules() if isinstance(mod, S6MambaModule)]
   #m2_mamba = [mod for mod in m2.layers.modules() if isinstance(mod, S6MambaModule)]
   #tmp_ = [mod for mod in tmp.layers.modules() if isinstance(mod, S6MambaModule)]
   tmp_clone = deepcopy(tmp)
   for i in range(6):
      #swap in mamba modules from the 2 different models to be the new farward and backjward model
      tmp.layers[i].mixer.forward_model = m1.layers[i].mixer
      tmp.layers[i].mixer.backward_model = m2.layers[i].mixer

      #swap in layernorm from first model
      tmp.layers[i].norm = m1.layers[i].norm

      # tie weights
      tmp.layers[i].mixer.forward_model.in_proj.weight = tmp.layers[i].mixer.backward_model.in_proj.weight
      tmp.layers[i].mixer.forward_model.out_proj.weight = tmp.layers[i].mixer.backward_model.out_proj.weight
      # tmp_[i*2] = m1_mamba[i]
      # tmp_[i*2+1] = m2_mamba[i]

   #take encoder from first model
   tmp.encoder = m1.encoder

   # tmp = tmp.cuda()
   # tmp_clone = tmp_clone.cuda()
   # print(sum([param.numel() for param in tmp.parameters() if param.requires_grad]))
   # print(sum([param.sum() for param in tmp.parameters() if param.requires_grad]))
   # print(sum([param.sum() for param in tmp_clone.parameters() if param.requires_grad]))
   # b = torch.randn(2,16).int().abs().cuda()
   # print(tmp(b)[0,0,0])
   # print(tmp_clone(b)[0,0,0])
   #
   # print("")

   return tmp



if __name__ == "__main__":
   import wandb
   from copy import deepcopy

   from mamba_fork.mamba_ssm.models.mixer_seq_simple import MambaModel
   from DNA_dataloader import Species
   from functools import partial
   #
   if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
      fast = False
   else:
      fast = True
   # fast=False

   n_layer = 6
   d_model = 116
   d_state = 16
   dropout = 0.15
   s6Mamba1 = partial(MambaModel, n_layer=n_layer, d_model=int(d_model*0.9), d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast, bi_s6={})
   s6Mamba2 = partial(MambaModel, n_layer=n_layer, d_model=int(d_model*0.9), d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast, bi_s6={})

   s6Mamba_bi = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                     fused_add_norm=fast, rms_norm=fast,
                     bi_module={"d_model_scale":0.9, "d_state_scale":1.0, "tie_linear_proj":True})
   #, s4dMamba]

   s6Mamba_reg = partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                           fused_add_norm=fast, rms_norm=fast)
   #max_length = 1024*2
   n_data_points = 50_000
   n_epochs = 5
   b = 32

   lr_base = 1e-3
   num_workers = 4
   d = "cuda"
   lr_scale = 0.1 # 0.1
   weight_decay = 0.03 # 0.01
   criterion = CrossEntropyLoss()
   # default params
   df = {"lr_base": lr_base, "weight_decay": weight_decay, "b":b, "n_epochs": n_epochs, "dropout":dropout}
   #pretraing params
   pt = {"lr_base": lr_base*10*4, "weight_decay": 0.0, "b": b, "n_epochs": n_epochs*2,
         "dropout":0.0, "max_length_mult":1}

   # "both, finetune, pretrain"
   Models = [s6Mamba_reg]#[s6Mamba1, s6Mamba_bi]
   sizes = [1024]#[1024 * 4]*2
   training_plans = ["finetune"]#["pretrain", "finetune"]

   pretrain_name = "pretrain_big" #"pretrain_big"
   finetune_name = "finetune"#"finetune_small" #"finetune_small"
   finetune_len = -1#1_000_000 #-1 disables i.e. takes all values

   species = ["hippo", "human", "pig", "sheep", "lemur"]
   species_dir = "../data/species"
   datas = [Species(species=species, species_dir=species_dir, max_length=max_length,
                      total_size=n_data_points, batch_size=b, classification=False,
                      num_workers=num_workers
                      ) for max_length in sizes]
   datasets = list(zip(training_plans, datas))

   d_output = 5
   vocab_size = 12
   d_input = 1

   test_throughput = True
   run_test_run = True
   wandb_logging = True
   wandb_name = "" #""
   data_name_add = "_v1"
   model_name_add = ""

   test_modes = [True, False] if run_test_run else [False]
   print("datasets:", [dataset[1].__class__.__name__+"_"+str(dataset[1].max_length)+"_"+dataset[0] for dataset in datasets])
   print("models:", [model.func.__name__ for model in Models])
   print("####################################################################################")
   for test_mode in test_modes:
      stored_models = []
      #for Model in Models:
      for idx, (Model, training_plan, dataset) in enumerate(zip(Models, training_plans, datas)):
         dataset = dataset.setup(pretrain_name) #always init to pretraining
         max_length_default = dataset.max_length # store default for later
         assert training_plan in ["pretrain", "finetune", "both"]#, "train"]

         #init model and give it a proper name
         model = Model(d_input=d_input, d_output=vocab_size, vocab_size=vocab_size, classification=False)
         stored_models.append(model)
         m_name = get_model_name(model, model_name_add)

         if len(training_plans) == 2 and idx == 1:
            assert training_plans[-2]=="pretrain" and training_plans[-1]=="finetune"
            assert bool(model.bi_module)
            model = fuse_model(stored_models[0], deepcopy(stored_models[0]), model)
            m_name += "F1"

         if len(training_plans) == 3 and idx == 2:
            assert training_plans[-3] == "pretrain", training_plans[-2] == "pretrain" and training_plans[-1] == "finetune"
            assert bool(model.bi_module)
            model = fuse_model(stored_models[0], stored_models[1], model)
            m_name += "F2"

         run = 0
         while run < 2: # run a maximum of 2 runs. pretraining and finintuning or either or depeding on "training_plan"
            if training_plan == "finetune" or run == 1: #either we are not pretraining or we have already pretrained
               print("\nfinetuning!")
               run+=1 #increment so we finish after this run either way
               if run == 2 and training_plan == "both": m_name += "_pretrained" # we have pretrained

               # set default params
               lr_base_, weight_decay_, b_, n_epochs_ = df["lr_base"], df["weight_decay"], df["b"], df["n_epochs"]
               dataset.max_length = max_length_default  # set default in case it was changed for pretraining
               model = set_model_dropout(model, df["dropout"])

               model.classification = True
               model.d_output = d_output
               model.decoder = torch.nn.Linear(model.d_model, d_output) # change head of model to output one of the 5 classes

               dataset.setup(finetune_name, classification=True, finetune_len=finetune_len)

            elif training_plan == "pretrain": #only pretraining!
               print("\npretraining only!")
               run+=1 #increment so finish after this run

               lr_base_, weight_decay_, b_, n_epochs_ = pt["lr_base"], pt["weight_decay"], pt["b"], pt["n_epochs"]
               model = set_model_dropout(model, pt["dropout"])

               dataset.max_length = int(max_length_default * (pt["max_length_mult"]))
               dataset.setup(pretrain_name, classification=False, bi=bool(model.bi_module))


            else: # will do both pretraining and finetuning
               print("\npretraining now and finetuning after!")
               lr_base_, weight_decay_, b_, n_epochs_ = pt["lr_base"], pt["weight_decay"], pt["b"], pt["n_epochs"]
               model = set_model_dropout(model, pt["dropout"])

               dataset.max_length = int(max_length_default * (pt["max_length_mult"]))
               dataset.setup(pretrain_name, classification=False, bi=bool(model.bi_module))

            model = model.to(d)
            train_loader = dataset.train_dataloader(batch_size=b_, num_workers=num_workers, shuffle=True)
            val_loader = dataset.val_dataloader(batch_size=b_, num_workers=num_workers)
            test_loader = dataset.test_dataloader(batch_size=b_, num_workers=num_workers)

            d_name = get_data_name(dataset, data_name_add)

            lr = lr_base_ * (2048 / dataset.max_length)
            optimizer, scheduler = setup_optimizer(model, lr=lr, epochs=n_epochs_, weight_decay=weight_decay_)

            print("####################################################################################")
            print("MODEL:", m_name)
            n_params = print_model_stats(model)
            if test_throughput: model_throughput(deepcopy(model), model.vocab_size, d_input=d_input, e=n_epochs_,
                                                 len_data_loader=len(train_loader), b=b, L=dataset.max_length)
            print("DATA:", d_name)
            if test_throughput: data_throughput(train_loader)
            print(f"hparams: e:{n_epochs_}, b:{b_}, lr:{lr}, w_d:{weight_decay_}, L:{dataset.max_length}, drop:{model.dropout}")

            if test_mode or not wandb_logging:
               wandb_run = None
            else:
               print("Logging with wandb! Happens after 2. epoch!")
               wandb_run = partial(wandb.init, project=d_name+wandb_name, name=m_name,
                                   config={"model":m_name, "data":d_name, "lr":lr, "b": b_, "weight_decay":weight_decay_,
                                           "n_layer":model.n_layer, "d_state":model.d_state, "dropout": model.dropout,
                                           "d_model":model.d_model, "n_params": n_params})
            #print("sum params", sum([param.sum() for param in model.parameters() if param.requires_grad]))
            _ = trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, test_mode=test_mode,
                        criterion=criterion, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs_,d=d,
                        wandb_run=wandb_run, classification=dataset.classification, bi=bool(model.bi_module))
            dataset.max_length = max_length_default
            model = model.to("cpu")
            run += 1

















