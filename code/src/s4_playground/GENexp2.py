
import torch
from torch.nn import CrossEntropyLoss
import sys, os
sys.path.append(os.getcwd())
from s4_playground.misc import setup_optimizer, print_model_stats, model_throughput, data_throughput

def fuse_model(m1, m2, tmp):

   for i in range(6):
      #swap in mamba modules from the 2 different models to be the new farward and backjward model
      tmp.layers[i].mixer.forward_model = m1.layers[i].mixer
      tmp.layers[i].mixer.backward_model = m2.layers[i].mixer

      #swap in layernorm from first model
      tmp.layers[i].norm = m1.layers[i].norm

      # tie weights
      tmp.layers[i].mixer.forward_model.in_proj.weight = tmp.layers[i].mixer.backward_model.in_proj.weight
      tmp.layers[i].mixer.forward_model.out_proj.weight = tmp.layers[i].mixer.backward_model.out_proj.weight

   #take encoder from first model
   tmp.encoder = m1.encoder
   # use random initilzied decoder

   return tmp



if __name__ == "__main__":
   from s4_playground.misc import get_model_name, get_data_name, set_model_dropout, trainer
   import wandb
   from copy import deepcopy

   from mamba_fork.mamba_ssm.models.mixer_seq_simple import MambaModel
   from genomics import Species
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

   #max_length = 1024*2
   n_data_points = 50_000
   n_epochs = 5
   b = 32

   lr_base = 1e-4
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
   Models = [s6Mamba1, s6Mamba_bi]
   sizes = [1024 * 4]*2
   training_plans = ["pretrain", "finetune"]

   pretrain_name = "pretrain_big" #"pretrain_big"
   finetune_name = "finetune_small" #"finetune_small"
   finetune_len = 1_000_000 #-1 disables i.e. takes all values

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
   data_name_add = "_v4"
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
            assert val_loader is not None, "EVAL LOADER NONE. CAOS"
            assert train_loader is not None, "TRAIN LOADER NONE. CAOS"

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

















