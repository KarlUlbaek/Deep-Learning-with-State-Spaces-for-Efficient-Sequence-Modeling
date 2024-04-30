import torch
from torch.nn import CrossEntropyLoss
import sys, os
sys.path.append(os.getcwd())


import wandb
from copy import deepcopy
from functools import partial

from s4_playground.misc import (setup_optimizer, print_model_stats, model_throughput, data_throughput,
                                get_model_name, get_data_name, get_data_dim, trainer)

from mamba_fork.mamba_ssm.models.mixer_seq_simple import MambaModel
from s4_modules import S4ClassicModel

from s4_playground.lra import IMDB, PathFinder
from s4_fork.src.dataloaders.basic import CIFAR10

# my own desktop is old and doesnt support some fused operations
if torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080 Ti":
   fast = False
else:
   fast = True



data = CIFAR10("cifar")
data.setup("../data/cifar10")
CIFAR10cont = deepcopy(data)
data.tokenize = True
data.grayscale = True
data.setup("../data/cifar10")
CIFAR10token = deepcopy(data)

# data = IMDB("imdb")
# data.l_max = 1024
# data.setup("../data/imdb")
# IMDBtoken = deepcopy(data)
#
# data = PathFinder("pathfinder")
# data.setup("../data")
# #data.setup("../data")
# Pathfindercont = deepcopy(data)
# data.tokenize = True
# data.setup("../data")
# #data.setup("../data")
# Pathfindertoken = deepcopy(data)


#pos_emb = {"loc": "all", "theta": 10_000, "seq_norm": 1024, "learned_freq": False, "b_c_dt_x": "b_c_dt"}
d_model = 116
d_state = 16
n_layer = 6
dropout = 0.15
m1 =   [partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                fused_add_norm=fast, rms_norm=fast)]

m2 =   [partial(MambaModel, n_layer=n_layer, d_model=int(0.93*d_model), d_state=d_state, dropout=dropout,
                fused_add_norm=fast, rms_norm=fast, bi_s6={"bi":True})]


m3 =   [partial(MambaModel, n_layer=n_layer, d_model=d_model, d_state=d_state, dropout=dropout,
                fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs"})]

m4 =   [partial(MambaModel, n_layer=n_layer, d_model=int(0.94*d_model), d_state=d_state, dropout=dropout,
                fused_add_norm=fast, rms_norm=fast, s4_kwargs={"mode": "diag", "init": "legs", "bi": "sequential_bi"})]

d_state = 64
d_model = 170
m5 =    [partial(S4ClassicModel, n_layer=n_layer, d_model=d_model,
                     d_state=d_state, dropout=dropout, s4_kwargs={"mode": "diag", "init": "legs"})]

m6 =    [partial(S4ClassicModel, n_layer=n_layer, d_model=int(0.815*d_model),
                     d_state=d_state, dropout=dropout, s4_kwargs={"mode": "diag", "init": "legs", "bi": "sequential_bi"})]

#models = [m2]#, m2, m3, m4, m5, m6]
models = m2 + m1 + m3 + m4 + m5 + m6
datasets = [CIFAR10cont]

n_epochs = 25
b = 64
num_workers = 0
d = "cuda"
lr = 3e-3
lr_scale = 0.1 # 0.1
weight_decay = 0.01 # 0.01


criterion = CrossEntropyLoss()
test_throughput = True
run_test_run = True
wandb_logging = True
wandb_name = "_bi_v3" #""
data_name_add = ""
model_name_add = ""

test_modes = [True, False] if run_test_run else [False]
print("datasets:", [dataset.__class__.__name__ for dataset in datasets])
print("models:", [model.func.__name__ for model in models])
for test_mode in test_modes:
   for dataset in datasets:
      train_loader = dataset.train_dataloader(batch_size=b, num_workers=num_workers, shuffle=True)
      val_loader = dataset.val_dataloader(batch_size=b, num_workers=num_workers)
      test_loader = dataset.test_dataloader(batch_size=b, num_workers=num_workers)

      for model in models:

         d_input, d_output, vocab_size, L = get_data_dim(train_loader, dataset)
         model = model(d_input=d_input, d_output=d_output, vocab_size=vocab_size, classification=True).to(d)
         m_name = get_model_name(model, model_name_add)
         d_name = get_data_name(dataset, data_name_add, cons_or_token=vocab_size)

         lr_L = lr * (1024 / L)
         optimizer, scheduler = setup_optimizer(model, lr=lr_L, epochs=n_epochs, weight_decay=weight_decay)

         print("####################################################################################")
         print("MODEL:", m_name)
         n_params = print_model_stats(model)
         if test_throughput:
            model_throughput(deepcopy(model), model.vocab_size, d_input=d_input, e=n_epochs,
                             len_data_loader=len(train_loader), b=b, L=L)
         print("DATA:", d_name)
         if test_throughput:
            data_throughput(train_loader)

         print(f"hparams: e:{n_epochs}, b:{b}, lr:{lr_L}, "
               f"w_d:{weight_decay}, L:{L}, drop:{model.dropout}")

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
                     criterion=criterion, optimizer=optimizer, scheduler=scheduler, n_epochs=n_epochs, d=d,
                     wandb_run=wandb_run, classification=dataset.classification, bi=bool(model.bi_module))
         model = model.to("cpu")













