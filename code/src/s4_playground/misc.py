import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR as CosSched

def setup_optimizer(model, opt=AdamW, Sched=CosSched, lr=1e-3, lr_scale = 0.1, weight_decay=0.01, epochs=100):
   """
   S4 requires a specific optimizer setup.

   The S4 layer (A, B, C, dt) parameters typically
   require a smaller learning rate (typically 0.001), with no weight decay.

   The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
   and weight decay (if desired).
   """

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