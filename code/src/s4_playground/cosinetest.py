import matplotlib.pyplot as plt
import torch

class tmp(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.p = torch.nn.Parameter(torch.randn(3,3))

opt = torch.optim.AdamW(tmp().parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 1, 2)

l = []
for epoch in range(50):
   scheduler.step()
   l.append(scheduler.get_last_lr()[0])

plt.plot(l)
plt.show()


def f(i, d, theta):
   return theta ** (-2 * (i) / d)


d = 512
theta = 10
data = [f(i,d, theta) for i in range(int(d/2))]
plt.plot(data)
plt.show()


