import os

import numpy as np
import pandas as pd
import pickle
from functools import reduce
import torch
from glob import glob
from itertools import cycle
from torch.utils.data import DataLoader, Dataset


class ImdbDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        config = get_text_classification_config()[0]
        data_paths = {'train': "../../data/datasets/aclImdb/train",
                      'eval': "../../data/datasets/aclImdb/test"}
        split_path = data_paths[split]
        neg_path = split_path + "/neg"
        pos_path = split_path + "/pos"
        neg_inputs = zip(glob(neg_path+"/*.txt"), cycle([0]))
        pos_inputs = zip(glob(pos_path+"/*.txt"), cycle([1]))
        self.data = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data[i]
        with open(data[0], 'r') as fo:
            source = fo.read()
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = int(data[1])
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class ListOpsDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        config = get_listops_config()[0]
        data_paths = {'train': "../../data/datasets/lra_release/listops-1000/basic_train.tsv",
                      'eval': "../../data/datasets/lra_release/listops-1000/basic_val.tsv"}
        self.data = pd.read_csv(data_paths[split], delimiter='\t')
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    def __getitem__(self, i):
        data = self.data.iloc[i]
        source = data.Source
        inputs = self.tokenizer(source, max_length=self.max_length) #return_tensors='pt', truncation=True, padding='max_length'
        target = data.Target
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data)


class Cifar10Dataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        config = get_cifar10_config()[0]

        data_paths = {'train': [f"../../data/datasets/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)],
                      'eval': ["../../data/datasets/cifar-10-batches-py/test_batch"]
                     }
        print("loading cifar-10 data...")
        data_dicts = [Cifar10Dataset.unpickle(path) for path in data_paths[split]]
        print("assembling cifar-10 files..")
        self.data = reduce((lambda x, y: {b'data': np.concatenate([x[b'data'], y[b'data']], axis=0), 
                                         b'labels': np.concatenate([x[b'labels'], y[b'labels']], axis=0)}), 
                           data_dicts)
        # TODO CHECK: i think this is the right shape 
        # see: https://www.cs.toronto.edu/~kriz/cifar.html 
        #      section "Dataset layouts" discusses the memory layout of the array
        self.data[b'data'] = self.data[b'data'].reshape((-1, 3, 1024)) 
       
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
    
    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d
    
    def __getitem__(self, i):
        r, g, b = self.data[b'data'][i]
        # grayscale image (assume pixels in [0, 255])
        source = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
        inputs = self.tokenizer(source, max_length=self.max_length)
        target = self.data[b'labels'][i]
        return inputs, torch.LongTensor([target])
    
    def __len__(self):
        return len(self.data[b'data'])


def make_tensor_dataset(Dataset, dtype=torch.int16):
    name = Dataset.__name__
    os.makedirs("../../data/{}_tensors".format(name) ,exist_ok=True)
    print("makeing {} tensor data".format(name))

    for split in ["train", "eval"]:
        all_x, all_y = [], []
        data = Dataset(split=split)
        for x, y in tqdm(DataLoader(dataset=data, batch_size=60, num_workers=6, shuffle=False)):
            all_x.append(x)
            all_y.append(y)

        try:
            torch.save(torch.cat([x["input_ids"] for x in all_x]).to(dtype),
                       "../../data/{}_tensors/x_{}.pt".format(name, split))

            torch.save(torch.cat(all_y).to(torch.int16),
                       "../../data/{}_tensors/y_{}.pt".format(name, split))
        except IndexError:
            torch.save(torch.cat([x for x in all_x]).to(dtype),
                       "../../data/{}_tensors/x_{}.pt".format(name, split))

            torch.save(torch.cat(all_y).to(torch.int16),
                       "../../data/{}_tensors/y_{}.pt".format(name, split))


class LRATensor(Dataset):
    def __init__(self, name, split="train"):
        Dataset = None
        if not isinstance(name, str):
            Dataset = name
            name = name.__name__

        if not name.endswith("_tensors"):
            name += "_tensors"

        assert name in ["ImdbDataset_tensors_tensors", "Cifar10DatasetToken_tensors",
                        "ListOpsDataset_tensors", "Cifar10DatasetCont_tensors"] , (
            "{} name is not supported".format(name))

        assert split in ["train", "eval"]

        if (not os.path.exists("../../data/{}/x_{}.pt".format(name, split)) or
            not os.path.exists("../../data/{}/y_{}.pt".format(name, split))):
            assert Dataset is not None
            print("making tensor dataset!")
            make_tensor_dataset(Dataset)


        x = torch.load("../../data/{}/x_{}.pt".format(name, split))
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            self.x = x.to(torch.long).squeeze()
        else:
            self.x = x.to(torch.float32)

        self.y = torch.load("../../data/{}/y_{}.pt".format(name, split)).to(torch.long).squeeze()
        #if needtranspose: self.x = self.x.transpose(-1, -2)


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

import torchvision
from einops import rearrange
ROOT_data = "../../data/cifar10/"
class Cifar10DatasetCont(Dataset):
    def __init__(self, split="train", d="cpu"):
        if split == "train":
            split = True
        data = torchvision.datasets.CIFAR10(root=ROOT_data, train=split, download=False)
        x = torch.from_numpy(data.data).to(torch.float)
        x = rearrange(x, "a b c d -> a (b c) d")
        x = x / x.max()
        self.x = x.to(d)
        self.y = torch.tensor(data.targets).to(d).to(torch.long)

    def __len__(self):
         return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



if __name__ == "__main__":
    from tqdm import tqdm
    from lra_config import (get_listops_config, get_cifar10_config, get_text_classification_config)

    # data = ListOpsDataset()

    make_tensor_dataset(Cifar10DatasetCont, dtype=torch.float16)

    for d in [ImdbDataset, Cifar10Dataset, ListOpsDataset]:
        data = LRATensor(d)
        for x, y in tqdm(DataLoader(data, batch_size=64, num_workers=6)):
            pass


    workers = 6
    data = ImdbDataset(split="eval")
    x,y = next(iter(DataLoader(dataset=data, batch_size=10, num_workers=workers, shuffle=True)))

    make_tensor_dataset(ImdbDataset)


    #all_x, all_y = [], []
    # for x,y in tqdm(DataLoader(dataset=data, batch_size=1000, num_workers=workers, shuffle=True)):
    #     all_x.append(x)
    #     all_y.append(y)
    #     pass
    #print("")

    # data = ListOpsDataset(config=get_listops_config()[0])
    # x,y = next(iter(DataLoader(dataset=data, batch_size=64, num_workers=6, shuffle=True)))
    # for x,y in tqdm(DataLoader(dataset=data, batch_size=64, num_workers=6, shuffle=True)):
    #     pass

    #
    # cifar10Dataset = Cifar10Dataset(config=get_cifar10_config()[0])
    # x,y = next(iter(DataLoader(dataset=cifar10Dataset, batch_size=64, num_workers=6, shuffle=True)))
    # for x,y in tqdm(DataLoader(dataset=cifar10Dataset, batch_size=64, num_workers=6, shuffle=True)):
    #     pass


