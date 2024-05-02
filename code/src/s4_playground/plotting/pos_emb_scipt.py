# def rename_columns(path = "data/cifar10_cons.csv"):
import pandas as pd
import numpy as np
def OSL_data(path="data/exp2/IMDBtoken_pos2_val.csv", test_or_val = "test"):
   #"test acc","train acc","val acc"
   df = pd.read_csv(path)
   print(df.shape)

   names = list(df["Name"])

   not_all = [idx for idx, name in enumerate(names) if "\'first\'" in name or "\'everyother\'" in name]
   print("not all", not_all)
   assert len(not_all) == 0, "there are fist and everyother"

   df = df[df.State == "finished"]
   # df = df.rename(columns={name: name.lower()[:-11] for name in list(df.columns)})

   names = [name.lower() for name in list(df["Name"])]
   imdb = True if "IMDB" in path else False
   vars = ["mamba", "diag", "10000", "1024", "true"]
   bonus_vars = ["b", "c", "dt", "x"]
   final_vars = ["no_emb", "imdb", "acc"]
   all_vars = vars + bonus_vars + final_vars
   num_vars = len(all_vars)
   # old_names = list(df.columns)

   binary_data = np.zeros((len(names), num_vars))
   for name_dix, name in enumerate(names):
      for var_idx, var in enumerate(vars):
         if var in name:
            binary_data[name_dix, var_idx] = 1.

         # get b,c,dt,x for s6
         if "s6" in name and "[" in name:
            subset = name.split(" ")[-1]
            for var_idx2, var in enumerate(bonus_vars):
               if var in subset:
                  binary_data[name_dix, len(vars) + var_idx2] = 1.

      # get acc and dataset
      acc = df[test_or_val+" acc"].values[-1]
      binary_data[name_dix, -1] = acc
      if imdb:
         binary_data[name_dix, -2] = 1.

      if len(name.split(" ")) < 3 or not " " in name:
         binary_data[name_dix, -3] = 1.

   return binary_data, all_vars


imdb_ols_data, all_vars = OSL_data(path="data/exp2/IMDBtoken_pos2_val.csv")
cifar10_ols_data, _ = OSL_data(path = "data/exp2/CIFAR10cons_pos2_val.csv")