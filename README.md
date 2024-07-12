# Code repository belonging to my master’s thesis


## Credits

This repo features multiple forks from other repos. Typically the forks has been highly modified by me but all credit goes to the respective repos:

https://github.com/HazyResearch/hyena-dna

https://github.com/state-spaces/mamba

https://github.com/state-spaces/s4

https://github.com/lucidrains/rotary-embedding-torch




## The main code developed by me is found in src/thesis_src which contains:

**DNA_dataloader.py:** Modifed dataloader from the hyenaDNA repo

**DNA_exp.py:** main function for the DNA experiments, which primarily supports the f1, f1r1 pretraining methodologies. Besides from overall functionality of 
DNA_exp.py is very similar to LRA_all.py

**LRA_all.py:** Main training function for all long range arena benchmarks. Notice that the employed LRA dataloaders are typically modified versions from the s4_fork.

**mamba_network.py:** Wrapper to create a mamba network around either s4, s4d or s6 SSP. supports bidirectionality and positional embeddings. The file originates from the mamba_fork but has been heavily modified.

**misc.py:** Contains training and testing loop as well a lot of model printing and testing functionality and other miscellaneous functionality.

**rope_fork.py:** My modified RoPE implementation, which supports all the hyperparameters described in the thesis. The original file is not mine.

**s4_modules.py:** Contains the simple-block. and a wrapper to create a network based on the simple block using either the s4 of s4d SSP. It also contains the s4 and s4d SSP themselves. Contains the s4/s4d-mambablock (which supports bidirectionality and RoPE) used to create mamba-s4/mamba-s4d by the mamba_network.py. 

**s6_modules.py:** Contains the fused as well as the nonfused s6-mamba-block used by mamba_network to create the full model. The non-fused variant supports bidirectionality and positional embeddings. 

**species_dataset.py:** Contains the species dataset used by the DNA_dataloader.py





