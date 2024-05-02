#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
#BSUB -J gen_rerun
### -- ask for number of cores (default: 1) --
#BSUB -n 6
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- send notification at start --

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load python3/3.10.11
module load cuda/11.7

pwd

source ../12venv/bin/activate

python s4_playground/GENexp2_rerun.py


