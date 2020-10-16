#!/bin/sh
#PBS -N run 
#PBS -q gpu
#PBS -l mem=11887mb,nodes=2:ppn=1
#PBS -V
export PATH=/home/bt1/18CS10037/anaconda3/bin:$PATH
 . /home/bt1/18CS10037/anaconda3/etc/profile.d/conda.sh
conda activate dl 
# User Directives
cd PECNet/
nvidia-smi
cd scripts
python test_pretrained_model.py -lf PECNET_social_model1.pt
#End of script
