#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/train_multimodal/train_multimodal_%j.out
#SBATCH --error=slurm_logs/train_multimodal/train_multimodal_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

export APPTAINERENV_TINI_SUBREAPER=1

apptainer exec --nv /utilities/containers/py311-torch-transformers-bits.sif /bin/bash -lc '

source ~/p311-torch/.env/bin/activate
python train_multimodal.py \
  --preprocessed_dir "/scratch/ahcie-gpu2/XieF-Req04048/z_final/data/new/preprocessed_6h" \
  --bert_dir "/utilities/models/Bio_ClinicalBERT" \
  --output_dir "/scratch/ahcie-gpu2/XieF-Req04048/z_final/output/new/multimodal_model_output_6h" \
  --epochs 4 \
  --max_notes 100 \
  --batch_size 8 \
  --grad_acc 4
'