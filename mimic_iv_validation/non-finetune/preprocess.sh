#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/preprocess/preprocess_%j.out
#SBATCH --error=slurm_logs/preprocess/preprocess_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # DataLoader 用到的 CPU 线程
#SBATCH --mem=32G

export APPTAINERENV_TINI_SUBREAPER=1

apptainer exec --nv /utilities/containers/py311-torch-transformers-bits.sif /bin/bash -lc '

source ~/p311-torch/.env/bin/activate

python preprocess.py \
  --mimic_csv /scratch/ahcie-gpu2/XieF-Req04048/FIN/data/mimic/master_dataset_with_notes_mimic.csv \
  --umn_scaler_path /scratch/ahcie-gpu2/XieF-Req04048/z_final/data/preprocessed_2h/scaler.pkl \
  --umn_train_pkl_path /scratch/ahcie-gpu2/XieF-Req04048/z_final/data/preprocessed_2h/train.pkl \
  --output_dir /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits \
  --prefix mimic_external \
  --seed 42 \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
'