#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/extract_stay_id/extract_stay_id_%j.out
#SBATCH --error=slurm_logs/extract_stay_id/extract_stay_id_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          
#SBATCH --mem=64G

export APPTAINERENV_TINI_SUBREAPER=1

apptainer exec --nv /utilities/containers/py311-torch-transformers-bits.sif /bin/bash -lc '

source ~/p311-torch/.env/bin/activate
python extract_stay_id.py \
  --preprocessed_dir /scratch/ahcie-gpu2/XieF-Req04048/z_final/data/preprocessed_0.5h \
  --outcome_col outcome_all_pne \
  --n 50 \
  --seed 123 \
  --out_txt pos_stayids_all_pne.txt \
  --out_csv pos_stayids_all_pne.csv
'