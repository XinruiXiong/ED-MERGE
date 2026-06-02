#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/create_multimodal_dataset/create_multimodal_dataset_%j.out
#SBATCH --error=slurm_logs/create_multimodal_dataset/create_multimodal_dataset_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          
#SBATCH --mem=64G

export APPTAINERENV_TINI_SUBREAPER=1

apptainer exec --nv /utilities/containers/py311-torch-transformers-bits.sif /bin/bash -lc '

source ~/p311-torch/.env/bin/activate
python create_multimodal_dataset.py \
  --master_parquet "/scratch/ahcie-gpu2/XieF-Req04048/z_final/data/proc_data/master_dataset_clean.parquet" \
  --notes_dir "/scratch/ahcie-gpu2/XieF-Req04048/z_final/data/proc_data/notes_all_parts" \
  --notes_stay_col SERVICE_ID \
  --notes_time_col FILING_DATE \
  --notes_text_col NOTE_TEXT \
  --vitals_file "/scratch/ahcie-gpu2/XieF-Req04048/z_final/data/proc_data/VITALS_SUBSET.tsv" \
  --output_dir "/scratch/ahcie-gpu2/XieF-Req04048/z_final/data/new/preprocessed_6h" \
  --time_window 6h \
  --train_year_le 2022 \
  --test_year_ge 2023 \
  --val_ratio 0.1 \
  --seed 42
'