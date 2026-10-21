# ED-MERGE

Dynamic multimodal risk prediction in the emergency department. Each ED encounter is represented as one continuous stream of time-stamped events — static features, vital-sign and laboratory updates, and signed clinical notes — and a single GRU model estimates the risk of 16 outcomes at any time after arrival from the events recorded up to that time.

## Layout

```
common/               config (channels, outcomes, landmark grid), event-stream lookup, focal loss
phase0_data/          build per-encounter event streams from the UMN tables
phase1_model/         note encoder, sequence model, training, evaluation, ablation, bootstrap CIs, case trajectories
external_validation/  MIMIC-IV cohort: leakage-controlled HPI notes and event streams
figures/              exported manuscript figures
```

Run everything from the repository root. Data are not included.

## Internal pipeline (UMN)

| Step | Script | Output |
|---|---|---|
| 1 | `phase0_data/build_event_streams.py` | `{split}_static / _structured_events / _notes_events.parquet`, scalers |
| 2 | `phase1_model/stage1_finetune_bert.py` | fine-tuned BioClinicalBERT encoder |
| 3 | `phase1_model/precompute_note_embeddings.py` | cached note embeddings per split |
| 4 | `phase1_model/train_sequence_model.py` | `best_model.pt`, `best_thresholds.json` |
| 5 | `phase1_model/evaluate_landmarks.py` | metrics at every landmark + per-encounter detail |
| 6 | `phase1_model/evaluate_ablation.py` | component ablation (inference-time removal) |
| 7 | `phase1_model/bootstrap_ci.py` | patient-level bootstrap 95% CIs |
| 8 | `phase1_model/case_trajectories.py` | single-encounter risk trajectories |

```bash
python phase0_data/build_event_streams.py \
  --master_parquet MASTER --notes_dir NOTES --vitals_parquet VITALS --lab_parquet LABS \
  --dx_tables_dir DX_DIR --output_dir EVENTS --train_year_le 2022 --test_year_ge 2023

python phase1_model/stage1_finetune_bert.py --event_dir EVENTS --bert_dir BIOCLINICALBERT --output_dir STAGE1
python phase1_model/precompute_note_embeddings.py --event_dir EVENTS --bert_dir STAGE1/bert_encoder --output_dir NOTE_CACHE
python phase1_model/train_sequence_model.py --event_dir EVENTS --embed_cache_dir NOTE_CACHE --output_dir MODEL
python phase1_model/evaluate_landmarks.py --event_dir EVENTS --embed_cache_dir NOTE_CACHE \
  --ckpt_path MODEL/best_model.pt --split test --out_dir EVAL

python phase1_model/evaluate_ablation.py --event_dir EVENTS --embed_cache_dir NOTE_CACHE \
  --ckpt_path MODEL/best_model.pt --split test --configs full only_static only_text only_structured --out_dir ABL
python phase1_model/bootstrap_ci.py --detail EVAL/landmark_encounter_detail.parquet \
  --static EVENTS/test_static.parquet --out_dir CI
```

## External validation (MIMIC-IV)

Requires credentialed PhysioNet access (MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note).

```bash
# 1. leakage-controlled HPI: label every HPI sentence, keep patient-reportable history only
python -m external_validation.build_hpi_label_input MIMIC_MASTER hpi_input.jsonl
python -m external_validation.hpi_clean_label hpi_input.jsonl hpi_labels.jsonl --model QWEN2.5-7B-INSTRUCT_DIR

# 2. event streams in the UMN format (split "external"), scaled with the UMN scalers
python external_validation/build_mimic_event_streams.py --master_path MIMIC_MASTER \
  --hpi_labels hpi_labels.jsonl --ed_vitalsign_path VITALSIGN --labevents_path LABEVENTS \
  --d_labitems_path D_LABITEMS --umn_event_dir EVENTS --output_dir MIMIC_EVENTS

# 3. embed notes with the UMN encoder, then score the UMN checkpoint
python phase1_model/precompute_note_embeddings.py --event_dir MIMIC_EVENTS --splits external \
  --bert_dir STAGE1/bert_encoder --output_dir MIMIC_NOTE_CACHE
python external_validation/evaluate_external.py --event_dir MIMIC_EVENTS \
  --embed_cache_dir MIMIC_NOTE_CACHE --ckpt_path MODEL/best_model.pt --out_dir EXT_EVAL
```

## Notes

- Training scripts need `--batch_size > 1` (BatchNorm).
- Every script lists its full arguments with `--help`.
