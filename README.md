# Anonymous code release — Soft Compressor Retargeting (ACL review)

Bundle for **RQ1–RQ3** analysis (CSVs + scripts), plus the **training / transfer / evaluation** pipeline (`src/`, `scripts/`). No paper LaTeX, no checkpoints, no internal server paths.

## Layout

| Area | Path | Contents |
|------|------|----------|
| **RQ1** | `rq1/data/` | BLEU CSVs (origin, ori-transfer, LS, random) |
| **RQ2** | `rq2/` | Perturbation + native/pair geometry code; `results/` CSVs |
| **RQ3** | `rq3/` | Pre-transfer prior code; `results/` (scores, priors, analysis CSVs) |
| **Core library** | `src/soft_compress/` | Compressor, converters (LS/BP/Procrustes), ori-transfer |
| **Launchers** | `scripts/soft_compress/simple_compressor/` | Train, evaluate, LS/random transfer, `recover_bleu` |
| **Sample data** | `datasets/simple_mem/` | Tiny JSON sample + `process_data.py` |

## Environment

```bash
export PROJECT_ROOT="$(pwd)"
export MODELS_DIR="${MODELS_DIR:-/path/to/huggingface_models}"
export DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
pip install -r requirements.txt
```

Evaluation JSON: `${DATA_ROOT}/fineweb_test.json` (`{"text": "..."}` items).

## Training & transfer (scripts + src)

| Step | Command |
|------|---------|
| Train compressors (grid) | `bash scripts/soft_compress/simple_compressor/run_train_all_compressors.sh` |
| Single compressor | `bash scripts/soft_compress/simple_compressor/run_compressor_single.sh <enc> <dec> <mem> <len>` |
| Evaluate native compressor | `bash scripts/soft_compress/simple_compressor/run_evaluate_origin.sh ...` |
| Ori-transfer sweep | `bash scripts/soft_compress/simple_compressor/transfer_compressor/rerun_transfer_all.sh` |
| LS transfer eval | `bash scripts/soft_compress/simple_compressor/transfer_compressor/eval_ls_transfer.sh` |
| Random baseline | `bash scripts/soft_compress/simple_compressor/transfer_compressor/eval_random_transfer.sh` |
| RQ1 BLEU from saved evals | `scripts/.../transfer_compressor/recover_bleu_from_saved_evals.py` |

DeepSpeed config: `scripts/soft_compress/simple_compressor/ds_config_zero1_bf16.json`.  
LS vocab: `scripts/soft_compress/simple_compressor/data/vocab_100k.txt`.

## RQ1

Bundled CSVs under `rq1/data/` — deduplicated BLEU metrics for paper tables (`metric_source=saved_generated_text_vs_reference`).

## RQ2

- `python rq2/perturbation/memory_perturbation_sensitivity.py` — Gaussian / control noise on memory vectors
- `bash rq2/run_perturbation_suite.sh` — batch driver (needs GPU + checkpoints)
- `python rq2/native_geometry/native_memory_geometry.py` — native memory geometry
- `python rq2/pair_geometry/source_target_pair_geometry.py` — directionality (uses RQ3 scores)

Precomputed: `rq2/results/perturbation/`, `native_geometry/`, `pair_geometry/`, `tables/`.

## RQ3

Canonical outputs: `rq3/results/analysis_mem{8,16,32}/full/{enc_conv,converter_only}/` (full target pool).  
Primary priors: `rq3/results/priors/priors_consolidated_mem*_with_pair_metrics.csv`.

- **Compute priors** (GPU): `python rq3/compute_pretransfer_priors.py` + checkpoint / model paths
- **Evaluate** (CPU): `python rq3/rq3_evaluate_priors.py --mem-tokens 32 --label enc_conv --suite full --prior-csv rq3/results/priors/priors_consolidated_mem32_with_pair_metrics.csv`

Bundled: `rq3/results/standard_scores/`, `priors/`, `analysis_mem*/**/*.csv`, controlled robustness CSVs at `rq3/results/rq3_controlled_*.csv`.

## Anonymity

Paths use `${PROJECT_ROOT}`, `${MODELS_DIR}`, `${DATA_ROOT}`. Do not commit credentials or machine-specific absolute paths.
