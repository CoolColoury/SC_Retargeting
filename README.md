# Anonymous code release — Soft Compressor Retargeting (ACL review)

Minimal bundle for **RQ1** (transfer evaluation CSVs), **RQ2** (Gaussian perturbation + native/pair geometry), and **RQ3** (pre-transfer priors). No paper LaTeX, no training checkpoints, no internal server paths.

## Layout

| Research question | Path | Contents |
|-------------------|------|----------|
| **RQ1** | `rq1/data/` | Shift-aligned BLEU tables: origin, ori-transfer, LS, random |
| **RQ2** | `rq2/` | `perturbation/`, `native_geometry/`, `pair_geometry/` scripts; `results/` CSVs |
| **RQ3** | `rq3/` | Prior computation & evaluation scripts; `results/` (scores, priors, analysis CSVs) |
| **Shared** | `src/soft_compress/simple_compressor/simple_compressor.py` | Loader used by RQ2/RQ3 GPU scripts |

## Environment

```bash
export PROJECT_ROOT="$(pwd)"
export MODELS_DIR="${MODELS_DIR:-/path/to/huggingface_models}"
export DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
pip install -r requirements.txt
```

Example evaluation JSON for RQ2/RQ3: `${DATA_ROOT}/fineweb_test.json` (same schema as FineWeb-style `{"text": "..."}` items).

## RQ1

Bundled CSVs under `rq1/data/` are the deduplicated, shift-aligned reconstruction metrics used in the paper tables (no re-run required for review).

## RQ2

- **Gaussian / control perturbations**: `python rq2/perturbation/memory_perturbation_sensitivity.py` (see flags in file). Suite driver: `bash rq2/run_perturbation_suite.sh`.
- **Native memory geometry**: `python rq2/native_geometry/native_memory_geometry.py`; driver: `bash rq2/native_geometry/run_native_memory_geometry.sh`.
- **Pair directionality** (uses RQ3 scores): `python rq2/pair_geometry/source_target_pair_geometry.py`.

Precomputed CSVs: `rq2/results/perturbation/`, `native_geometry/`, `pair_geometry/`, `tables/`.

## RQ3

- **Compute priors** (GPU): `python rq3/compute_pretransfer_priors.py` with `--source_checkpoint`, `--compressor_model`, `--source_decoder_model`, `--target_model`, `--dataset`, `--output`.
- **Evaluate priors** (CPU): `python rq3/rq3_evaluate_priors.py --mem-tokens 32 --label enc_conv --suite full --prior-csv rq3/results/priors/priors_consolidated_mem32.csv`

Bundled: `rq3/results/standard_scores/`, `rq3/results/priors/`, `rq3/results/analysis_mem*/**/*.csv`.

## Anonymity

Paths and hostnames are placeholders (`${PROJECT_ROOT}`, `${MODELS_DIR}`, `${DATA_ROOT}`). Do not commit real credentials or absolute machine paths.
