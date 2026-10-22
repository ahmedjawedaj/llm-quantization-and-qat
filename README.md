# LLM Quantization and Quantization-Aware Training

GPT-2 post-training quantization and ResNet-18 QAT experiments, with calibration, reconstruction diagnostics, packed export and accuracy/storage/performance analysis.

Maintained by [Ahmed Jawed](https://github.com/ahmedjawedaj).

## Methods and engineering work

- Twelve GPT-2 configurations: linear PTQ, SmoothQuant and GPTQ at static/dynamic W8A8/W4A8.
- Per-channel scales, activation observers, reconstruction errors and INT4 packing/reference checks.
- Independent eight-epoch ResNet W8A8 and W4A8 QAT runs with STE.
- Separate packed model size, runtime memory, perplexity and device measurements.

## Repository contents

- [`quantization_and_qat.ipynb`](notebooks/quantization_and_qat.ipynb)
- [`data/README.md`](data/README.md): required external inputs and data handling.
- [`PROVENANCE.md`](PROVENANCE.md): original source-date evidence, attribution and publication changes.
- [`scripts/validate_sources.py`](scripts/validate_sources.py): notebook/Python syntax and exclusion checks.

## Recorded observations

- CIFAR-10 test top-1 in saved results: FP32 94.98%, W8A8 QAT 94.61%, W4A8 QAT 94.42%.
- Selected public result tables and implementation checks are included under results/.

## Setup and use

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd notebooks
jupyter lab
```

`requirements-recorded.txt` preserves the package versions recorded with the original run. CUDA-tagged PyTorch builds need the matching PyTorch package index; the general `requirements.txt` is a starting environment, not a verified lockfile.

Select the intended kernel and run the notebook top to bottom after providing its required inputs. GPU notebooks can consume substantial runtime; validation below does not start training.

## Data and weights

Public GPT-2, WikiText-2 and CIFAR-10 are loaded by the notebook. Large datasets, weights and caches are not committed.

## Validation

```bash
python -m pip install -r requirements-validation.txt
python scripts/validate_sources.py
```

The publication checks validate notebook structure and compile source cells/Python modules. They do not reproduce training, GPU timing or ROS hardware behaviour. Cleaned notebooks have no captured outputs or stale execution counters.

## Scope and limitations

- Quantization is simulated around FP32 kernels. These timings do not establish native integer-kernel acceleration.
- Checkpoint exposure to the later validation pool is discussed in the study.
- Saved results are artifact evidence; the publication checks do not retrain models.

## References and attribution

- [SmoothQuant](https://github.com/mit-han-lab/smoothquant)
- [GPTQ](https://github.com/IST-DASLab/gptq)

See PROVENANCE.md for the distinction between original work dates and actual publication dates. This repository excludes instructor handouts, raw data, model weights and private records.
