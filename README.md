# HW1 — Analytical performance model

## Setup

- GPU: RTX 2070 (8GB)
- Python 3.13.5
- PyTorch 2.11.0+cu128
- CUDA 12.8
- NumPy 2.2.4
- SciPy 1.15.3
- Matplotlib 3.10.1
- nvidia-ml-py 13.615.71

## Reproduce

From the repo root, with the existing `.venv`:

```bash
.venv/bin/python measure.py
.venv/bin/python calibrate.py
.venv/bin/python build_plots.py
```

`measure.py` appends to `results/measurements.csv` and skips `(S, B)` pairs already in the file. `calibrate.py` writes `results/theta.json`. `build_plots.py` writes `results/figures/*.png`.

## Results summary

TODO (я заполню)

## Discussion (1 page)

TODO (я заполню — объяснение launch-bound/memory-bound/compute-bound переходов)
