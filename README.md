# Constant Overhead Entanglement Distillation via Scrambling

Code repository for generating figures in "Constant Overhead Entanglement Distillation via Scrambling".

## Overview

This repository contains Python scripts that generate the figures used in the paper, comparing different entanglement distillation protocols including our scrambling-based approach, DEJMPS protocol, Pattison et al.'s method, and hashing.

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

## Generating Figures

To generate all figures, run:

```bash
uv run python src/generate_app_figure.py
uv run python src/generate_concat_small_figure.py
uv run python src/generate_evolution_small_figure.py
uv run python src/generate_phase_figure.py
```

Generated figures will be saved in the `figs/` directory.

## Figure Descriptions

- **app.pdf**: Comparison of different quantum key distribution protocols showing overhead vs distance and secret bit rates at various error rates
- **concat-small.pdf**: Protocol performance with concatenation
- **evolution-small.pdf**: Evolution of the distillation process
- **phase.pdf**: Phase diagram showing protocol performance regions

## Citation

If you use this code in your research, please cite:

```
@misc{gu2025distillation,
      title={Constant Overhead Entanglement Distillation via Scrambling}, 
      author={Andi Gu and Lorenzo Leone and Kenneth Goodenough and Sumeet Khatri},
      year={2025},
      eprint={2502.09483},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2502.09483}, 
}
```