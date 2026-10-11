# Example: AutoAM

## Dataset

- `dat/small_feature/AutoAM.csv`; source and license in `dat/README.md`.
- 100 extrusion prints from an autonomous additive manufacturing study.
- Descriptors: `Prime Delay`, `Print Speed`, `X Offset Correction`, `Y Offset Correction`.
- Response: print-quality `Score` (higher is better), scaled to `[0.1, 0.9]`.

## Commands

```bash
python analyze.py dat/small_feature/AutoAM.csv --target Score --log-inputs on                       # descriptor importance
python analyze.py dat/small_feature/AutoAM.csv --target Score --log-inputs on --simulate --seeds 3  # simulated optimization
```

## Expected output

- `X Offset Correction` and `Y Offset Correction` rank first and second.
- Simulated optimization uses `Score > 0.82` as the target.
- Reported medians over ten seeds: 5 iterations for KAN-DI (`DI-UCB-H`) and 32 for ZERO-EI.
- `benchmark/main.py --dataset small_feature --data AutoAM --methods KAN:DI-UCB-H,ZERO:EI` reruns the comparison.

## Colab

- `kan_di_colab.ipynb` runs the same example.
