# Worked example: AutoAM

The example uses AutoAM, the smallest of the paper's benchmark pools (`dat/small_feature/AutoAM.csv`;
source and license in `dat/README.md`). It records 100 extrusion prints from an autonomous additive
manufacturing study. The four printer settings `Prime Delay`, `Print Speed`, `X Offset Correction` and
`Y Offset Correction` are the descriptors, and the print-quality `Score` is the response (higher is
better). The values are already scaled to `[0.1, 0.9]`, as in the paper.

```bash
python analyze.py dat/small_feature/AutoAM.csv --target Score --log-inputs on                       # descriptor importance
python analyze.py dat/small_feature/AutoAM.csv --target Score --log-inputs on --simulate --seeds 3  # + campaign replay
```

The paper finds that the two offset corrections lead on this pool. `analyze.py` should rank them
first and second, with `Prime Delay` and `Print Speed` well behind.

The campaign replay takes `Score` above 0.82 (the top 10% of the scaled range) as the target and
counts KAN-DI and ZERO-EI iterations from the same random initial designs the paper uses. Over ten
seeds the paper reports a median of 5 iterations for KAN-DI (`DI-UCB-H`) and 32 for ZERO-EI on this
pool. `benchmark/main.py --dataset small_feature --data AutoAM --methods KAN:DI-UCB-H,ZERO:EI`
reruns that comparison.

The same example runs in Colab (`kan_di_colab.ipynb`) and in the web app (`app/streamlit_app.py`,
"Use the AutoAM benchmark pool").
