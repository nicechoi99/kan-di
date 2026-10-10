# Example dataset

`example_dataset.csv` is synthetic, so the answer is known. Six descriptors are recorded for 150
experiments, but the yield depends only on `temperature` (a peak near 108 °C) and `amine_ratio`
(monotone); `pressure`, `time`, `stir_rate` and `batch` do not enter it. Gaussian noise (sd 2) is added.

```bash
python analyze.py examples/example_dataset.csv --target yield              # descriptor importance
python analyze.py examples/example_dataset.csv --target yield --simulate   # + campaign replay
```

A correct analysis ranks `temperature` and `amine_ratio` first and gives the other four no measurable
gradient energy. On this table `temperature` carries most of the gradient energy: the peak is steep,
while the `amine_ratio` effect is monotone and shallow, and AGE measures squared slope, not effect size.

On this table the KAN fits better without the log transform of its inputs (held-out R² 0.97 against
0.09 with it), so `--log-inputs auto` selects `none`. With the log transform the expression fits
poorly and the ranking is unreliable, which the report flags.
