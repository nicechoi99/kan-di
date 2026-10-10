# Benchmark pools

These are the nine candidate pools the paper's benchmark runs select from, exactly as used
(`benchmark/main.py` reads them through `common/datamanager.load_dataset`; the arrays are bit-identical
to the ones behind the reported results).

Each `<group>/<name>.csv` holds one row per candidate. All columns but the last are the descriptors,
and the last column is the objective. The pools are preprocessed, not raw.

- Duplicate descriptor rows are averaged.
- The high-dimensional sets are reduced to at most 50 descriptors.
- Descriptors and the objective are min–max scaled to `[0.1, 0.9]`.
- The objective is oriented so that larger is better. AgNP, Perovskite, dilute-solute diffusion,
  metallic-glass forming and polymer `C_p` are minimization targets in their sources and appear inverted.
- Perovskite's instability index is log-transformed before scaling.

`<group>/embedding/<name>.csv` holds the 2-D t-SNE coordinates used only by the BO monitoring plots.

| Group | Pool | Candidates | d | Source | License of the source |
|---|---|---:|---:|---|---|
| small_feature | AgNP | 164 | 5 | Liang et al., *npj Comput. Mater.* **7**, 188 (2021), [PV-Lab/Benchmarking](https://github.com/PV-Lab/Benchmarking) | MIT |
| small_feature | AutoAM | 100 | 4 | same | MIT |
| small_feature | P3HT | 178 | 5 | same | MIT |
| small_feature | Perovskite | 94 | 3 | same | MIT |
| small_feature | Crossed barrel | 600 | 4 | same | MIT |
| large_feature | dilute_solute_diffusion | 408 | 27 | Wu, Mayeshiba & Morgan, *Sci. Data* **3**, 160054 (2016), via [MAST-ML Education Datasets](https://figshare.com/articles/dataset/MAST-ML_Education_Datasets/7017254) | CC BY 4.0 |
| large_feature | metallic_glass_forming | 585 | 22 | Ward et al., *npj Comput. Mater.* **2**, 16028 (2016), via MAST-ML Education Datasets | CC BY 4.0 |
| large_feature | MOF_Td | 3131 | 50 | Nandy, Duan & Kulik, *J. Am. Chem. Soc.* **143**, 17535 (2021), [Zenodo 5508357](https://zenodo.org/records/5508357) | CC BY 4.0 |
| large_feature | polymer_Cp | 67 | 50 | Kim et al., *J. Phys. Chem. C* **122**, 17575 (2018), [Materials Cloud rgw38-xtf82](https://archive.materialscloud.org/records/rgw38-xtf82) | CC BY 4.0 |

If you use these pools, cite the original sources above as well as this repository. The
amine-screening campaign data of the paper are not part of this release.

To rebuild a pool from a raw source CSV, use `datamanager.preprocessing` (needs
`requirements-optional.txt` for the embedding). The t-SNE coordinates it produces depend on the
library version; the descriptors and objective do not.
