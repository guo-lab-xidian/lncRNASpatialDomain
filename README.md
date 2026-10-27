# lncRNASpatialDomain

Analysis code for:

> **What can long non-coding RNAs tell us about spatial domains? A multi-platform
> reassessment in mouse and human brain**
> Xingli Guo, Xidian University

The study asks whether lncRNAs carry more spatial-domain information than
protein-coding genes expressed at a comparable level. It combines an
isoform-resolved MERFISH atlas of mouse brain (two independently designed probe
panels) with mouse and human 10x Visium, and it reports three things that are
easy to miss in class-level comparisons: a detection-dependent **noise floor**
in domain-specificity indices, a strong dependence on the **detection-rate
matching bandwidth**, and a **panel-coverage** constraint that limits what most
platforms can measure at all.

---

## 1. Data

No data are redistributed here. All inputs are public.

| Dataset | Source | Notes |
|---|---|---|
| MERFISH mouse brain (WT and ISO panels) | Zenodo `10.5281/zenodo.20682863` | ~14 GB; two panels, `.h5ad` plus Allen CCF coordinates |
| 10x Visium mouse (sagittal) | 10x Genomics public datasets | Space Ranger 1.1.0, mm10-2020-A |
| 10x Visium human (sagittal) | 10x Genomics public datasets | GRCh38-2020-A |
| Visium HD mouse probe set | 10x Genomics | panel composition only |
| Xenium Prime Mouse 5K | 10x Genomics | panel metadata only |
| GENCODE M25 | GENCODE | gene biotype annotation |

---

## 2. Environment

Python 3.12. Key versions (`requirements.txt` has the full list):

```
numpy 2.5.4      pandas 3.1.0      scipy 1.18.1
scanpy 1.12.4    anndata 0.13.4    scikit-learn 1.9.1
statsmodels 0.15 gseapy 1.3.1      matplotlib 3.11.2
```

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Memory.** The full MERFISH matrices hold 1.1 × 10^8 (WT) and 8.4 × 10^7 (ISO)
non-zero entries. Loading either one whole exceeds 8 GB. `code/spatial_io.py`
implements a row-block streaming layer (peak RSS < 100 MB) that every MERFISH
script uses; do not replace it with a plain `anndata.read_h5ad`.

---

## 3. Directory layout

The scripts were written for the following layout and refer to it by absolute
path. Reproduce it, or retarget the paths (section 4):

```
<ROOT>/
├── data/
│   ├── <MERFISH h5ad files>
│   ├── visium/cache/          # produced by code/32_visium_load.py
│   ├── visium_human/
│   └── gencode.vM25.annotation.gtf.gz
├── analysis/                  # QC and annotation intermediates
└── lncRNA空间域/
    ├── results/               # all outputs (CSV, figures)
    └── manuscript/            # section files assembled into the paper
```

---

## 4. Retargeting paths

Every script contains the original root as an absolute path. To point them at
your own checkout:

```bash
python retarget_paths.py /path/to/your/root
```

On Windows this rewrites, for example,
`r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"` to
`r"/path/to/your/root\lncRNA空间域\results"` in place. A backup of each file is
written as `*.py.bak`. **Review the diff before running anything.**

---

## 5. Pipeline

Scripts are numbered in execution order. Each stage writes into
`results/` and is consumed by later stages.

### 5.1 Data preparation

| Script | Produces |
|---|---|
| `01_qc.py` | per-cell QC metrics → Fig S1 |
| `02_annotation.py`, `03_annotation_v2.py` | gene biotypes (Ensembl with GENCODE M25 fallback, 99.9% resolved) |
| `10_gene_coords.py` | `results/gene_coords_m25.tsv` |
| `32_visium_load.py` | Visium cache (counts, coordinates) |

### 5.2 Spatial domains and specificity

| Script | Produces |
|---|---|
| `12_build_domain.py` | `L-domain` partitions → Fig S2 |
| `13_domain_specificity.py` | DSI for every gene → Fig 1, Table S7 |
| `14_spatial_autocorr.py` | Moran's *I* → Fig 6 |
| `33_visium_domain.py`, `39_visium_human.py` | Visium mouse / human equivalents |

### 5.3 The three main analyses

| Script | Produces |
|---|---|
| `54_matching_sensitivity.py` | matched ΔDSI at five caliper widths → Table S8 |
| `37_neg_result_robustness.py`, `38_interaction_test.py` | decile strata and interaction model → Fig 3 |
| `56_dsi_null_check.py` | null-model noise-floor curve → Fig 4A–B |
| `62_noise_adjusted_v2.py` | standardised comparison, per-gene nulls → Tables S9–S11, Fig 4C |
| `63_moran_robustness.py` | partition-free Moran's *I* check → Table S12 |
| `34_visium_hd.py`, `probe_panel_composition.py` | platform coverage survey → Fig 5 |

> **Note.** `57_noise_adjusted_dsi.py` is an earlier version of `62_` and is kept
> only for the record. It applies a binomial null to the Visium index, which is
> computed on normalised expression rather than detection, and it shares one
> random stream between null simulation and control sampling. Its Visium numbers
> are wrong. **Use `62_noise_adjusted_v2.py`.**

### 5.4 The catalogue

| Script | Produces |
|---|---|
| `40_catalog_validation.py` | bootstrap rank stability → Fig 6B |
| `25_lncrna_annotation.py`, `26_go_enrichment.py`, `29_coexpr_stratified.py`, `30_coexpr_depth_adjusted.py` | depth-corrected co-expression and GO |
| `50_final_catalog.py` | the 15-gene catalogue → Table S1 |

### 5.5 Figures and manuscript

| Script | Produces |
|---|---|
| `64_figures_bib.py` | main figures 1–7 (vector PDF + 600 dpi PNG, 180/88 mm) |
| `65_supp_figures_bib.py` | supplementary figures S1–S6 |
| `51_figures_main.py`, `52_figure_catalog.py`, `53_supplementary_figures.py` | earlier drafts of the same figures |
| `60_assemble_manuscript.py` | assembles section files and runs consistency checks |

### 5.6 Negative results, reported in the paper

| Script | What it tested |
|---|---|
| `11_cis_coupling.py`, `11b_cis_aggregate.py` | cis-neighbourhood coupling → Fig S3a |
| `21_simulation.py`, `22_apply_method.py`, `23_effect_size_rank.py` | three alternative estimators vs a simple Pearson statistic → Fig S3b, S5 |
| `24_isoform_divergence.py` | isoform-level spatial divergence → Fig S4 |
| `41_allen_check.py` … `44_test_atlas_id.py` | Allen ISH validation attempt → Fig S6 |

---

## 6. Conventions

- **MERFISH analyses use binarised detection**, not normalised expression: the
  median cell yields 592 (WT) and 471 (ISO) detected genes, so mean–variance
  statistics are noise-dominated. Visium uses log-normalised expression.
- **Domain specificity index** is `DSI(g) = max_k f_k(g) − median_k f_k(g)`.
- **Figures** are produced at final publication size; fonts are set for 7 pt at
  180 mm width, so do not rescale them afterwards.
- All random seeds are fixed and reset per dataset. The control-sampling stream
  is deliberately separate from the null-simulation stream so that the raw and
  noise-corrected comparisons share the same matched controls.

---

## 7. Contact

Xingli Guo, School of Computer Science and Technology, Xidian University —
xlguo@mail.xidian.edu.cn
