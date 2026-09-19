# Osteosarcoma CCL13–ACKR4 reproducibility archive

This repository accompanies the manuscript **“Biochemical and cross cohort audit of a database derived ACKR4 signal in osteosarcoma.”** It contains the source tables and analysis/figure-generation scripts used for the reported pharmacology-informed bulk, single-cell, and spatial analyses.

## Contents

- `Figure3_scRNA_detection.tsv` and `Figure3_spatial_detection_and_correlation.tsv`: source tables for the cross-modal detection summary.
- `quality_upgrade_v3/`: patient-level single-cell and spatial robustness outputs.
- `quality_upgrade_v4/`: TARGET-OS expression, clinical-model, and repeated cross-validation outputs.
- `quality_upgrade_v5/`: independent GSE42352 context and chemotherapy-response analyses.
- `quality_upgrade_v7/`: ACKR4 ligand curation, fixed six-ligand TARGET-OS models, and cross-modal detectability summaries.
- `JBO_regeneration/`: consolidated tables used to regenerate the Journal of Bone Oncology manuscript results.
- `figure_generation/`: the manuscript figure-generation script.
- `figure_generation_final/`: title-free four-main-figure and three-supplementary-figure assembly script.

## Public datasets

The final manuscript uses de-identified public data from TARGET-OS and GEO accessions GSE42352, GSE152048, GSE162454, GSE270231, GSE293065, and GSE299025. Raw public matrices are not duplicated here; retrieve them from the Genomic Data Commons or GEO using the listed identifiers. Earlier audit tables retained in the archive may reference additional GEO cohorts.

## Reproducibility notes

The tab-separated files are the exact source-data snapshots accompanying the submitted figures and statistical summaries. SHA-256 manifests within the analysis folders record the inputs and outputs used in each frozen analysis stage. Scripts require Python 3 and the scientific Python packages imported at the top of each file; some regeneration scripts also expect the accession-level raw matrices to be placed in the paths documented in the source code.

## Authorship

Xiaoning Guo and Lin Ling contributed equally as co-first authors. Zhengxiao Ouyang is the corresponding author.

## Citation

Please cite the associated manuscript. Bibliographic details will be added after publication.
