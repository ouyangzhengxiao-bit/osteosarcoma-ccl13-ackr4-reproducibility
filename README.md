# Osteosarcoma CCL13–ACKR4 reproducibility archive

This repository accompanies the manuscript **“Cross-cohort single-cell and spatial testing of a CCL13-ACKR4 communication hypothesis in osteosarcoma.”** It contains the source tables and analysis/figure-generation scripts used for the reported bulk, single-cell, and spatial sensitivity analyses.

## Contents

- `Figure3_scRNA_detection.tsv` and `Figure3_spatial_detection_and_correlation.tsv`: source tables for the cross-modal detection summary.
- `quality_upgrade_v3/`: patient-level single-cell and spatial robustness outputs.
- `quality_upgrade_v4/`: TARGET-OS expression, clinical-model, and repeated cross-validation outputs.
- `quality_upgrade_v5/`: independent GSE42352 context and chemotherapy-response analyses.
- `JBO_regeneration/`: consolidated tables used to regenerate the Journal of Bone Oncology manuscript results.
- `figure_generation/`: the manuscript figure-generation script.

## Public datasets

The analyses use de-identified public data from TARGET-OS and GEO accessions GSE21257, GSE39055, GSE32981, GSE42352, GSE152048, GSE162454, GSE270231, GSE293065, and GSE299025. Raw public matrices are not duplicated here; retrieve them from the Genomic Data Commons or GEO using the listed identifiers.

## Reproducibility notes

The tab-separated files are the exact source-data snapshots accompanying the submitted figures and statistical summaries. SHA-256 manifests within the analysis folders record the inputs and outputs used in each frozen analysis stage. Scripts require Python 3 and the scientific Python packages imported at the top of each file; some regeneration scripts also expect the accession-level raw matrices to be placed in the paths documented in the source code.

## Authorship

Xiaoning Guo and Lin Ling contributed equally as co-first authors. Zhengxiao Ouyang is the corresponding author.

## Citation

Please cite the associated manuscript. Bibliographic details will be added after publication.
