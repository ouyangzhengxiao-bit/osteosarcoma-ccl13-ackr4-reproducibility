#!/usr/bin/env python3
"""Patient-level state specificity and depth-matched spatial permutation audit.

This analysis treats patients or tissue sections as the replication unit. It
does not infer ligand-receptor binding. The spatial test asks only whether
spots with a nonzero candidate-gene count have higher scores for the expected
cell-state program than library-size-matched negative spots.
"""

from itertools import product
from pathlib import Path
import sys

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "quality_upgrade_v3"
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_boundary_modules import load_sample


NAVY = "#17324D"
BLUE = "#2F6B9A"
TEAL = "#1F9E89"
GOLD = "#E9A23B"
CORAL = "#D95F59"
GREY = "#6B7280"
GRID = "#D9E1E8"


def state_group(value):
    value = str(value).lower()
    if "osteoclast" in value:
        return "Osteoclast"
    if "myeloid" in value or "mast" in value:
        return "Myeloid"
    if "t_nk" in value or "t_cell" in value or "proliferating_t" in value:
        return "T_NK"
    if "b_cell" in value or "plasma" in value:
        return "B_plasma"
    if "endothelial" in value:
        return "Endothelial"
    if "pericyte" in value:
        return "Pericyte"
    if "fibroblast" in value:
        return "Fibroblast_MSC"
    if any(x in value for x in ["osteo", "chondro", "mesenchymal", "myoblast"]):
        return "Tumor_mesenchymal"
    return "Other"


def gse152_labels(obs):
    mapping = pd.read_csv(ROOT / "config/GSE152048_manual_cluster_map.tsv", sep="\t")
    mapping = mapping.set_index("leiden_r07")["manual_label"].to_dict()
    return obs["leiden_r07"].astype(int).map(mapping).fillna("Other").map(state_group)


def gene_vector(adata, gene):
    names = np.char.upper(np.asarray(adata.var_names).astype(str))
    hit = np.flatnonzero(names == gene.upper())
    if len(hit) != 1:
        raise ValueError(f"Expected one feature for {gene}, found {len(hit)}")
    x = adata.layers["counts"][:, hit[0]]
    return np.asarray(x.toarray() if sparse.issparse(x) else x).ravel()


def cell_state_table(path, cohort, label_mode):
    adata = ad.read_h5ad(path)
    if "accession" in adata.obs:
        keep = adata.obs["accession"].astype(str).to_numpy() == cohort
        adata = adata[keep].copy()
    labels = gse152_labels(adata.obs) if label_mode == "gse152" else adata.obs["broad_state_manual"].map(state_group)
    totals = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    rows = []
    for gene in ["CCL13", "ACKR4"]:
        counts = gene_vector(adata, gene)
        log_cp10k = np.log1p(counts * 1e4 / np.maximum(totals, 1))
        for patient in sorted(adata.obs["sample_id"].astype(str).unique()):
            patient_mask = adata.obs["sample_id"].astype(str).to_numpy() == patient
            contexts = adata.obs.loc[patient_mask, "lesion_type"].astype(str).unique()
            context = contexts[0] if len(contexts) == 1 else ";".join(sorted(contexts))
            for state in sorted(pd.unique(labels)):
                mask = patient_mask & (labels.to_numpy() == state)
                if not mask.any():
                    continue
                rows.append({
                    "cohort": cohort,
                    "patient": patient,
                    "context": context,
                    "gene": gene,
                    "state": state,
                    "n_cells": int(mask.sum()),
                    "positive_cells": int(np.sum(counts[mask] > 0)),
                    "pct_positive": float(100 * np.mean(counts[mask] > 0)),
                    "mean_log1p_cp10k": float(np.mean(log_cp10k[mask])),
                })
    return pd.DataFrame(rows)


def exact_signflip_p(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan
    observed = abs(values.mean())
    if n <= 20:
        null = np.array([np.mean(values * np.asarray(s)) for s in product([-1, 1], repeat=n)])
    else:
        rng = np.random.default_rng(73021)
        null = np.mean(values * rng.choice([-1, 1], size=(200000, n)), axis=1)
    return float(np.mean(np.abs(null) >= observed - 1e-15))



def full_cell_state_table():
    """Load candidate summaries calculated from all QC-passing singlet cells."""
    legacy = ROOT / "work/data/processed/single_cell/fullcell_candidate_summaries"
    specs = [
        ("GSE152048", legacy / "GSE152048_candidate_expression_by_patient_lineage.tsv"),
        ("GSE162454", legacy / "GSE162454_candidate_expression_by_patient_lineage.tsv"),
        ("GSE270231", legacy / "GSE270231_candidate_expression_by_patient_lineage.tsv"),
    ]
    context = {}
    for config in ["single_cell_samples_GSE152048.tsv", "single_cell_samples_GSE162454.tsv"]:
        meta = pd.read_csv(ROOT / "config" / config, sep="\t")
        context.update(meta.set_index("patient_id")["lesion_type"].astype(str).to_dict())
    frames = []
    for cohort, path in specs:
        table = pd.read_csv(path, sep="\t")
        lineage_col = "published_lineage" if "published_lineage" in table else "lineage"
        mean_col = "mean_log_normalized_expression" if "mean_log_normalized_expression" in table else "mean_expression"
        table = table[table["gene"].isin(["CCL13", "ACKR4"])].copy()
        table = table.rename(columns={"patient_id": "patient", lineage_col: "state",
                                      "cells": "n_cells", "percent_expressing": "pct_positive",
                                      mean_col: "mean_log1p_cp10k"})
        table["cohort"] = cohort
        table["context"] = table["patient"].map(context).fillna("lung_metastasis")
        table["state"] = table["state"].map(state_group)
        table["positive_cells"] = np.rint(table["n_cells"] * table["pct_positive"] / 100).astype(int)
        frames.append(table[["cohort", "patient", "context", "gene", "state", "n_cells",
                             "positive_cells", "pct_positive", "mean_log1p_cp10k"]])
    return pd.concat(frames, ignore_index=True)

def specificity_tables(cell):
    targets = {"CCL13": "Myeloid", "ACKR4": "Osteoclast"}
    patient_rows = []
    for (cohort, context, patient, gene), group in cell.groupby(["cohort", "context", "patient", "gene"]):
        target = targets[gene]
        eligible = group[group["n_cells"] >= 20].copy()
        target_row = eligible[eligible["state"] == target]
        alternatives = eligible[eligible["state"] != target]
        if target_row.empty or alternatives.empty:
            continue
        target_row = target_row.iloc[0]
        for metric in ["pct_positive", "mean_log1p_cp10k"]:
            strongest = alternatives.loc[alternatives[metric].idxmax()]
            patient_rows.append({
                "cohort": cohort,
                "patient": patient,
                "context": context,
                "gene": gene,
                "target_state": target,
                "metric": metric,
                "target_value": float(target_row[metric]),
                "strongest_alternative_state": strongest["state"],
                "strongest_alternative_value": float(strongest[metric]),
                "target_minus_strongest_alternative": float(target_row[metric] - strongest[metric]),
                "target_n_cells": int(target_row["n_cells"]),
            })
    paired = pd.DataFrame(patient_rows)
    summaries = []
    for (cohort, context, gene, metric), group in paired.groupby(["cohort", "context", "gene", "metric"]):
        delta = group["target_minus_strongest_alternative"].to_numpy()
        summaries.append({
            "cohort": cohort,
            "context": context,
            "gene": gene,
            "target_state": group["target_state"].iloc[0],
            "metric": metric,
            "patients": len(group),
            "patients_target_above_all_alternatives": int(np.sum(delta > 0)),
            "median_delta": float(np.median(delta)),
            "mean_delta": float(np.mean(delta)),
            "exact_signflip_p": exact_signflip_p(delta),
        })
    primary = paired[paired["context"] == "primary"]
    for (gene, metric), group in primary.groupby(["gene", "metric"]):
        delta = group["target_minus_strongest_alternative"].to_numpy()
        summaries.append({
            "cohort": "Combined_primary",
            "context": "primary",
            "gene": gene,
            "target_state": group["target_state"].iloc[0],
            "metric": metric,
            "patients": len(group),
            "patients_target_above_all_alternatives": int(np.sum(delta > 0)),
            "median_delta": float(np.median(delta)),
            "mean_delta": float(np.mean(delta)),
            "exact_signflip_p": exact_signflip_p(delta),
        })
    return paired, pd.DataFrame(summaries)


def signature_genes():
    table = pd.read_csv(ROOT / "outputs/GSE152048_patient_robust_signatures.tsv", sep="\t")
    return {
        state: table.loc[table["state"] == state, "gene"].head(15).str.upper().tolist()
        for state in ["Myeloid", "Osteoclast"]
    }


def depth_bins(totals):
    ranks = pd.Series(totals).rank(method="first")
    return pd.qcut(ranks, q=min(10, len(ranks)), labels=False, duplicates="drop").to_numpy()


def matched_null(module, positive, bins, n_perm, seed):
    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(positive)
    negative_idx = np.flatnonzero(~positive)
    draws = np.empty(n_perm)
    for b in range(n_perm):
        chosen = []
        for depth_bin in np.unique(bins[positive_idx]):
            required = int(np.sum(bins[positive_idx] == depth_bin))
            candidates = negative_idx[bins[negative_idx] == depth_bin]
            if len(candidates) == 0:
                candidates = negative_idx
            chosen.extend(rng.choice(candidates, size=required, replace=len(candidates) < required))
        draws[b] = np.mean(module[np.asarray(chosen, int)])
    return draws


def spatial_permutation(n_perm=20000):
    signatures = signature_genes()
    paths = sorted((ROOT / "work/data/raw/spatial/GSE293065").glob("*_filtered_feature_bc_matrix.h5"))
    paths += sorted((ROOT / "work/data/raw/spatial/GSE299025").glob("*_matrix.mtx.gz"))
    rows = []
    for path_index, path in enumerate(paths):
        sample, matrix, genes, _, _, _ = load_sample(path)
        gene_upper = np.char.upper(np.asarray(genes).astype(str))
        totals = np.asarray(matrix.sum(axis=0)).ravel()
        bins = depth_bins(totals)
        section = sample.split("_")[-1]
        cohort = "GSE299025" if "Patient" in section else "GSE293065"
        for gene, state in [("CCL13", "Myeloid"), ("ACKR4", "Osteoclast")]:
            gene_hit = np.flatnonzero(gene_upper == gene)
            counts = np.asarray(matrix[gene_hit[0], :].toarray()).ravel() if len(gene_hit) else np.zeros(matrix.shape[1])
            positive = counts > 0
            signature_hit = np.flatnonzero(np.isin(gene_upper, signatures[state]))
            signature_counts = np.asarray(matrix[signature_hit, :].sum(axis=0)).ravel()
            module = np.log1p(signature_counts * 1e4 / np.maximum(totals, 1) / max(len(signature_hit), 1))
            base = {
                "cohort": cohort,
                "section": section,
                "gene": gene,
                "expected_state": state,
                "n_spots": matrix.shape[1],
                "positive_spots": int(positive.sum()),
                "pct_positive": float(100 * positive.mean()),
                "signature_genes_detected": int(len(signature_hit)),
                "positive_spot_mean_signature": float(np.mean(module[positive])) if positive.any() else np.nan,
            }
            if not positive.any() or positive.all():
                base.update({"matched_null_mean": np.nan, "depth_matched_difference": np.nan,
                             "standardized_difference": np.nan, "empirical_two_sided_p": np.nan})
            else:
                null = matched_null(module, positive, bins, n_perm, 991 + path_index * 11 + (0 if gene == "CCL13" else 1))
                observed = np.mean(module[positive])
                center = np.mean(null)
                sd = np.std(null, ddof=1)
                p = (1 + np.sum(np.abs(null - center) >= abs(observed - center))) / (n_perm + 1)
                base.update({
                    "matched_null_mean": float(center),
                    "depth_matched_difference": float(observed - center),
                    "standardized_difference": float((observed - center) / sd) if sd > 0 else np.nan,
                    "empirical_two_sided_p": float(p),
                })
            rows.append(base)
    result = pd.DataFrame(rows)
    result["bh_fdr_within_gene"] = np.nan
    for gene, idx in result.groupby("gene").groups.items():
        valid = result.loc[idx, "empirical_two_sided_p"].dropna().sort_values()
        if valid.empty:
            continue
        ranks = np.arange(1, len(valid) + 1)
        adjusted = np.minimum.accumulate((valid.to_numpy() * len(valid) / ranks)[::-1])[::-1]
        result.loc[valid.index, "bh_fdr_within_gene"] = np.minimum(adjusted, 1)
    return result


def make_figure(paired, spatial):
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.titleweight": "bold", "text.color": NAVY, "axes.labelcolor": NAVY,
        "axes.edgecolor": "#A8B3BD", "xtick.color": "#465563", "ytick.color": "#465563",
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), gridspec_kw={"hspace": .42, "wspace": .30})
    colors = {"GSE152048": BLUE, "GSE162454": TEAL, "GSE270231": "#7A6FAC"}
    for col, gene in enumerate(["CCL13", "ACKR4"]):
        ax = axes[0, col]
        q = paired[(paired["gene"] == gene) & (paired["metric"] == "pct_positive") &
                   (paired["context"] == "primary")].copy()
        q = q.sort_values(["cohort", "context", "patient"])
        x = np.arange(len(q))
        for cohort, g in q.groupby("cohort"):
            ix = q.index.get_indexer(g.index)
            ax.scatter(ix, g["target_minus_strongest_alternative"], s=38, color=colors[cohort],
                       edgecolor="white", linewidth=.6, label=cohort, zorder=3)
        ax.axhline(0, color="#7A8793", lw=1, ls=(0, (4, 3)))
        ax.set_xticks(x, q["patient"], rotation=50, ha="right", fontsize=7.5)
        ax.set_ylabel("Target minus strongest alternative\npositive cells percentage points")
        ax.set_title(f"{gene} specificity in primary tumors", loc="left")
        ax.grid(axis="y", color=GRID, lw=.7, alpha=.8)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.legend(frameon=False, fontsize=8)
        ax.text(-.10, 1.07, chr(ord("A") + col), transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax = axes[1, 0]
    q = spatial[spatial["positive_spots"] > 0].copy()
    for gene, color, marker in [("CCL13", CORAL, "o"), ("ACKR4", GOLD, "s")]:
        z = q[q["gene"] == gene]
        y = np.arange(len(z)) + (-.13 if gene == "CCL13" else .13)
        ax.scatter(z["standardized_difference"], y, s=42, color=color, marker=marker,
                   edgecolor="white", linewidth=.6, label=gene, zorder=3)
    sections = list(dict.fromkeys(q.sort_values(["section", "gene"])["section"]))
    # Rebuild a common y axis to keep the two genes aligned by section.
    ax.clear()
    ymap = {s: i for i, s in enumerate(sections)}
    for gene, color, marker, offset in [("CCL13", CORAL, "o", -.12), ("ACKR4", GOLD, "s", .12)]:
        z = q[q["gene"] == gene]
        ax.scatter(z["standardized_difference"], [ymap[s] + offset for s in z["section"]],
                   s=42, color=color, marker=marker, edgecolor="white", linewidth=.6,
                   label=gene, zorder=3)
    ax.axvline(0, color="#7A8793", lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(sections)), sections)
    ax.set_xlabel("Depth-matched standardized signature difference")
    ax.set_title("Sparse spatial positives lack consistent enrichment", loc="left")
    ax.grid(axis="x", color=GRID, lw=.7, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.text(-.10, 1.07, "C", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    ax = axes[1, 1]
    z = spatial.pivot(index="section", columns="gene", values="positive_spots").fillna(0)
    z = z.reindex(sorted(z.index))
    image = ax.imshow(np.log10(z.to_numpy() + 1), cmap="Blues", vmin=0,
                      vmax=max(1, np.log10(z.to_numpy().max() + 1)), aspect="auto")
    ax.set_xticks(range(len(z.columns)), z.columns)
    ax.set_yticks(range(len(z.index)), z.index)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, str(int(z.iat[i, j])), ha="center", va="center",
                    color="white" if z.iat[i, j] >= max(2, z.to_numpy().max() / 2) else NAVY,
                    fontsize=8, fontweight="bold")
    ax.set_title("Candidate positive spots per section", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-.10, 1.07, "D", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    fig.subplots_adjust(top=.95)
    fig.text(.06, .025,
             "Spatial null distributions match gene-negative spots by library-size decile within each section; 20,000 permutations. Positive means higher expected-state signature.",
             ha="left", fontsize=7.8, color=GREY)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 400}), ("pdf", {}), ("tiff", {"dpi": 400, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(OUT / f"Supplementary_Figure_S1_candidate_specificity.{ext}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cell = full_cell_state_table()
    paired, summary = specificity_tables(cell)
    spatial = spatial_permutation()
    cell.to_csv(OUT / "patient_cell_state_candidate_expression.tsv", sep="\t", index=False)
    paired.to_csv(OUT / "patient_candidate_state_specificity.tsv", sep="\t", index=False)
    summary.to_csv(OUT / "candidate_state_specificity_summary.tsv", sep="\t", index=False)
    spatial.to_csv(OUT / "spatial_depth_matched_permutation.tsv", sep="\t", index=False)
    make_figure(paired, spatial)
    print(summary.to_string(index=False))
    print("\nSpatial sections with candidate detection")
    print(spatial.loc[spatial.positive_spots > 0, ["cohort", "section", "gene", "positive_spots",
          "depth_matched_difference", "standardized_difference", "empirical_two_sided_p",
          "bh_fdr_within_gene"]].to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
