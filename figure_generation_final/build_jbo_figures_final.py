#!/usr/bin/env python3
"""Build the final title-free JBO figure set from frozen source tables."""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "JBO_submission_final" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17324D"
BLUE = "#2F6B9A"
TEAL = "#1F9E89"
GOLD = "#E9A23B"
CORAL = "#D95F59"
PURPLE = "#7A6FAC"
GREY = "#6B7280"
GRID = "#D9E1E8"
WHITE = "#FFFFFF"


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": NAVY,
        "axes.edgecolor": "#A8B3BD",
        "xtick.color": "#465563",
        "ytick.color": "#465563",
        "text.color": NAVY,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def label(ax, letter, x=-0.11, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", color=NAVY)


def clean(ax, grid=None):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tif", dpi=400, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def figure1():
    q7 = ROOT / "outputs" / "quality_upgrade_v7"
    cur = pd.read_csv(q7 / "ACKR4_ligand_curation.tsv", sep="\t")
    models = pd.read_csv(q7 / "TARGET_OS_ACKR4_ligand_panel_models.tsv", sep="\t")
    models = models[models.model.eq("age_sex")].copy()
    sc = pd.read_csv(q7 / "primary_scRNA_ACKR4_ligand_detection_summary.tsv", sep="\t")
    sp = pd.read_csv(q7 / "primary_spatial_ACKR4_ligand_detection_summary.tsv", sep="\t")

    fig = plt.figure(figsize=(14.6, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.72, 1.75, 1.35],
                          width_ratios=[0.95, 1.45], hspace=0.42, wspace=0.32,
                          left=0.07, right=0.97, top=0.97, bottom=0.09)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); label(ax, "A", -0.02, 1.02)
    boxes = [
        (0.01, "Database screen", "TARGET-OS", BLUE),
        (0.22, "Ligand curation", "binding evidence", CORAL),
        (0.43, "Fixed-panel audit", "1 candidate + 5 ligands", TEAL),
        (0.66, "Patient-level scRNA", "13 primary tumors", "#4C9F50"),
        (0.86, "Spatial detection", "11 sections", GOLD),
    ]
    widths = [0.16, 0.16, 0.17, 0.17, 0.13]
    for i, ((x, head, sub, color), width) in enumerate(zip(boxes, widths)):
        ax.add_patch(FancyBboxPatch((x, 0.32), width, 0.45,
                                    boxstyle="round,pad=0.012,rounding_size=0.015",
                                    transform=ax.transAxes, fc=WHITE, ec=color, lw=1.8))
        ax.text(x + width/2, 0.61, head, transform=ax.transAxes,
                ha="center", va="center", fontweight="bold", color=color, fontsize=9.5)
        ax.text(x + width/2, 0.44, sub, transform=ax.transAxes,
                ha="center", va="center", color=GREY, fontsize=8.2)
        if i < len(boxes) - 1:
            nx = boxes[i+1][0]
            ax.add_patch(FancyArrowPatch((x + width + 0.006, 0.545), (nx - 0.008, 0.545),
                                         transform=ax.transAxes, arrowstyle="-|>",
                                         mutation_scale=11, color="#9AA6B2", lw=1.2))
    ax.text(0.5, 0.12, "Inference  →  biochemical plausibility  →  cohort-level evidence",
            transform=ax.transAxes, ha="center", fontweight="bold", fontsize=9)

    ax = fig.add_subplot(gs[1, 0]); label(ax, "B", -0.13, 1.02); ax.axis("off")
    evidence_colors = {"not established": CORAL, "established": "#4C9F50", "supported": TEAL}
    for i, row in enumerate(cur.itertuples(index=False)):
        y = 0.89 - i * 0.135
        ax.scatter([0.12], [y], s=95, color=evidence_colors[row.ACKR4_evidence], transform=ax.transAxes)
        ax.text(0.25, y, row.ligand, transform=ax.transAxes, va="center", fontsize=10.5)
        ax.text(0.48, y, row.ACKR4_evidence, transform=ax.transAxes, va="center", fontsize=9.4, color=GREY)
    ax.text(0.03, 0.04,
            "CCL13 was retained as the database-derived candidate,\nnot as an established ACKR4 ligand.",
            transform=ax.transAxes, fontsize=8.4, color=GREY)

    ax = fig.add_subplot(gs[1, 1]); label(ax, "C", -0.13, 1.02)
    order = ["CCL13", "CCL19", "CCL20", "CCL21", "CCL22", "CCL25"]
    q = models.set_index("ligand").loc[order].reset_index()
    y = np.arange(len(q))[::-1]
    for yy, row in zip(y, q.itertuples(index=False)):
        color = CORAL if row.ligand == "CCL13" else BLUE
        ax.plot([row.ci_low, row.ci_high], [yy, yy], color=color, lw=1.8)
        ax.scatter(row.odds_ratio_per_SD, yy, s=42, color=color, edgecolor=WHITE, zorder=3)
        ax.text(5.7, yy, f"OR {row.odds_ratio_per_SD:.2f} | q={row.BH_q_within_model:.3f}",
                va="center", fontsize=8, color=color if row.ligand == "CCL13" else GREY)
    ax.axvline(1, color="#7A8793", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.48, 9.4)
    ax.set_yticks(y, [f"{g}-ACKR4" for g in order])
    ax.set_xlabel("Metastasis odds ratio per 1 SD pair score")
    clean(ax, "x")

    ax = fig.add_subplot(gs[2, :]); label(ax, "D", -0.035, 1.05)
    genes = order + ["ACKR4"]
    scv = sc.set_index("gene").loc[genes, "patient_detection_fraction"].to_numpy() * 100
    spv = sp.set_index("gene").loc[genes, "section_detection_fraction"].to_numpy() * 100
    x = np.arange(len(genes)); w = 0.34
    ax.bar(x - w/2, scv, width=w, color=TEAL, label="Primary scRNA patients")
    ax.bar(x + w/2, spv, width=w, color=GOLD, label="Primary spatial sections")
    ax.set(xticks=x, xticklabels=genes, ylabel="Datasets with any detection (%)", ylim=(0, 108))
    ax.legend(ncol=2, loc="upper right")
    clean(ax, "y")
    fig.text(0.07, 0.025,
             "TARGET models adjust for age and sex; q values are Benjamini–Hochberg corrected across six fixed ligands. Any transcript detection does not establish binding or signaling.",
             fontsize=7.8, color=GREY)
    save(fig, "Figure_1")


def figure2():
    src = ROOT / "outputs" / "JBO_submission" / "Source_Data"
    det = pd.read_csv(src / "Figure3_scRNA_detection.tsv", sep="\t")
    spec = pd.read_csv(ROOT / "outputs" / "quality_upgrade_v3" / "patient_candidate_state_specificity.tsv", sep="\t")
    det["patient_label"] = det["cohort"].str.replace("GSE", "") + " · " + det["patient"].astype(str)
    wide = det.pivot_table(index="patient_label", columns="gene", values="pct_positive", aggfunc="first")
    wide = wide.sort_index()

    fig = plt.figure(figsize=(14.2, 10.2))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.27,
                          left=0.08, right=0.97, top=0.96, bottom=0.10)

    ax = fig.add_subplot(gs[0, 0]); label(ax, "A")
    m = np.column_stack([(wide.get("CCL13", pd.Series(index=wide.index, dtype=float)) > 0).astype(float),
                         (wide.get("ACKR4", pd.Series(index=wide.index, dtype=float)) > 0).astype(float)])
    m = np.column_stack([m, (m[:, 0] * m[:, 1])])
    ax.imshow(m, aspect="auto", cmap=mpl.colors.ListedColormap(["#EDF2F5", TEAL]), vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(wide)), wide.index, fontsize=7.2)
    ax.set_xticks([0, 1, 2], ["CCL13 in\nmyeloid", "ACKR4 in\nosteoclast", "Both\ndetected"])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, "yes" if m[i, j] else "no", ha="center", va="center",
                    fontsize=7, color=WHITE if m[i, j] else GREY)
    for s in ax.spines.values(): s.set_visible(False)

    ax = fig.add_subplot(gs[0, 1]); label(ax, "B")
    x = np.arange(len(wide))
    ax.scatter(x - 0.12, wide.get("CCL13"), s=34, color=CORAL, label="CCL13 in myeloid")
    ax.scatter(x + 0.12, wide.get("ACKR4"), s=34, color=GOLD, label="ACKR4 in osteoclast")
    ax.set_yscale("symlog", linthresh=0.08)
    ax.set(xticks=x, xticklabels=[i.split(" · ")[-1] for i in wide.index],
           ylabel="Positive cells within state (%)")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=7)
    split = sum(i.startswith("152048") for i in wide.index) - 0.5
    ax.axvline(split, color=NAVY, lw=0.9)
    ax.legend(ncol=2, fontsize=7.5)
    clean(ax, "y")

    primary = spec[(spec.context.eq("primary")) & (spec.metric.eq("pct_positive"))].copy()
    cohort_colors = {"GSE152048": BLUE, "GSE162454": TEAL}
    for pos, gene in enumerate(["CCL13", "ACKR4"]):
        ax = fig.add_subplot(gs[1, pos]); label(ax, "C" if pos == 0 else "D")
        q = primary[primary.gene.eq(gene)].copy().sort_values(["cohort", "patient"])
        xx = np.arange(len(q))
        colors = q.cohort.map(cohort_colors)
        ax.axhline(0, color="#7A8793", ls="--", lw=1)
        ax.vlines(xx, 0, q.target_minus_strongest_alternative, color=colors, lw=1.3, alpha=0.85)
        ax.scatter(xx, q.target_minus_strongest_alternative, c=colors, s=38, edgecolor=WHITE, zorder=3)
        ax.set(xticks=xx, xticklabels=q.patient, ylabel="Target minus strongest alternative\npositive-cell percentage points")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7.5)
        clean(ax, "y")
        p = 0.12548828125 if gene == "CCL13" else 0.9111328125
        n_above = int((q.target_minus_strongest_alternative > 0).sum())
        ax.text(0.02, 0.97, f"{n_above}/{len(q)} above all alternatives; exact P={p:.3f}",
                transform=ax.transAxes, va="top", fontsize=8, color=GREY)
        handles = [mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                                   markeredgecolor="none", label=k) for k, c in cohort_colors.items()]
        ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    fig.text(0.08, 0.03,
             "Primary-tumor analyses use patients as biological replicates. Positive values indicate that the prespecified state exceeded every eligible alternative lineage.",
             fontsize=7.8, color=GREY)
    save(fig, "Figure_2")


def figure3():
    src = ROOT / "outputs" / "JBO_submission" / "Source_Data"
    sp = pd.read_csv(src / "Figure3_spatial_detection_and_correlation.tsv", sep="\t")
    perm = pd.read_csv(ROOT / "outputs" / "quality_upgrade_v3" / "spatial_depth_matched_permutation.tsv", sep="\t")
    order = sp.section.tolist()

    fig = plt.figure(figsize=(14.2, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32,
                          left=0.08, right=0.97, top=0.96, bottom=0.11)

    ax = fig.add_subplot(gs[0, 0]); label(ax, "A")
    x = np.arange(len(sp)); w = 0.36
    ax.bar(x - w/2, sp.CCL13_pct, w, color=CORAL, label="CCL13")
    ax.bar(x + w/2, sp.ACKR4_pct, w, color=GOLD, label="ACKR4")
    ax.set_yscale("symlog", linthresh=0.08)
    ax.set(xticks=x, xticklabels=order, ylabel="Positive spots (%) · symlog")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(ncol=2)
    clean(ax, "y")

    ax = fig.add_subplot(gs[0, 1]); label(ax, "B")
    y = np.arange(len(sp))
    ax.axvline(0, color=NAVY, lw=1, ls="--")
    ax.scatter(sp.CCL13_myeloid_rho, y - 0.13, color=CORAL, s=38, label="CCL13 vs myeloid")
    ax.scatter(sp.ACKR4_osteoclast_rho, y + 0.13, color=GOLD, s=38, label="ACKR4 vs osteoclast")
    ax.set(yticks=y, yticklabels=order, xlim=(-0.5, 0.5), xlabel="Within-section Spearman ρ")
    ax.legend(fontsize=7.5)
    clean(ax, "x")

    ax = fig.add_subplot(gs[1, 0]); label(ax, "C")
    q = perm[perm.positive_spots.fillna(0).gt(0)].copy()
    q["section"] = pd.Categorical(q.section, categories=order, ordered=True)
    q = q.sort_values(["section", "gene"])
    jitter = {"CCL13": -0.10, "ACKR4": 0.10}
    for gene, color in [("CCL13", CORAL), ("ACKR4", GOLD)]:
        z = q[q.gene.eq(gene)]
        xx = np.array([order.index(str(v)) for v in z.section]) + jitter[gene]
        ax.scatter(xx, z.standardized_difference, s=42, color=color, edgecolor=WHITE, label=gene, zorder=3)
        for xi, yi, fdr in zip(xx, z.standardized_difference, z.bh_fdr_within_gene):
            if pd.notna(fdr) and fdr < 0.05:
                ax.text(xi, yi + 0.18, "*", ha="center", color=color, fontweight="bold")
    ax.axhline(0, color="#7A8793", ls="--", lw=1)
    ax.set(xticks=np.arange(len(order)), xticklabels=order,
           ylabel="Depth-matched standardized\nsignature difference")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(ncol=2)
    clean(ax, "y")

    ax = fig.add_subplot(gs[1, 1]); label(ax, "D")
    piv = perm.pivot_table(index="section", columns="gene", values="positive_spots", aggfunc="first").reindex(order)
    piv = piv.reindex(columns=["ACKR4", "CCL13"])
    im = ax.imshow(piv.fillna(0), cmap="Blues", aspect="auto")
    ax.set_yticks(np.arange(len(piv)), piv.index)
    ax.set_xticks([0, 1], ["ACKR4", "CCL13"])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(v) else f"{int(v)}", ha="center", va="center",
                    fontsize=8, color=WHITE if pd.notna(v) and v > np.nanmax(piv.to_numpy()) * 0.45 else NAVY)
    for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Positive spots")
    fig.text(0.08, 0.035,
             "Positive means raw count >0. Depth-matched permutations control library size but not spatial autocorrelation; sparse ACKR4-positive sets limit inference.",
             fontsize=7.8, color=GREY)
    save(fig, "Figure_3")


def copy_existing():
    old = ROOT / "outputs" / "JBO_submission" / "Figures"
    mapping = {
        "Supplementary_Figure_S3": "Figure_4",
        "Figure_2": "Supplementary_Figure_S1",
        "Supplementary_Figure_S2": "Supplementary_Figure_S2",
        "Supplementary_Figure_S4": "Supplementary_Figure_S3",
    }
    for src_stem, dst_stem in mapping.items():
        for ext in ("png", "tif"):
            shutil.copy2(old / f"{src_stem}.{ext}", OUT / f"{dst_stem}.{ext}")


def main():
    style()
    figure1()
    figure2()
    figure3()
    copy_existing()
    print(OUT)


if __name__ == "__main__":
    main()
