#!/usr/bin/env python3
"""Publication-style figures for the revised osteosarcoma reproducibility study."""
from pathlib import Path
import argparse, sys
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_boundary_modules import load_sample

NAVY, BLUE, TEAL = "#17324D", "#2F6B9A", "#1F9E89"
GOLD, CORAL, PURPLE = "#E9A23B", "#D95F59", "#7A6FAC"
GREY, GRID, WHITE = "#6B7280", "#D9E1E8", "#FFFFFF"
STATE_COLORS = {"T/NK":"#4C78A8", "B/plasma":"#9C6ADE", "Myeloid":"#E45756",
 "Osteoclast":"#F2A541", "Endothelial":"#54A24B", "Pericyte":"#72B7B2",
 "Fibroblast/MSC":"#B279A2", "Tumor/mesenchymal":"#8C6D5A", "Other":"#B9C0C7"}

def style():
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":11,
      "axes.titleweight":"bold","axes.labelcolor":NAVY,"axes.edgecolor":"#A8B3BD",
      "xtick.color":"#465563","ytick.color":"#465563","text.color":NAVY,
      "legend.frameon":False,"figure.facecolor":WHITE,"axes.facecolor":WHITE,
      "savefig.facecolor":WHITE,"pdf.fonttype":42,"ps.fonttype":42})

def label(ax, x): ax.text(-.08,1.06,x,transform=ax.transAxes,fontsize=15,fontweight="bold",va="top")
def clean(ax, axis=None):
    ax.spines[["top","right"]].set_visible(False)
    if axis: ax.grid(axis=axis,color=GRID,lw=.7,alpha=.75); ax.set_axisbelow(True)
def header(fig,title,sub):
    # Whole-figure titles belong in the manuscript legend, not in the artwork.
    return None
def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{stem}.png",dpi=400,bbox_inches="tight")
    fig.savefig(out/f"{stem}.pdf",bbox_inches="tight")
    fig.savefig(out/f"{stem}.tiff",dpi=400,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

def state_group(v):
    x=str(v).lower()
    if "osteoclast" in x:return "Osteoclast"
    if "myeloid" in x or "mast" in x:return "Myeloid"
    if "t_nk" in x or "t_cell" in x or "proliferating_t" in x:return "T/NK"
    if "b_cell" in x or "plasma" in x:return "B/plasma"
    if "endothelial" in x:return "Endothelial"
    if "pericyte" in x:return "Pericyte"
    if "fibroblast" in x:return "Fibroblast/MSC"
    if any(k in x for k in ["osteo","chondro","mesenchymal","myoblast"]):return "Tumor/mesenchymal"
    return "Other"
def manual(obs,path):
    m=pd.read_csv(path,sep="\t").set_index("leiden_r07")["manual_label"].to_dict()
    return obs.leiden_r07.astype(int).map(m).fillna("Other")

def workflow(ax):
    ax.axis("off"); boxes=[(.02,"Bulk discovery","TARGET-OS\n88 tumors",BLUE),(.27,"Patient-level scRNA","2 cohorts\n17 patients",TEAL),(.52,"Independent context","GSE270231\nlung metastases",PURPLE),(.77,"Spatial stress test","2 cohorts\n11 sections",GOLD)]
    for i,(x,t,s,c) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,.2),.19,.58,boxstyle="round,pad=.018,rounding_size=.025",transform=ax.transAxes,fc=WHITE,ec=c,lw=1.6))
        ax.text(x+.095,.62,t,transform=ax.transAxes,ha="center",fontweight="bold",color=c)
        ax.text(x+.095,.38,s,transform=ax.transAxes,ha="center",va="center",fontsize=8,color=GREY)
        if i<3: ax.add_patch(FancyArrowPatch((x+.195,.49),(boxes[i+1][0]-.01,.49),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=11,color="#9AA6B2"))
    ax.text(.5,.03,"Inference → detectability → reproducibility → spatial plausibility",transform=ax.transAxes,ha="center",fontweight="bold",fontsize=8.5)

def ci(beta,p):
    b=beta.to_numpy(float); p=np.clip(p.to_numpy(float),1e-300,.999999); se=np.abs(b)/norm.isf(p/2)
    return np.exp(b),np.exp(b-1.96*se),np.exp(b+1.96*se)
def forest(ax,names,est,lo,hi,p,title,xlabel,xlim):
    y=np.arange(len(names))[::-1]
    for yi,e,l,h,pv in zip(y,est,lo,hi,p):
        c=CORAL if pv<.05 else BLUE; ax.plot([l,h],[yi,yi],c=c,lw=1.7); ax.scatter(e,yi,s=34,c=c,edgecolor=WHITE,lw=.7,zorder=3)
    ax.axvline(1,c="#7A8793",lw=1,ls=(0,(4,3))); ax.set(xscale="log",xlim=xlim,yticks=y,yticklabels=names,xlabel=xlabel); ax.set_title(title,loc="left"); clean(ax,"x")
    for yi,e,pv in zip(y,est,p): ax.text(xlim[1]*1.03,yi,f"{e:.2f} | p={pv:.3f}",va="center",fontsize=7.3,color=CORAL if pv<.05 else GREY,clip_on=False)

def figure1(legacy,out):
    d=pd.read_csv(legacy/"results/validation/TARGET_OS_candidate_pair_validation.tsv",sep="\t")
    e=pd.read_csv(legacy/"results/integrated_evidence/candidate_evidence_matrix.tsv",sep="\t")
    order=d.sort_values("meta_p").pair_id.tolist(); d=d.set_index("pair_id").loc[order].reset_index(); names=[x.replace("--","–") for x in order]
    mo,ml,mh=ci(d.meta_beta,d.meta_p); so,sl,sh=ci(d.os_beta,d.os_p)
    fig=plt.figure(figsize=(14.2,10.2)); gs=fig.add_gridspec(3,2,height_ratios=[.75,2.45,2.15],hspace=.55,wspace=.72,left=.075,right=.93,top=.96,bottom=.08)
    a=fig.add_subplot(gs[0,:]);workflow(a);label(a,"A")
    a=fig.add_subplot(gs[1,0]);forest(a,names,mo,ml,mh,d.meta_p,"Metastasis at diagnosis","Odds ratio per 1-unit pair score",(.55,8.5));label(a,"B")
    a=fig.add_subplot(gs[1,1]);forest(a,names,so,sl,sh,d.os_p,"Overall survival","Hazard ratio per 1-unit pair score",(.22,4.2));label(a,"C")
    a=fig.add_subplot(gs[2,:]); cols=["GSE152048_Primary","GSE162454_Primary","GSE152048_Lung metastasis","GSE270231_Lung metastasis"]; cl=["GSE152048\nprimary","GSE162454\nprimary","GSE152048\nlung metastasis","GSE270231\nlung metastasis"]
    m=e.set_index("pair_id").reindex(order)[cols].astype(float).to_numpy(); im=a.imshow(np.ma.masked_invalid(m),aspect="auto",cmap=LinearSegmentedColormap.from_list("rep",["#F1F5F8","#93D4C8",TEAL]),vmin=0,vmax=1)
    a.set(yticks=np.arange(len(order)),yticklabels=names,xticks=np.arange(4),xticklabels=cl);a.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=6);a.set_title("Patient-level co-detection across single-cell cohorts",loc="left")
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
        t="NA" if np.isnan(m[i,j]) else f"{m[i,j]*100:.0f}%";a.text(j,i,t,ha="center",va="center",fontsize=8,color=WHITE if np.isfinite(m[i,j]) and m[i,j]>.58 else NAVY,fontweight="bold" if t!="NA" else "normal")
    for s in a.spines.values():s.set_visible(False)
    cb=fig.colorbar(im,ax=a,fraction=.02,pad=.025);cb.set_label("Patients with both genes detected");label(a,"D")
    header(fig,"Candidate communication signals require cross-modal replication","TARGET associations are discovery-cohort estimates; confidence intervals are reconstructed from two-sided p values.")
    fig.text(.075,.025,"Red: nominal p<0.05 (not multiplicity-adjusted). NA: unavailable/not evaluated. Same-cohort selection and testing are not confirmatory.",fontsize=7.8,color=GREY);save(fig,out,"Figure1_revised_multicohort_evidence")

def umap(ax,xy,lab,title,seed):
    g=lab.map(state_group);rng=np.random.default_rng(seed);take=rng.choice(len(xy),min(26000,len(xy)),False)
    for s,c in STATE_COLORS.items():
        k=g.iloc[take].to_numpy()==s
        if k.any():ax.scatter(xy[take][k,0],xy[take][k,1],s=1.1,c=c,alpha=.56,lw=0,rasterized=True)
    ax.set_title(title,loc="left");ax.set_xlabel("UMAP 1");ax.set_ylabel("UMAP 2");ax.set_xticks([]);ax.set_yticks([]);ax.spines[:].set_visible(False)
def figure2(out):
    x=ad.read_h5ad(ROOT/"work/data/processed/single_cell/osteosarcoma_reference_patient_balanced.h5ad",backed="r");k=x.obs.accession.astype(str).to_numpy()=="GSE152048";o=x.obs.loc[k].copy();l=manual(o,ROOT/"config/GSE152048_manual_cluster_map.tsv");xy=np.asarray(x.obsm["X_umap"])[k]
    z=ad.read_h5ad(ROOT/"work/data/processed/single_cell/GSE162454_patient_balanced_reference.h5ad",backed="r");oz=z.obs.copy();lz=oz.broad_state_manual.astype(str);xyz=np.asarray(z.obsm["X_umap"])
    fig=plt.figure(figsize=(14.5,11));gs=fig.add_gridspec(2,2,height_ratios=[1.06,1],wspace=.30,hspace=.34,left=.07,right=.96,top=.91,bottom=.10)
    a=fig.add_subplot(gs[0,0]);umap(a,xy,l,"GSE152048 · 11 patients",11);label(a,"A")
    a=fig.add_subplot(gs[0,1]);umap(a,xyz,lz,"GSE162454 · 6 patients",13);label(a,"B")
    hs=[Line2D([0],[0],marker="o",ls="",ms=6,mfc=c,mec="none",label=s) for s,c in STATE_COLORS.items() if s!="Other"];fig.legend(handles=hs,ncol=4,loc="upper center",bbox_to_anchor=(.52,.895),fontsize=8)
    a=fig.add_subplot(gs[1,0]);s=pd.read_csv(ROOT/"outputs/GSE152_signatures_validated_in_GSE162454.tsv",sep="\t",index_col=0);keep=["T_NK","Proliferating_T_NK","Myeloid","Inflammatory_myeloid","Osteoclast","Fibroblast_MSC","Endothelial","Osteoblastic_mesenchymal_provisional"];s=s.reindex([q for q in keep if q in s.index]);im=a.imshow(s,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-.8,vmax=1.6),aspect="auto");a.set(yticks=range(len(s)),yticklabels=[q.replace("_"," ") for q in s.index],xticks=range(len(s.columns)),xticklabels=[q.replace("_","/") for q in s.columns]);plt.setp(a.get_xticklabels(),rotation=35,ha="right");a.set_title("GSE152048-derived signatures in GSE162454",loc="left")
    for i in range(s.shape[0]):
      for j in range(s.shape[1]):
        v=s.iat[i,j];a.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=7,color=WHITE if abs(v)>.85 else NAVY)
    for sp in a.spines.values():sp.set_visible(False)
    fig.colorbar(im,ax=a,fraction=.035,pad=.025,label="Mean signature score");label(a,"C")
    a=fig.add_subplot(gs[1,1]);frames=[]
    for cohort,f in [("GSE152048","GSE152048_manual_composition.tsv"),("GSE162454","GSE162454_manual_composition.tsv")]:
        q=pd.read_csv(ROOT/"outputs"/f,sep="\t");q["group"]=q.broad_state_manual.map(state_group);q=q.groupby(["sample_id","group"],as_index=False).fraction_within_sample.sum();q["cohort"]=cohort;frames.append(q)
    w=pd.concat(frames).pivot_table(index=["cohort","sample_id"],columns="group",values="fraction_within_sample",fill_value=0);bot=np.zeros(len(w));xx=np.arange(len(w))
    for st in [q for q in STATE_COLORS if q in w]:v=w[st].to_numpy()*100;a.bar(xx,v,bottom=bot,width=.8,color=STATE_COLORS[st],lw=0);bot+=v
    a.set(xticks=xx,xticklabels=[i[1] for i in w.index],ylabel="Cells per patient (%)",ylim=(0,100));plt.setp(a.get_xticklabels(),rotation=60,ha="right",fontsize=7);split=sum(w.index.get_level_values(0)=="GSE152048")-.5;a.axvline(split,c=NAVY,lw=1);a.set_title("Patient-level cellular composition",loc="left");clean(a,"y");label(a,"D")
    header(fig,"Cell-state structure is reproducible at the patient level","Manual labels use canonical markers; cohorts are shown separately to avoid visual over-integration.")
    fig.text(.07,.035,"Tumor/mesenchymal labels remain provisional because copy-number or orthogonal pathology confirmation is incomplete.",fontsize=7.8,color=GREY);save(fig,out,"Figure2_revised_single_cell_reference")

def figure3(out):
    d=pd.read_csv(ROOT/"outputs/ica_signed_distance_spot_scores.tsv.gz",sep="\t");m=pd.read_csv(ROOT/"outputs/ica_signed_distance_metrics.tsv",sep="\t");sel=["PT2","PT3","PT5"];mods=["T_NK","Myeloid","Osteoclast","Fibroblast_MSC","Endothelial"];cs=dict(zip(mods,[BLUE,CORAL,GOLD,PURPLE,TEAL]))
    fig=plt.figure(figsize=(15,10.8));gs=fig.add_gridspec(2,3,height_ratios=[1.05,1],hspace=.28,wspace=.24,left=.055,right=.98,top=.96,bottom=.08)
    for j,s in enumerate(sel):
        a=fig.add_subplot(gs[0,j]);p=next((ROOT/"work/data/raw/spatial/GSE293065").glob(f"GSM*_{s}_filtered_feature_bc_matrix.h5"));_,_,_,pos,scale,img=load_sample(p);q=d[d["sample"]==s].copy(); valid=pos["barcode"].astype(str).isin(q["barcode"].astype(str)); pos=pos.loc[valid].copy(); xy=pos[["pxl_col_in_fullres","pxl_row_in_fullres"]].to_numpy()*scale; q=q.set_index("barcode").loc[pos["barcode"].astype(str)].reset_index()
        if len(q)!=len(xy):raise ValueError(f"Spot mismatch {s}")
        a.imshow(img);m0=q.candidate_tumor.to_numpy(bool);a.scatter(xy[~m0,0],xy[~m0,1],s=9,c="#5BA3C6",alpha=.63,lw=0,label="Stroma-side");a.scatter(xy[m0,0],xy[m0,1],s=9,c="#D85C5C",alpha=.63,lw=0,label="Tumor-side");b=np.abs(q.signed_distance_spots.to_numpy())<=1.2;a.scatter(xy[b,0],xy[b,1],s=28,facecolors="none",edgecolors=GOLD,lw=.75);a.set_title(f"{s} · candidate interface",loc="left");a.axis("off")
        if j==0:label(a,"A");a.legend(loc="lower left",fontsize=7,markerscale=1.4,frameon=True,facecolor=WHITE)
    a=fig.add_subplot(gs[1,:2]);edges=np.arange(-12,13,2)
    for s in sel:
      q=d[d["sample"]==s];bn=pd.cut(q.signed_distance_spots,edges,include_lowest=True);cen=np.array([x.mid for x in bn.cat.categories])
      for mod in mods:y=q.groupby(bn,observed=True)[mod].mean().reindex(bn.cat.categories).to_numpy();y=(y-np.nanmean(y))/(np.nanstd(y)+1e-9);a.plot(cen,y,c=cs[mod],lw=1.1,alpha=.25)
    q=d[d["sample"].isin(sel)];bn=pd.cut(q.signed_distance_spots,edges,include_lowest=True);cen=np.array([x.mid for x in bn.cat.categories])
    for mod in mods:y=q.groupby(bn,observed=True)[mod].mean().reindex(bn.cat.categories).to_numpy();y=(y-np.nanmean(y))/(np.nanstd(y)+1e-9);a.plot(cen,y,c=cs[mod],lw=2.5,label=mod.replace("_","/"))
    a.axvline(0,c=NAVY,lw=1,ls=(0,(4,3)));a.set(xlabel="Signed distance from candidate interface (spot units)",ylabel="Standardized module score");a.set_title("Expression gradients across candidate interfaces",loc="left");a.legend(ncol=3,fontsize=7.5);clean(a,"y");label(a,"B")
    a=fig.add_subplot(gs[1,2]);cols=[f"distance_rho_{x}" for x in mods];q=m.set_index("sample")[cols];q.columns=[x.replace("distance_rho_","").replace("_","/") for x in q.columns];im=a.imshow(q,cmap="RdBu_r",vmin=-.5,vmax=.5,aspect="auto");a.set(yticks=range(len(q)),yticklabels=q.index,xticks=range(len(q.columns)),xticklabels=q.columns);plt.setp(a.get_xticklabels(),rotation=45,ha="right");a.set_title("Direction varies across sections",loc="left")
    for i in range(q.shape[0]):
      for j in range(q.shape[1]):a.text(j,i,f"{q.iat[i,j]:+.2f}",ha="center",va="center",fontsize=7,color=WHITE if abs(q.iat[i,j])>.32 else NAVY)
    for sp in a.spines.values():sp.set_visible(False)
    fig.colorbar(im,ax=a,fraction=.055,pad=.03,label="Spearman ρ with distance");label(a,"C")
    header(fig,"Spatial gradients are heterogeneous across candidate tumor–stroma interfaces","Yellow rings mark interface-adjacent spots; ICA-derived boundaries remain exploratory pending pathology review.")
    fig.text(.055,.025,"Thin curves: PT2/PT3/PT5; thick curves: pooled visualization only. Formal inference must aggregate section effects at patient level.",fontsize=7.8,color=GREY);save(fig,out,"Figure3_revised_spatial_interface")

def gene_counts(a,g,k):
    ix=np.flatnonzero(np.char.upper(np.asarray(a.var_names).astype(str))==g);x=a.layers["counts"][k,ix[0]] if len(ix) else np.zeros(k.sum());return np.asarray(x.toarray() if sparse.issparse(x) else x).ravel()
def sc_audit(path,cohort,mapmode):
    a=ad.read_h5ad(path);keep=a.obs.accession.astype(str).to_numpy()==cohort if mapmode else np.ones(a.n_obs,bool);o=a.obs.loc[keep].copy();lab=manual(o,ROOT/"config/GSE152048_manual_cluster_map.tsv") if mapmode else o.broad_state_manual.astype(str);idx=np.flatnonzero(keep);rows=[]
    for patient in o.sample_id.astype(str).unique():
      for g,st in [("CCL13","Myeloid"),("ACKR4","Osteoclast")]:
        local=(o.sample_id.astype(str).to_numpy()==patient)&(lab.map(state_group).to_numpy()==st);k=np.zeros(a.n_obs,bool);k[idx[local]]=1;c=gene_counts(a,g,k);rows.append([cohort,patient,g,st,len(c),100*np.mean(c>0) if len(c) else np.nan])
    return rows
def spatial_audit():
    t=pd.read_csv(ROOT/"outputs/GSE152048_patient_robust_signatures.tsv",sep="\t");sig={s:t[t["state"]==s].gene.head(15).str.upper().tolist() for s in ["Myeloid","Osteoclast"]};rows=[];paths=sorted((ROOT/"work/data/raw/spatial/GSE293065").glob("*_filtered_feature_bc_matrix.h5"))+sorted((ROOT/"work/data/raw/spatial/GSE299025").glob("*_matrix.mtx.gz"))
    for p in paths:
      sm,x,g,_,_,_=load_sample(p);tot=np.asarray(x.sum(0)).ravel();tot[tot==0]=1;up=np.char.upper(g.astype(str));v={}
      for gene in ["CCL13","ACKR4"]:ix=np.flatnonzero(up==gene);c=np.asarray(x[ix[0],:].toarray()).ravel() if len(ix) else np.zeros(x.shape[1]);v[gene]=np.log1p(c*1e4/tot)
      mod={s:np.log1p(x[np.flatnonzero(np.isin(up,gs)),:].multiply(1e4/tot).toarray()).mean(0) for s,gs in sig.items()};short=sm.split("_")[-1];rows.append(["GSE299025" if "Patient" in short else "GSE293065",short,x.shape[1],100*np.mean(v["CCL13"]>0),100*np.mean(v["ACKR4"]>0),spearmanr(v["CCL13"],mod["Myeloid"]).statistic,spearmanr(v["ACKR4"],mod["Osteoclast"]).statistic])
    return pd.DataFrame(rows,columns=["cohort","section","n_spots","CCL13_pct","ACKR4_pct","CCL13_myeloid_rho","ACKR4_osteoclast_rho"])
def figure4(out):
    rows=sc_audit(ROOT/"work/data/processed/single_cell/osteosarcoma_reference_patient_balanced.h5ad","GSE152048",True)+sc_audit(ROOT/"work/data/processed/single_cell/GSE162454_patient_balanced_reference.h5ad","GSE162454",False);sc=pd.DataFrame(rows,columns=["cohort","patient","gene","cell_state","n_cells","pct_positive"]);sp=spatial_audit();src=out/"source_data";src.mkdir(parents=True,exist_ok=True);sc.to_csv(src/"Figure4_scRNA_detection.tsv",sep="\t",index=False);sp.to_csv(src/"Figure4_spatial_detection_and_correlation.tsv",sep="\t",index=False)
    fig=plt.figure(figsize=(14.2,9.8));gs=fig.add_gridspec(2,2,hspace=.40,wspace=.30,left=.08,right=.96,top=.96,bottom=.10)
    a=fig.add_subplot(gs[0,:]);p=sc.pivot_table(index=["cohort","patient"],columns="gene",values="pct_positive");x=np.arange(len(p));w=.36;a.bar(x-w/2,p.CCL13,w,color=CORAL,label="CCL13 in myeloid");a.bar(x+w/2,p.ACKR4,w,color=GOLD,label="ACKR4 in osteoclasts");a.set(xticks=x,xticklabels=[i[1] for i in p.index],ylabel="Positive cells within state (%)");plt.setp(a.get_xticklabels(),rotation=55,ha="right",fontsize=7.5);a.axvline(sum(p.index.get_level_values(0)=="GSE152048")-.5,c=NAVY,lw=1);a.set_title("Single-cell detection is patient-variable",loc="left");a.legend(ncol=2);clean(a,"y");label(a,"A")
    a=fig.add_subplot(gs[1,0]);x=np.arange(len(sp));w=.35;a.bar(x-w/2,sp.CCL13_pct,w,color=CORAL,label="CCL13");a.bar(x+w/2,sp.ACKR4_pct,w,color=GOLD,label="ACKR4");a.set_yscale("symlog",linthresh=.08);a.set(xticks=x,xticklabels=sp.section,ylabel="Positive spots (%) · symlog");plt.setp(a.get_xticklabels(),rotation=45,ha="right");a.axvline(sum(sp.cohort=="GSE293065")-.5,c=NAVY,lw=1);a.set_title("ACKR4 is rarely detected spatially",loc="left");a.legend();clean(a,"y");label(a,"B")
    a=fig.add_subplot(gs[1,1]);y=np.arange(len(sp));a.axvline(0,c=NAVY,lw=1,ls=(0,(4,3)));a.scatter(sp.CCL13_myeloid_rho,y-.14,c=CORAL,s=35,label="CCL13 vs myeloid module");a.scatter(sp.ACKR4_osteoclast_rho,y+.14,c=GOLD,s=35,label="ACKR4 vs osteoclast module");a.set(yticks=y,yticklabels=sp.section,xlim=(-.5,.5),xlabel="Within-section Spearman ρ");a.set_title("Spatial module correlations are weak",loc="left");a.legend(fontsize=7.5);clean(a,"x");label(a,"C")
    header(fig,"ACKR4 does not receive orthogonal spatial support","Detection prevalence and within-section correlations are reported directly; zero-heavy expression limits mechanistic inference.")
    fig.text(.08,.035,"Positive = raw count >0. Correlations use log1p(CP10K) expression and 15-gene state signatures. Descriptive sensitivity analyses only.",fontsize=7.8,color=GREY);save(fig,out,"Figure4_revised_ACKR4_robustness_audit")

def main():
    p=argparse.ArgumentParser();p.add_argument("--legacy-repo",type=Path,default=Path("/private/tmp/osteosarcoma-bulksignalR-review"));p.add_argument("--output",type=Path,default=ROOT/"outputs/manuscript_figures_v2");a=p.parse_args();style();figure1(a.legacy_repo,a.output);figure2(a.output);figure3(a.output);figure4(a.output);print(a.output)
if __name__=="__main__":main()
