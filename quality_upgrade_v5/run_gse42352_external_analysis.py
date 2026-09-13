#!/usr/bin/env python3
"""Independent GSE42352 response/context analysis; no survival endpoint is available."""
from pathlib import Path
import argparse, hashlib, io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
from statsmodels.miscmodels.ordinal_model import OrderedModel

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'work/data/raw/survival'; OUT=ROOT/'outputs/quality_upgrade_v5'
MODULES=['T_NK','Myeloid','Osteoclast','Fibroblast_MSC','Endothelial']; CANDIDATES=['CCL13','ACKR4']
def z(x):
 x=pd.to_numeric(x,errors='coerce'); return (x-x.mean())/x.std(ddof=1)
def platform(path):
 lines=path.read_text(errors='replace').splitlines(True); a=next(i for i,x in enumerate(lines) if x.startswith('!platform_table_begin'))+1; b=next(i for i,x in enumerate(lines) if x.startswith('!platform_table_end'))
 q=pd.read_csv(io.StringIO(''.join(lines[a:b])),sep='\t',dtype=str,low_memory=False)[['ID','Symbol']].dropna(); q['gene']=q.Symbol.str.strip().str.upper().replace({'CCRL1':'ACKR4'}); return q[['ID','gene']].drop_duplicates()
def clinical(path):
 q=pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False).drop_duplicates('Assay Name').rename(columns={'Assay Name':'geo_accession','Comment [Sample_source_name]':'source','Characteristics [age]':'age','Characteristics [sex]':'sex','Characteristics [huvos grade]':'huvos_grade','Characteristics [histological subtype]':'histology','Characteristics [tumor location]':'tumor_location'})
 q['age_years']=pd.to_numeric(q.age.str.extract(r'([0-9.]+)')[0],errors='coerce')/12; q['sex_male']=q.sex.str.lower().map({'male':1,'female':0}); q['huvos_grade']=pd.to_numeric(q.huvos_grade,errors='coerce'); q['good_response']=np.where(q.huvos_grade.notna(),(q.huvos_grade>=3).astype(int),np.nan); q['sample_class']=q.source.map({'High-grade osteosarcoma pre-chemotherapy biopsy':'Osteosarcoma biopsy','Mesenchymal stem cell':'MSC','Osteoblast':'Osteoblast'})
 return q[['geo_accession','sample_class','age_years','sex','sex_male','huvos_grade','good_response','histology','tumor_location']]
def build(sample_dir,out):
 sig=pd.read_csv(ROOT/'outputs/GSE152048_patient_robust_signatures.tsv',sep='\t'); sig=sig[sig.state.isin(MODULES)]; wanted=set(sig.gene.str.upper())|set(CANDIDATES); mp=platform(RAW/'GPL10295_platform_data.txt'); mp=mp[mp.gene.isin(wanted)]; lookup=dict(zip(mp.ID,mp.gene)); rec=[]
 for f in sorted(sample_dir.glob('GSM*_sample_table.txt')):
  v=pd.read_csv(f,sep='\t',dtype={'Reporter Identifier':str}); v=v[v['Reporter Identifier'].isin(lookup)].copy(); v['gene']=v['Reporter Identifier'].map(lookup); v['VALUE']=pd.to_numeric(v.VALUE,errors='coerce'); x=v.groupby('gene').VALUE.median().to_dict(); x['geo_accession']=f.name.split('_',1)[0]; rec.append(x)
 if len(rec)!=99: raise RuntimeError(f'Expected 99 processed tables, found {len(rec)}')
 e=pd.DataFrame(rec).set_index('geo_accession').sort_index(); out.parent.mkdir(parents=True,exist_ok=True); e.to_csv(out,sep='\t'); return e
def scores(e,c):
 sig=pd.read_csv(ROOT/'outputs/GSE152048_patient_robust_signatures.tsv',sep='\t'); sig=sig[sig.state.isin(MODULES)].copy(); sig['gene']=sig.gene.str.upper(); d=c.set_index('geo_accession').join(e,how='inner'); ix=d.sample_class.eq('Osteosarcoma biopsy')
 for g in CANDIDATES: d.loc[ix,g+'_z']=z(d.loc[ix,g])
 d.loc[ix,'pair_mean_z']=z((d.loc[ix,'CCL13_z']+d.loc[ix,'ACKR4_z'])/2); cov=[]
 for m in MODULES:
  genes=sig.loc[sig.state.eq(m),'gene'].drop_duplicates().tolist(); found=[g for g in genes if g in e.columns]; d.loc[ix,m+'_z']=d.loc[ix,found].apply(z).mean(axis=1,skipna=True); cov.append({'module':m,'signature_genes':len(genes),'mapped_genes':len(found),'coverage_fraction':len(found)/len(genes),'mapped_gene_list':';'.join(found)})
 return d.reset_index(),pd.DataFrame(cov)
def fit(d,name,formula,term):
 f=smf.logit(formula,d).fit(disp=0,maxiter=500); ci=f.conf_int().loc[term]; return {'model':name,'term':term,'n':int(f.nobs),'good_responders':int(f.model.endog.sum()),'odds_ratio':np.exp(f.params[term]),'ci_low':np.exp(ci.iloc[0]),'ci_high':np.exp(ci.iloc[1]),'p':f.pvalues[term],'aic':f.aic}
def analyse(d):
 b=d[d.sample_class.eq('Osteosarcoma biopsy')].copy(); specs=[('Pair, unadjusted','good_response ~ pair_mean_z','pair_mean_z'),('Pair + age/sex','good_response ~ pair_mean_z + age_years + sex_male','pair_mean_z'),('Pair + age/sex + lineages','good_response ~ pair_mean_z + age_years + sex_male + Myeloid_z + Osteoclast_z','pair_mean_z'),('CCL13 + age/sex','good_response ~ CCL13_z + age_years + sex_male','CCL13_z'),('ACKR4 + age/sex','good_response ~ ACKR4_z + age_years + sex_male','ACKR4_z'),('Interaction + age/sex','good_response ~ CCL13_z * ACKR4_z + age_years + sex_male','CCL13_z:ACKR4_z')]; models=pd.DataFrame([fit(b,*s) for s in specs]); q=b.dropna(subset=['huvos_grade','pair_mean_z','age_years','sex_male']); o=OrderedModel(q.huvos_grade.astype(int),q[['pair_mean_z','age_years','sex_male']],distr='logit').fit(method='bfgs',disp=False); ci=o.conf_int().loc['pair_mean_z']; ordinal=pd.DataFrame([{'model':'Ordinal Huvos grade + age/sex','term':'pair_mean_z','n':len(q),'proportional_odds_ratio':np.exp(o.params.pair_mean_z),'ci_low':np.exp(ci.iloc[0]),'ci_high':np.exp(ci.iloc[1]),'p':o.pvalues.pair_mean_z}]); cols=['CCL13_z','ACKR4_z']+[m+'_z' for m in MODULES]; cor=b[cols].corr(method='spearman'); ctx=[]
 for g in CANDIDATES:
  t=d.loc[d.sample_class.eq('Osteosarcoma biopsy'),g].dropna(); r=d.loc[d.sample_class.isin(['MSC','Osteoblast']),g].dropna(); u,p=mannwhitneyu(t,r); ctx.append({'gene':g,'tumor_n':len(t),'reference_n':len(r),'tumor_median':t.median(),'reference_median':r.median(),'median_difference':t.median()-r.median(),'mann_whitney_u':u,'p':p})
 return models,ordinal,cor,pd.DataFrame(ctx)
def figure(d,models,cor):
 mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.titleweight':'bold','text.color':'#17324D','axes.labelcolor':'#17324D','xtick.color':'#465563','ytick.color':'#465563','axes.edgecolor':'#A8B3BD','figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42}); fig,ax=plt.subplots(1,3,figsize=(15.3,5.3),gridspec_kw={'wspace':.42}); pal={'Osteosarcoma biopsy':'#2F6B9A','MSC':'#E9A23B','Osteoblast':'#93D4C8'}; long=d.melt(id_vars=['sample_class'],value_vars=CANDIDATES,var_name='Gene',value_name='Expression'); sns.boxplot(data=long,x='Gene',y='Expression',hue='sample_class',palette=pal,showfliers=False,width=.72,linewidth=1,ax=ax[0]); sns.stripplot(data=long,x='Gene',y='Expression',hue='sample_class',palette=pal,dodge=True,alpha=.38,size=2.2,linewidth=0,ax=ax[0]); h,l=ax[0].get_legend_handles_labels(); ax[0].legend(h[:3],l[:3],frameon=False,fontsize=8); ax[0].set(ylabel='Normalized microarray expression',xlabel=''); ax[0].set_title('Tumor-versus-reference context',loc='left')
 order=['Pair, unadjusted','Pair + age/sex','Pair + age/sex + lineages','CCL13 + age/sex','ACKR4 + age/sex','Interaction + age/sex']; q=models.set_index('model').loc[order].reset_index(); y=np.arange(len(q))[::-1]
 for yy,row in zip(y,q.itertuples()): ax[1].plot([row.ci_low,row.ci_high],[yy,yy],color='#2F6B9A',lw=1.8); ax[1].scatter(row.odds_ratio,yy,s=42,color='#D95F59' if row.p<.05 else '#2F6B9A',edgecolor='white',zorder=3)
 ax[1].axvline(1,color='#7A8793',ls='--'); ax[1].set_xscale('log'); ax[1].set_xlim(.25,3); ax[1].set_xticks([.25,.5,1,2,3],['0.25','0.5','1','2','3'],minor=False); ax[1].set_xticks([],minor=True); ax[1].set_yticks(y,[s.replace(' + ','\n+ ') for s in order]); ax[1].set_xlabel('Odds ratio for good response per 1 SD'); ax[1].set_title('Independent Huvos-response analysis',loc='left'); ax[1].grid(axis='x',color='#D9E1E8')
 cols=['CCL13_z','ACKR4_z','Myeloid_z','Osteoclast_z','Fibroblast_MSC_z','Endothelial_z','T_NK_z']; cm=cor.loc[cols,cols]; im=ax[2].imshow(cm,cmap='RdBu_r',vmin=-1,vmax=1); labels=[x.replace('_z','').replace('_','/') for x in cols]; ax[2].set_xticks(range(len(labels)),labels,rotation=45,ha='right'); ax[2].set_yticks(range(len(labels)),labels)
 for i in range(len(labels)):
  for j in range(len(labels)):
   v=cm.iat[i,j]; ax[2].text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(v)>.55 else '#17324D')
 ax[2].set_title('Candidate–lineage correlations',loc='left'); fig.colorbar(im,ax=ax[2],fraction=.046,pad=.04,label='Spearman correlation')
 for i,a in enumerate(ax): a.spines[['top','right']].set_visible(False); a.text(-.13,1.08,chr(65+i),transform=a.transAxes,fontsize=15,fontweight='bold',va='top')
 fig.text(.055,.015,'Good response: Huvos grades 3–4. Reference cells are cultured MSCs/osteoblasts; tumor–reference contrasts may include culture effects.',fontsize=8,color='#6B7280')
 for ext,kw in [('png',{'dpi':400}),('pdf',{}),('tiff',{'dpi':400,'pil_kwargs':{'compression':'tiff_lzw'}})]: fig.savefig(OUT/f'Supplementary_Figure_S3_GSE42352_external_context.{ext}',bbox_inches='tight',facecolor='white',**kw)
 plt.close(fig)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--sample-dir',type=Path); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); sp=RAW/'GSE42352_candidate_module_expression.tsv'; e=build(a.sample_dir,sp) if a.sample_dir else pd.read_csv(sp,sep='\t',index_col=0); d,cov=scores(e,clinical(RAW/'E-GEOD-42352.sdrf.txt')); models,ordinal,cor,ctx=analyse(d); d.to_csv(OUT/'GSE42352_candidate_module_clinical.tsv',sep='\t',index=False); cov.to_csv(OUT/'GSE42352_module_coverage.tsv',sep='\t',index=False); models.to_csv(OUT/'GSE42352_huvos_logistic_models.tsv',sep='\t',index=False); ordinal.to_csv(OUT/'GSE42352_huvos_ordinal_model.tsv',sep='\t',index=False); cor.to_csv(OUT/'GSE42352_candidate_module_spearman.tsv',sep='\t'); ctx.to_csv(OUT/'GSE42352_tumor_reference_tests.tsv',sep='\t',index=False); figure(d,models,cor); paths=[Path(__file__),RAW/'E-GEOD-42352.sdrf.txt',RAW/'GPL10295_platform_data.txt',sp,ROOT/'outputs/GSE152048_patient_robust_signatures.tsv']+sorted(OUT.glob('*.tsv')); (OUT/'reproducibility_sha256.txt').write_text(''.join(f'{hashlib.sha256(x.read_bytes()).hexdigest()}  {x.relative_to(ROOT)}\n' for x in paths)); print(models.to_string(index=False)); print('\nOrdinal\n',ordinal.to_string(index=False)); print('\nContext\n',ctx.to_string(index=False)); print('\nCorrelations\n',cor.loc[[g+'_z' for g in CANDIDATES],[m+'_z' for m in MODULES]].to_string())
if __name__=='__main__': main()
