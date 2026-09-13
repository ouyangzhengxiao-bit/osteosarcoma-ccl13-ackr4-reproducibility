#!/usr/bin/env python3
"""Composition adjustment and incremental-value audit for CCL13-ACKR4 in TARGET-OS."""
from pathlib import Path
import hashlib
import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/quality_upgrade_v4'
OUT.mkdir(parents=True,exist_ok=True)
GENES={'ENSG00000129048':'ACKR4','ENSG00000181374':'CCL13'}
MODULES=['T_NK','Myeloid','Osteoclast','Fibroblast_MSC','Endothelial']

def z(x):
    x=pd.Series(x,dtype=float)
    return (x-x.mean())/x.std(ddof=1)

def load_data():
    e=pd.read_csv(ROOT/'work/data/raw/survival/TARGET_OS_CCL13_ACKR4_uqfpkm.tsv',sep='\t',index_col=0)
    e.index=e.index.astype(str).str.replace(r'\.[0-9]+$','',regex=True)
    e=e.rename(index=GENES).T[['CCL13','ACKR4']].astype(float)
    e.index.name='case_id'
    d=pd.read_csv(ROOT/'outputs/TARGET_OS_module_clinical_analysis.tsv',sep='\t',index_col=0).join(e,how='inner')
    d['metastasis']=d['metastasis_at_diagnosis'].map({'Metastasis, NOS':1,'No Metastasis':0})
    d['sex_male']=d['sex'].astype(str).str.lower().map({'male':1,'female':0})
    d['CCL13_log2']=np.log2(d['CCL13']+1)
    d['ACKR4_log2']=np.log2(d['ACKR4']+1)
    d['CCL13_z']=z(d['CCL13_log2']); d['ACKR4_z']=z(d['ACKR4_log2'])
    d['pair_original']=(d['CCL13_z']+d['ACKR4_z'])/2
    d['pair_mean_z']=z(d['pair_original'])
    d['pair_min_z']=z(np.minimum(d['CCL13_z'],d['ACKR4_z']))
    rp=d[['CCL13_log2','ACKR4_log2']].rank(pct=True)
    d['pair_rank_product_z']=z(np.sqrt(rp.CCL13_log2*rp.ACKR4_log2))
    for m in MODULES: d[m+'_z']=z(d[m])
    return d

def fit_logit(d,name,formula,term):
    vars_=list(dict.fromkeys([x for x in formula.replace('~','+').replace('*','+').split('+') if x.strip()]))
    fit=smf.logit(formula,d).fit(disp=0,maxiter=500)
    ci=fit.conf_int().loc[term]
    return fit, {'endpoint':'metastasis','model':name,'term':term,'n':int(fit.nobs),'events':int(fit.model.endog.sum()),
                 'beta':fit.params[term],'effect':np.exp(fit.params[term]),'ci_low':np.exp(ci.iloc[0]),
                 'ci_high':np.exp(ci.iloc[1]),'p':fit.pvalues[term],'aic':fit.aic,'log_likelihood':fit.llf,
                 'df_model':fit.df_model}

def likelihood_ratio(reduced,full,label):
    stat=2*(full.llf-reduced.llf); df=int(full.df_model-reduced.df_model)
    return {'comparison':label,'lr_chisq':stat,'df':df,'p':chi2.sf(stat,df)}

def bootstrap_pair(d,formula,term='pair_mean_z',B=2000,seed=20260909):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(B):
        q=d.iloc[rng.integers(0,len(d),len(d))]
        try:
            fit=smf.logit(formula,q).fit(disp=0,maxiter=300)
            v=fit.params.get(term,np.nan)
            if np.isfinite(v): vals.append(v)
        except Exception: pass
    vals=np.asarray(vals)
    return {'bootstrap_replicates':B,'bootstrap_valid':len(vals),
            'bootstrap_or_median':float(np.exp(np.median(vals))),
            'bootstrap_ci_low':float(np.exp(np.quantile(vals,.025))),
            'bootstrap_ci_high':float(np.exp(np.quantile(vals,.975)))}

def cv_models(d):
    q=d.dropna(subset=['metastasis','age_at_diagnosis_years','sex_male','pair_mean_z','Myeloid_z','Osteoclast_z']).copy()
    y=q.metastasis.astype(int).to_numpy()
    specs={'clinical':['age_at_diagnosis_years','sex_male'],
           'clinical_pair':['age_at_diagnosis_years','sex_male','pair_mean_z'],
           'clinical_lineages':['age_at_diagnosis_years','sex_male','Myeloid_z','Osteoclast_z'],
           'clinical_lineages_pair':['age_at_diagnosis_years','sex_male','Myeloid_z','Osteoclast_z','pair_mean_z']}
    rows=[]
    for repeat in range(200):
        cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=1,random_state=1000+repeat)
        for name,cols in specs.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000))
            pred=cross_val_predict(model,q[cols],y,cv=cv,method='predict_proba')[:,1]
            rows.append({'repeat':repeat,'model':name,'auc':roc_auc_score(y,pred),'brier':brier_score_loss(y,pred)})
    out=pd.DataFrame(rows)
    wide=out.pivot(index='repeat',columns='model',values=['auc','brier'])
    contrasts=[]
    for metric in ['auc','brier']:
        for base,added,label in [('clinical','clinical_pair','pair_vs_clinical'),
                                 ('clinical_lineages','clinical_lineages_pair','pair_vs_clinical_lineages')]:
            delta=wide[(metric,added)]-wide[(metric,base)]
            contrasts.append({'metric':metric,'comparison':label,'repeats':len(delta),
                              'median_delta':delta.median(),'q1_delta':delta.quantile(.25),'q3_delta':delta.quantile(.75),
                              'fraction_improved':float(np.mean(delta>0 if metric=='auc' else delta<0))})
    return out,pd.DataFrame(contrasts)

def fit_cox(d,name,formula,term):
    q=d.dropna(subset=['os_days','os_event']+[x for x in ['age_at_diagnosis_years','sex_male','metastasis','pair_mean_z','Myeloid_z','Osteoclast_z','CCL13_z','ACKR4_z'] if x in formula]).copy()
    q=q[q.os_days>0]
    fit=PHReg.from_formula(formula,status=q.os_event.astype(int),data=q).fit(disp=0)
    names=fit.model.exog_names; j=names.index(term); ci=fit.conf_int()[j]
    return {'endpoint':'overall_survival','model':name,'term':term,'n':len(q),'events':int(q.os_event.sum()),
            'beta':fit.params[j],'effect':np.exp(fit.params[j]),'ci_low':np.exp(ci[0]),'ci_high':np.exp(ci[1]),'p':fit.pvalues[j]}

def plot_results(models,cv,cor):
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.titleweight':'bold',
                         'text.color':'#17324D','axes.labelcolor':'#17324D','axes.edgecolor':'#A8B3BD',
                         'xtick.color':'#465563','ytick.color':'#465563','figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42})
    fig,ax=plt.subplots(1,3,figsize=(15,5.2),gridspec_kw={'wspace':.42})
    q=models[(models.endpoint=='metastasis')&(models.term.isin(['pair_mean_z','pair_min_z','pair_rank_product_z']))].copy()
    order=['Mean z score clinical','Mean z score plus lineages','Minimum z score plus lineages','Rank product plus lineages']
    q=q.set_index('model').reindex(order).reset_index(); y=np.arange(len(q))[::-1]
    for yy,row in zip(y,q.itertuples()):
        ax[0].plot([row.ci_low,row.ci_high],[yy,yy],color='#2F6B9A',lw=1.8)
        ax[0].scatter(row.effect,yy,s=42,color='#D95F59' if row.p<.05 else '#2F6B9A',edgecolor='white',zorder=3)
    ax[0].axvline(1,color='#7A8793',ls='--'); ax[0].set_xscale('log'); ax[0].set_yticks(y,[x.replace(' plus ','\nplus ') for x in order]); ax[0].set_xlim(.9,7); ax[0].set_xticks([1,2,4,6],['1','2','4','6'],minor=False); ax[0].set_xticks([],minor=True); ax[0].set_xlabel('Metastasis odds ratio per 1 SD'); ax[0].set_title('Association persists across score definitions',loc='left'); ax[0].grid(axis='x',color='#D9E1E8'); ax[0].spines[['top','right']].set_visible(False)
    c=cv.pivot(index='repeat',columns='model',values='auc')
    vals=[c.clinical_pair-c.clinical,c.clinical_lineages_pair-c.clinical_lineages]
    bp=ax[1].boxplot(vals,patch_artist=True,tick_labels=['Add pair to\nclinical model','Add pair to clinical\nplus lineages'],showfliers=False)
    for b,col in zip(bp['boxes'],['#93D4C8','#E9A23B']): b.set_facecolor(col)
    ax[1].axhline(0,color='#7A8793',ls='--'); ax[1].set_ylabel('Repeated 5-fold CV ΔAUC'); ax[1].set_title('Incremental discrimination is internal only',loc='left'); ax[1].grid(axis='y',color='#D9E1E8'); ax[1].spines[['top','right']].set_visible(False)
    im=ax[2].imshow(cor,cmap='RdBu_r',vmin=-1,vmax=1)
    labels=[x.replace('_z','').replace('_','/') for x in cor.columns]; ax[2].set_xticks(range(len(labels)),labels,rotation=45,ha='right'); ax[2].set_yticks(range(len(labels)),labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax[2].text(j,i,f'{cor.iat[i,j]:+.2f}',ha='center',va='center',fontsize=7,color='white' if abs(cor.iat[i,j])>.55 else '#17324D')
    ax[2].set_title('Candidate genes correlate with distinct lineage programs',loc='left'); [s.set_visible(False) for s in ax[2].spines.values()]; fig.colorbar(im,ax=ax[2],fraction=.046,pad=.04,label='Spearman correlation')
    for i,a in enumerate(ax): a.text(-.13,1.08,chr(65+i),transform=a.transAxes,fontsize=15,fontweight='bold',va='top')
    fig.text(.055,.02,'Lineage covariates: patient-derived myeloid and osteoclast module scores. Cross-validation does not correct the original candidate-selection bias.',fontsize=8,color='#6B7280')
    for ext,kwargs in [('png',{'dpi':400}),('pdf',{}),('tiff',{'dpi':400,'pil_kwargs':{'compression':'tiff_lzw'}})]: fig.savefig(OUT/f'Supplementary_Figure_S2_TARGET_incremental_value.{ext}',bbox_inches='tight',facecolor='white',**kwargs)
    plt.close(fig)

def main():
    d=load_data(); d.to_csv(OUT/'TARGET_candidate_module_clinical_analysis.tsv',sep='\t',index=True)
    formulas=[
      ('Mean z score clinical','metastasis ~ pair_mean_z + age_at_diagnosis_years + sex_male','pair_mean_z'),
      ('Mean z score plus lineages','metastasis ~ pair_mean_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','pair_mean_z'),
      ('Minimum z score plus lineages','metastasis ~ pair_min_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','pair_min_z'),
      ('Rank product plus lineages','metastasis ~ pair_rank_product_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','pair_rank_product_z'),
      ('CCL13 plus lineages','metastasis ~ CCL13_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','CCL13_z'),
      ('ACKR4 plus lineages','metastasis ~ ACKR4_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','ACKR4_z'),
      ('Separate genes plus lineages','metastasis ~ CCL13_z + ACKR4_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','ACKR4_z'),
      ('Interaction plus lineages','metastasis ~ CCL13_z * ACKR4_z + age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z','CCL13_z:ACKR4_z')]
    fits={}; rows=[]
    for name,formula,term in formulas:
        fit,row=fit_logit(d,name,formula,term); fits[name]=fit; rows.append(row)
    separate=fits['Separate genes plus lineages']
    ci=separate.conf_int().loc['CCL13_z']
    rows.append({'endpoint':'metastasis','model':'Separate genes plus lineages','term':'CCL13_z',
                 'n':int(separate.nobs),'events':int(separate.model.endog.sum()),
                 'beta':separate.params['CCL13_z'],'effect':np.exp(separate.params['CCL13_z']),
                 'ci_low':np.exp(ci.iloc[0]),'ci_high':np.exp(ci.iloc[1]),'p':separate.pvalues['CCL13_z'],
                 'aic':separate.aic,'log_likelihood':separate.llf,'df_model':separate.df_model})
    lr=[]
    clinical=smf.logit('metastasis ~ age_at_diagnosis_years + sex_male',d).fit(disp=0)
    lineages=smf.logit('metastasis ~ age_at_diagnosis_years + sex_male + Myeloid_z + Osteoclast_z',d).fit(disp=0)
    lr.append(likelihood_ratio(clinical,fits['Mean z score clinical'],'mean_pair_beyond_clinical'))
    lr.append(likelihood_ratio(lineages,fits['Mean z score plus lineages'],'mean_pair_beyond_clinical_lineages'))
    lr.append(likelihood_ratio(fits['Separate genes plus lineages'],fits['Interaction plus lineages'],'interaction_beyond_gene_main_effects'))
    pd.DataFrame(lr).to_csv(OUT/'TARGET_pair_likelihood_ratio_tests.tsv',sep='\t',index=False)
    boot=bootstrap_pair(d.dropna(subset=['metastasis','age_at_diagnosis_years','sex_male','pair_mean_z','Myeloid_z','Osteoclast_z']),formulas[1][1])
    for name,formula,term in [
      ('OS mean z clinical','os_days ~ pair_mean_z + age_at_diagnosis_years + sex_male + metastasis','pair_mean_z'),
      ('OS mean z plus lineages','os_days ~ pair_mean_z + age_at_diagnosis_years + sex_male + metastasis + Myeloid_z + Osteoclast_z','pair_mean_z')]:
        rows.append(fit_cox(d,name,formula,term))
    models=pd.DataFrame(rows)
    for k,v in boot.items(): models.loc[models.model=='Mean z score plus lineages',k]=v
    models.to_csv(OUT/'TARGET_pair_model_sensitivity.tsv',sep='\t',index=False)
    cv,contrasts=cv_models(d); cv.to_csv(OUT/'TARGET_pair_repeated_cv.tsv',sep='\t',index=False); contrasts.to_csv(OUT/'TARGET_pair_cv_contrasts.tsv',sep='\t',index=False)
    cols=['CCL13_z','ACKR4_z','Myeloid_z','Osteoclast_z','Fibroblast_MSC_z','Endothelial_z','T_NK_z']
    cor=d[cols].corr(method='spearman'); cor.to_csv(OUT/'TARGET_candidate_module_spearman.tsv',sep='\t')
    plot_results(models,cv,cor)
    with open(OUT/'reproducibility_sha256.txt','w') as h:
        for path in [Path(__file__),ROOT/'work/data/raw/survival/TARGET_OS_CCL13_ACKR4_uqfpkm.tsv',ROOT/'outputs/TARGET_OS_module_clinical_analysis.tsv']+sorted(OUT.glob('*.tsv')):
            if path.name=='reproducibility_sha256.txt': continue
            h.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT))+'\n')
    print(models.to_string(index=False)); print('\nLikelihood ratio tests\n',pd.DataFrame(lr).to_string(index=False)); print('\nCV contrasts\n',contrasts.to_string(index=False))
if __name__=='__main__': main()
