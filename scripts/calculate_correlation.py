import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
def filter_condition(adata, condition):
    mask = adata.obs['condition'].isin([condition])
    return adata[mask].copy()
adata = sc.read_h5ad('/home/jiguo/data/data/anndata/adata_all_pp_cc.h5')
# chaneg condition to string
adata.obs['condition'] = adata.obs['condition'].astype(str)
control_anndata = filter_condition(adata, 'control')
centrinone_anndata = filter_condition(adata, 'centrinone')
wo2h_anndata = filter_condition(adata, 'wo-2h')
wo8h_anndata = filter_condition(adata, 'wo-8h')
def calculate_correlation_matrix(adata):
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    return df.corr(method='pearson')
corr_control = calculate_correlation_matrix(control_anndata)
corr_centrinone = calculate_correlation_matrix(centrinone_anndata)
corr_wo2h = calculate_correlation_matrix(wo2h_anndata)
corr_wo8h = calculate_correlation_matrix(wo8h_anndata)
# store the correlation matrices
corr_control.to_csv('/home/jiguo/output/correlation/corr_control.csv')
corr_centrinone.to_csv('/home/jiguo/output/correlation/corr_centrinone.csv')
corr_wo2h.to_csv('/home/jiguo/output/correlation/corr_wo2h.csv')
corr_wo8h.to_csv('/home/jiguo/output/correlation/corr_wo8h.csv')