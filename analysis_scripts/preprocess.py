import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

# Load data and set condition as categorical
adata = ad.read_h5ad("/home/jiguo/data/data/anndata/adata_all.h5")
cat_dtype = pd.CategoricalDtype(['RPE1-Ctrl', 'RPE1-Plus-Centrinone', 'RPE1-2h-WO', 'RPE1-8h-WO'], ordered=True)
adata.obs["condition"] = adata.obs["condition"].astype(cat_dtype)

# Calculate QC metrics
adata.obs['n_counts'] = adata.X.sum(axis=1).A1
adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1
mt_gene_mask = [gene.startswith('MT-') for gene in adata.var_names]
adata.obs['mt_frac'] = np.sum(adata[:, mt_gene_mask].X, axis=1).A1 / adata.obs['n_counts']

# Filter cells: min genes & max mt-frac
cell_mask = (adata.obs['n_genes'] >= 200) & (adata.obs['mt_frac'] <= 0.2)
adata = adata[cell_mask].copy()

# Filter genes: min_cells first
sc.pp.filter_genes(adata, min_cells=20)

# Select the genes of interest
gene_list = pd.read_csv("/home/jiguo/denovo_rpe1_scrnaseq/Genes_with_GO_terms_RNA-metabolic-process - select.csv")["Gene names"].tolist()
gene_list = [gene.upper() for gene in gene_list]  # Match case if needed
gene_mask = adata.var_names.isin(gene_list)
adata = adata[:, gene_mask].copy()

# Recalculate log counts AFTER filtering
adata.obs['log_n_counts'] = np.log(adata.obs['n_counts'])

# Plot QC (now reflects filtered data)
sc.pl.violin(adata, ['n_counts', 'n_genes', 'mt_frac'], jitter=0.4, multi_panel=True)
plt.savefig("/home/jiguo/output/adata_quality_control.png", dpi=300, bbox_inches='tight')

# Save
adata.write_h5ad("/home/jiguo/output/adata_all_preprocessed.h5")
