import sys
import os
# Add the path where functions.py is located
sys.path.append('/home/jiguo/denovo_rpe1_scrnaseq/analysis_pipeline')
import functions as f
import pandas as pd
adata = f.load_adata(adata_file='/home/jiguo/data/data/anndata/adata_all_pp_cc.h5', gene_list=None)
RNA_meta = f.load_gene_list(gene_list_file="/home/jiguo/denovo_rpe1_scrnaseq/Genes_with_GO_terms_RNA-metabolic-process - select.csv", gene_index="Gene names")
DE_genes = f.load_gene_list(gene_list_file="/home/jiguo/denovo_rpe1_scrnaseq/all_DE_genes.txt", gene_index="Gene name")
RNA_meta = f.check_genes_in_data(RNA_meta, adata)
DE_genes = f.check_genes_in_data(DE_genes, adata)
corr_centrinone = pd.read_csv('/home/jiguo/output/correlation/corr_centrinone.csv', index_col=0)
corr_control = pd.read_csv('/home/jiguo/output/correlation/corr_control.csv', index_col=0)
f.plot_gene_network_old(
    corr_control,
    gene_list1=RNA_meta,
    gene_list2=DE_genes,
    threshold=0.5,
    iteration=25,
    title='control',
)
f.plot_gene_network_old(
    corr_centrinone,
    gene_list1=RNA_meta,
    gene_list2=DE_genes,
    threshold=0.5,
    iteration=25,
    title='centrinone',
)
f.plot_correlation_heatmap(
    corr_control,
    gene_list1=RNA_meta,
    gene_list2=DE_genes,
    cluster=True,
    title='control',
)
f.plot_correlation_heatmap(
    corr_centrinone,
    gene_list1=RNA_meta,
    gene_list2=DE_genes,
    cluster=True,
    title='centrinone',
)