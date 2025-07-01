# GeneSet-Mapper

A comprehensive Python package for gene set analysis in Genomic data, featuring differential expression analysis, gene correlation networks, and protein localization mapping.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Main Features](#main-features)

## Installation

```bash
# Clone the repository
git clone https://github.com/NicoNiCoN11/GeneSet-Mapper.git
cd GeneSet-Mapper

# Install required dependencies
pip install scanpy pandas numpy seaborn matplotlib scipy networkx scikit-learn
```

## Quick Start

```python
import modules as gsm

# Load your single-cell data
adata = gsm.load_adata('path/to/your/data.h5ad')

# Load a gene list of interest
gene_list = gsm.load_gene_list('path/to/genes.txt')

# Check which genes are present in your data
filtered_genes = gsm.check_genes_in_data(gene_list, adata)

# Calculate gene expression percentages and visualize
gsm.plot_simple_violin(adata, percentage_col='your_gene_percentage')
```

## Main Features

### 🧬 Gene set specific Expression Analysis
- Load the adata while only keep the genes of interest
- Expression percentage of the given genes in the gene set calculation across conditions
- Violin plots with cell cycle phase stratification
- Gene list filtering and validation

### 📊 Differential Expression Analysis
- Pseudobulk creation from single-cell data with a given gene set
- Multiple DE analysis methods
- Interactive volcano plots with gene highlighting

### 🕸️ Network Analysis
- Gene correlation matrices
- Co-expression network construction
- Spectral clustering of gene networks

### 🎯 Protein Localization
- Subcellular localization mapping
- Reliability scoring system
- Visualization of protein distributions

