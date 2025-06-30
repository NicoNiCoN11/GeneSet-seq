import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import sparse
import matplotlib.pyplot as plt
import os
import networkx as nx
from matplotlib.lines import Line2D


"""
This module contains functions for processing single-cell RNA-seq data,
these functions are specialized for gene-set related single cell analysis, and visualization
including:
- Loading AnnData objects from files
- Gene expression distribution analysis across conditions and phases
- Gene expression percentage calculation
- Gene expression visualization using violin plots

- Loading a gene list from a file
- Filtering adata by condition
- Checking if genes in a given list are present in an AnnData object

- Creating a bulk AnnData object from single-cell data with optional gene filtering
- Extract Differentially Expressed Genes (DEGs) from an Pseudobulk AnnData object
- Plotting volcano plots for DEGs to visualize the results
- Plotting customized volcano plots with specific genes of interest highlighted

- Calculating correlation matrices for gene expression
- Calculating coexpression changes between two correlation matrices
- Plotting correlation heatmaps for specified gene lists

- Constructing and plotting gene regulatory networks (GRNs) based on correlation matrices

- Find the proteins' locations of the given gene lists
"""
def load_adata(adata_file, gene_list=None):
    if not os.path.exists(adata_file):
        raise FileNotFoundError(f"File {adata_file} not found")
    adata = sc.read(adata_file)
    if gene_list is not None:
        # if given a path, load the list, if given a list, use it directly
        if isinstance(gene_list, str):
            gene_list = load_gene_list(gene_list)
        elif isinstance(gene_list, list):
            gene_list = gene_list
        gene_mask = adata.var_names.isin(gene_list)
        adata = adata[:, gene_mask].copy()
    return adata


# these functions are used to calculate the percentage of expression of selected genes in the full adata object, and visualize the distribution of expression percentages across different conditions and phases.
def calculate_expression_percentage(full_adata, selected_adata):    
    if 'counts' in selected_adata.layers:
        selected_counts = np.array(selected_adata.layers['counts'].sum(axis=1)).flatten()
    else:
        selected_counts = np.array(selected_adata.X.sum(axis=1)).flatten()   
    if 'total_counts' in full_adata.obs.columns:
        total_counts = full_adata.obs['total_counts'].values
    else:
        if 'counts' in full_adata.layers:
            total_counts = np.array(full_adata.layers['counts'].sum(axis=1)).flatten()
        else:
            total_counts = np.array(full_adata.X.sum(axis=1)).flatten()
    selected_gene_exp_percentage = (selected_counts / total_counts) * 100
    return selected_gene_exp_percentage


def plot_Gene_expression_violin(adata, selected_adata, palette='Set1'):
    percentage_col = calculate_expression_percentage(
        full_adata=adata, 
        selected_adata=adata
    )
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 1. Create violin plot (split by phase)
    sns.violinplot(
        data=adata.obs,
        x='condition',
        y=percentage_col,
        hue='phase',
        inner='quartile',  # Show quartile lines
        palette='dark',
        dodge=True,       # Split violins by phase
        ax=ax,
        legend=True,  # Hide legend for now
    )
    # 2. Overlay stripplot with jittered points
    sns.stripplot(
        data=adata.obs,
        x='condition',
        y=percentage_col,
        hue='phase',
        dodge=True,        # Align points with violins
        jitter=0.2,        # Spread points horizontally
        size=2,            # Point size
        palette='gray',  # Use dark palette for points
        linewidth=0.5,
        ax=ax,
        legend=False       # Reuse violin legend
    )
    plt.title('Genes of interest Expression by Condition and Phase', 
              fontsize=16, pad=20)
    plt.xlabel('Condition', fontsize=14)
    plt.ylabel('Expression (%)', fontsize=14)
    plt.xticks(rotation=0)  # Keep labels horizontal
    plt.grid(axis='y', alpha=0.3)
    # Consolidate legends
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:len(adata.obs['phase'].unique())],  # Unique phases
               labels[:len(adata.obs['phase'].unique())], 
               title='Cell Cycle Phase',
               loc='best')
    plt.tight_layout()
    plt.show()

def plot_simple_violin(adata, percentage_col='rna_metabolic_percentage'):

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Violin + stripplot combo
    sns.violinplot(
        data=adata.obs,
        x='condition',
        y=percentage_col,
        color='skyblue',  # Uniform color
        inner=None        # Hide inner bars
    )
    sns.stripplot(
        data=adata.obs,
        x='condition',
        y=percentage_col,
        color='black',   # Distinct points
        size=2.5,
        alpha=0.7,       # Semi-transparent
        jitter=0.3
    )
    plt.title('Genes of interest Expression by Condition', fontsize=15)
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Expression (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()


# Gene list loading and filtering functions
def load_gene_list(gene_list_file, gene_index="Gene_name"):
    if not os.path.exists(gene_list_file):
        raise FileNotFoundError(f"File {gene_list_file} not found")
    if gene_list_file.endswith('.csv'):
        gene_list = pd.read_csv(gene_list_file)[gene_index].tolist()
    elif gene_list_file.endswith('.txt'):
        gene_list = pd.read_csv(gene_list_file, sep='\t')[gene_index].tolist()
    else:
        raise ValueError("gene_list_file must be a .csv or .txt file")
    return gene_list

def filter_condition(adata, condition):
    if 'condition' not in adata.obs.columns:
        raise KeyError("'condition' column not found in adata.obs")
    if condition not in adata.obs['condition'].unique():
        raise ValueError(f"Condition '{condition}' not found in data")
    mask = adata.obs['condition'].isin([condition])
    return adata[mask].copy()

def check_genes_in_data(gene_list, data, show_missing=False):
    """Check which genes from gene_list are present in adata and return filtered list
    """
    missing_genes = []
    # if the given input is anndata object
    if isinstance(data, sc.AnnData):
        for gene in gene_list:
            if gene not in data.var_names:
                missing_genes.append(gene)
    if isinstance(data, pd.DataFrame):
        for gene in gene_list:
            if gene not in data.columns:
                missing_genes.append(gene)
    print(f"There are {len(missing_genes)} genes out of {len(gene_list)} not found in the data object.")
    if show_missing:
        print("Missing genes:")
        for gene in missing_genes:
            print(f" - {gene}")
    return [g for g in gene_list if g not in missing_genes]



# Pseudobulk creation and DE analysis functions
def create_bulkadata(adata, gene_list=None, replicate_num=3, method='sum'):
    if 'replicate' not in adata.obs.columns:
        # create a new column 'replicate' to adata.obs give replicate number to each cell in each condition
        for condition in adata.obs['condition'].unique():
            mask = adata.obs['condition'] == condition
            adata.obs.loc[mask, 'replicate'] = np.random.choice(
                range(1, replicate_num + 1), size=mask.sum(), replace=True)
    if gene_list is not None:
        # filter adata by gene_list
        mask = adata.var_names.isin(gene_list)
        adata = adata[:, mask].copy()
    # create bulk adata
    adata.obs['sample'] = (
		adata.obs['condition'].astype(str) + '_' +
		adata.obs['replicate'].astype(str))
    pseudobulk_counts = pd.DataFrame(
        columns=adata.var_names,
        index=adata.obs['sample'].unique())
    for sample in adata.obs['sample'].unique():
        sample_mask = adata.obs['sample'] == sample
        # based on the method, we can sum or average the raw counts
        if 'counts' in adata.layers:
            if method == 'sum':
                sample_counts = adata.layers['counts'][sample_mask].sum(axis=0)
            elif method == 'mean':
                sample_counts = adata.layers['counts'][sample_mask].mean(axis=0)
        else:
            print("No 'counts' layer found in the AnnData object. Using adata.X instead.")
            if method == 'sum':
                sample_counts = adata[sample_mask].X.sum(axis=0)
            elif method == 'mean':
                sample_counts = adata[sample_mask].X.mean(axis=0)
        
        # Convert to array if sparse and assign to DataFrame
        if sparse.issparse(sample_counts):
            sample_counts = sample_counts.A1
        pseudobulk_counts.loc[sample] = sample_counts
    pseudobulk_meta = adata.obs[['condition', 'replicate', 'sample']].drop_duplicates().set_index('sample')
    pseudobulk_adata = sc.AnnData(
        X=sparse.csr_matrix(pseudobulk_counts.fillna(0)),
        obs=pseudobulk_meta,
        var=adata.var)
    return pseudobulk_adata

def extract_de_results(adata, condition, output_path=None, threshold_pval=0.05, threshold_logfc=1):
    # The input parameter 'condition' should be the CONTROL condition
    all_conditions = adata.obs['condition'].unique()
    control_condition = condition
    conditions = [c for c in all_conditions if c != control_condition]
    de_results = {}
    all_results_list = []  # To store all results for saving later
    for cond in conditions:
        de_df = sc.get.rank_genes_groups_df(
            adata, 
            group=cond,
            key='rank_genes_groups' 
        )
        de_df['-log10_pvals_adj'] = -np.log10(de_df['pvals_adj'])  # Use adjusted p-values
        de_df = de_df.merge(
            adata.var[['gene_name']],
            left_on='names',
            right_index=True
        )
        # Add condition information
        de_df['comparison'] = f"{control_condition}_vs_{cond}"
        de_df['control'] = control_condition
        de_df['treatment'] = cond
        de_results[cond] = de_df
        all_results_list.append(de_df)
        # add "significance" column based on thresholds to show significant DEGs, add 'significant' and 'not significant' in this column
        for gene in de_df['names']:
            if (de_df.loc[de_df['names'] == gene, '-log10_pvals_adj'].values[0] > -np.log10(threshold_pval)) and \
               (abs(de_df.loc[de_df['names'] == gene, 'logfoldchanges'].values[0]) > threshold_logfc):
                de_df.loc[de_df['names'] == gene, 'significance'] = 'significant'
            else:
                de_df.loc[de_df['names'] == gene, 'significance'] = 'not significant'
    # Concatenate all results into a single DataFrame
    all_results_df = pd.concat(all_results_list)
    if output_path is not None:
        output_file = f"{output_path}/de_results_{control_condition}.csv"
        all_results_df.to_csv(output_file, index=False)  # Fixed variable name
        print(f"DE results saved to {output_file} with condition information")
    return de_results

def plot_DEvolcano(de_results, control_condition, threshold_pval=0.05, threshold_logfc=1, figsize=(15, 6), title_fontsize=16, font_size=14):
    # Get conditions from de_results keys, not adata.obs
    conditions = list(de_results.keys())
    n_conditions = len(conditions)
    # Calculate grid dimensions
    n_cols = min(2, n_conditions)
    n_rows = (n_conditions + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 6*n_rows))
    sns.set(style='whitegrid', font_scale=1.2)
    
    for i, cond in enumerate(conditions, 1):
        df = de_results[cond]
        ax = plt.subplot(n_rows, n_cols, i) 
        df['significance'] = 'Not significant'
        df.loc[(df['pvals_adj'] < threshold_pval) &  # Use ADJUSTED p-values
               (df['logfoldchanges'] > threshold_logfc), 'significance'] = 'Up-regulated'
        df.loc[(df['pvals_adj'] < threshold_pval) &  # Use ADJUSTED p-values
               (df['logfoldchanges'] < -threshold_logfc), 'significance'] = 'Down-regulated'
        sns.scatterplot(
            data=df,
            x='logfoldchanges',
            y='-log10_pvals_adj',
            hue='significance',
            palette={'Up-regulated': '#e41a1c', 
                    'Down-regulated': '#377eb8',
                    'Not significant': '#bdbdbd'},
            alpha=0.7,
            s=40,
            ax=ax)
        # Gene labeling - better approach
        df['combined_score'] = np.abs(df['logfoldchanges']) * df['-log10_pvals_adj']
        top_genes = df.nlargest(10, 'combined_score')

        for _, row in top_genes.iterrows():
            ax.text(
                row['logfoldchanges'],
                row['-log10_pvals_adj'] + 0.1,  # Offset to avoid overlap
                row['gene_name'],
                fontsize=9,
                alpha=0.8,
                fontweight='bold'
            )
        # Add thresholds
        ax.axhline(-np.log10(threshold_pval), color='gray', linestyle='--', alpha=0.7)
        ax.axvline(threshold_logfc, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(-threshold_logfc, color='gray', linestyle='--', alpha=0.7)    
        # Correct title
        ax.set_title(f'{control_condition} vs {cond}', fontsize=title_fontsize)  # Fixed title
        ax.set_xlabel('Log2 Fold Change', fontsize=font_size)
        ax.set_ylabel('-Log10(Adjusted p-value)', fontsize=font_size)
        ax.set_xlim(df['logfoldchanges'].min()-0.5, df['logfoldchanges'].max()+0.5)
        ax.legend().remove()  # Remove individual legends
    # Add common legend
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3, 
                  bbox_to_anchor=(0.5, -0.05 if n_rows > 1 else -0.1))
    
    return plt

def plot_customize_volcano(de_results, control_condition, gene_of_interest_1, 
                          gene_of_interest_2=None, gene_of_interest_3=None, 
                          threshold_pval=0.05, threshold_logfc=1, figsize=(15, 6), 
                          title_fontsize=16, font_size=14):
    """
    plot customized volcano plot for DE results with specific genes of interest highlighted.
    """
    # Get conditions from de_results keys, not adata.obs
    conditions = list(de_results.keys())
    n_conditions = len(conditions)
    # if give a path, load the list, if give a list, use it directly, gothrough this process for every gene_list of interest
    for gene_list in [gene_of_interest_1, gene_of_interest_2, gene_of_interest_3]:
        if isinstance(gene_list, str):
            gene_list = load_gene_list(gene_list)
        elif isinstance(gene_list, list):
            gene_list = gene_list
        else:
            raise ValueError("gene_of_interest should be a file path or a list of genes")
    # Calculate grid dimensions
    n_cols = min(2, n_conditions)
    n_rows = (n_conditions + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 6*n_rows))
    sns.set(style='whitegrid', font_scale=1.2)
    
    for i, cond in enumerate(conditions, 1):
        df = de_results[cond]
        ax = plt.subplot(n_rows, n_cols, i)  # Fixed index: use i not i+1
        n=1
        # Significance thresholds (use pvals_adj for FDR)
        sig_threshold = threshold_pval
        logfc_threshold = threshold_logfc
        df['significance'] = 'Not significant'
        for gene_list in [gene_of_interest_1, gene_of_interest_2, gene_of_interest_3]:
            df[f'is in gene_list {n}'] = df['gene_name'].isin(gene_list)
            df['significance'] = np.where(df[f'is in gene_list {n}'], f'gene_list{n}', df['significance'])
            n += 1
        df['significance'] = pd.Categorical(df['significance'], 
                                            categories=['gene_list1', 'gene_list2', 'gene_list3','Not significant'],)
        sns.scatterplot(
            data=df,
            x='logfoldchanges',
            y='-log10_pvals_adj',  # Match the column name
            hue='significance',
            palette={
                'gene_list1': '#ff7f00',  # Orange for genes in gene_list1
                'gene_list2': '#4daf4a',  # Green for genes in gene_list2
                'gene_list3': '#ff00ff',  # Magenta for genes in gene_list3
                'Not significant': '#bdbdbd'  # Gray for non-significant genes
            },
            alpha=0.7,
            s=40,
            ax=ax
        )
        # Gene labeling - better approach
        df['combined_score'] = np.abs(df['logfoldchanges']) * df['-log10_pvals_adj']
        top_genes = df.nlargest(10, 'combined_score')

        for _, row in top_genes.iterrows():
            ax.text(
                row['logfoldchanges'],
                row['-log10_pvals_adj'] + 0.1,  # Offset to avoid overlap
                row['gene_name'],
                fontsize=9,
                alpha=0.8,
                fontweight='bold'
            )
        # Add thresholdsa
        ax.axhline(-np.log10(sig_threshold), color='gray', linestyle='--', alpha=0.7)
        ax.axvline(logfc_threshold, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(-logfc_threshold, color='gray', linestyle='--', alpha=0.7)
        # Correct title
        ax.set_title(f'{control_condition} vs {cond}', fontsize=16)  # Fixed title
        ax.set_xlabel('Log2 Fold Change', fontsize=14)
        ax.set_ylabel('-Log10(Adjusted p-value)', fontsize=14)
        ax.set_xlim(df['logfoldchanges'].min()-0.5, df['logfoldchanges'].max()+0.5)
        ax.legend().remove()  # Remove individual legends
    
    # Add common legend
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3)
    return plt


def plotting_volcano(adata, control_condition, method='t-test_overestim_var', output_path=None, volcano_plot_method="default", gene_of_interest_1=None, gene_of_interest_2=None, gene_of_interest_3=None):
    # First run DE analysis
    sc.tl.rank_genes_groups(
        adata,
        groupby='condition',
        reference=control_condition,  # Set control as reference
        method=method,  # Specify the method for DE analysis
    )
    # Extract results
    de_results = extract_de_results(
        adata, 
        condition=control_condition # Specify control condition
    )
    # Plot
    if volcano_plot_method == "default":
        plt = plot_DEvolcano(de_results, control_condition)
    elif volcano_plot_method == "customize":
        if gene_of_interest_1 is None:
            raise ValueError("gene_of_interest_1 must be provided for customize volcano plot")
        plt = plot_customize_volcano(
            de_results, 
            control_condition, 
            gene_of_interest_1, 
            gene_of_interest_2, 
            gene_of_interest_3
        )
    plt.show()
    if output_path is not None:
        output_file = f"{output_path}/volcano_plot_{control_condition}.png"
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Volcano plot saved to {output_file}")





# Correlation and coexpression functions
def calculate_correlation_matrix(adata):
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    return df.corr(method='pearson')


def calculate_coexpression_change(corr_1, corr_2):
    coexpression_change = corr_2 - corr_1
    return coexpression_change

def plot_correlation_heatmap(corr_matrix, gene_list1, gene_list2, cluster=False, title=''):
    idx_1 = [np.where(corr_matrix.index == g)[0][0] for g in gene_list1 if g in corr_matrix.index]
    idx_2 = [np.where(corr_matrix.columns == g)[0][0] for g in gene_list2 if g in corr_matrix.columns]
    sub_corr_matrix = corr_matrix.iloc[idx_1, idx_2]
    if cluster:
        sns.clustermap(sub_corr_matrix, cmap='coolwarm',
                    figsize=(16, 10), vmin=-0.8, vmax=1)
    else:
        plt.figure(figsize=(16, 10))
        sns.heatmap(sub_corr_matrix, cmap='coolwarm',
                    vmin=-0.8, vmax=1, square=True)
    clustering_status = "Clustered" if cluster else "Unclustered"
    plt.title(f'Correlation Heatmap ({clustering_status}) for {len(gene_list1)} genes vs {len(gene_list2)} genes')
    # save image
    plt.savefig(f"correlation_heatmap_{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


# Construct a GRN for 2 given gene lists
def plot_gene_network_old(corr_matrix, gene_list1, gene_list2, threshold=0.4, title=" ", method="spring",iteration=30):
    all_genes = list(set(gene_list1 + gene_list2))
    sub_corr = corr_matrix.loc[all_genes, all_genes]
    # create a graph object
    G = nx.Graph()
    # add nodes
    for gene in all_genes:
        in_list1 = gene in gene_list1
        in_list2 = gene in gene_list2
        if in_list1 and in_list2:
            category = 'Both'
        elif in_list1:
            category = 'List1'
        elif in_list2:
            category = 'List2'
        else:
            category = 'Other'
        G.add_node(gene, category=category)
    # add edges for strong correlations
    for i, gene1 in enumerate(all_genes):
        for j, gene2 in enumerate(all_genes):
            if i<j:
                corr = sub_corr.loc[gene1, gene2]
                if abs(corr) > threshold:
                    G.add_edge(gene1, gene2, weight=corr, strength=abs(corr))

    # remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    if G.number_of_edges() == 0:
        print("No edges found with the given threshold. Adjust the threshold or check the correlation matrix.")
        return None
    if method == "spring":
        pos = nx.spring_layout(G, k=0.5, iterations=iteration,seed=42, weight='strength')
    if method == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, weight='strength', iterations=iteration, seed=42)
    # set up figure
    plt.figure(figsize=(12, 10))
    # Prepare node colors and sizes
    node_colors = {
        'List1': '#FF6B6B',  # Green for genes in gene_list1
        'List2': '#4ECDC4',  # Cyan for genes in gene_list2
        'Both': '#FFE66D',   # Yellow for genes in both lists
        'Other': '#D3D3D3'   # Light gray for other genes
    }
    # edge color based on correlation strength
    edge_colors = {
        'positive': '#9E0A0A',  # Red for positive correlations
        'negative': '#4ECDC4'   # Cyan for negative correlations
    }
    # Draw network components
    for u, v in G.edges():
        weight = G[u][v]['weight']
        color = edge_colors['positive'] if weight > 0 else edge_colors['negative']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width = np.log(abs(weight)) * 5, alpha=0.7, edge_color=color)
    for category in ['List1', 'List2', 'Both', 'Other']:
        nodes = [n for n in G.nodes if G.nodes[n]["category"] == category]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                  node_color=node_colors[category], 
                                  node_size=500)
            nx.draw_networkx_labels(G, pos, labels={n: n for n in nodes}, 
                                   font_size=10, font_weight='bold')
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=10, label='List1 Genes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=10, label='List2 Genes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFE66D', markersize=10, label='Both Lists'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3', markersize=10, label='Other Genes'),
        Line2D([0], [0], color='#9E0A0A', lw=2, label='Positive Correlation'),
        Line2D([0], [0], color='#4ECDC4', lw=2, label='Negative Correlation')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title(f"Co-expression Network: {title}\n(|r| > {threshold})", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    # save image 
    plt.savefig(f"gene_network_{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


# contruct a Gene correaltion network via networkx by calculate laplacian mmatrix
def build_gene_network(corr_matrix, gene_list1=None, gene_list2=None, threshold=0.5):
    """
    This function build a gene correlation network and compute the laplacian matrix based on its correlation matrix.
    """
    if not isinstance(corr_matrix, pd.DataFrame):
        raise TypeError("corr_matrix must be a pandas DataFrame")
    # if only gene_list1 is given, use it as both lists
    if gene_list2 is None:
        gene_list2 = gene_list1
    # if no gene list is given, use all genes in the correlation matrix
    if gene_list1 is None:
        gene_list1 = corr_matrix.index.tolist()
    
    all_genes = list(set(gene_list1+ gene_list2))
    all_genes = check_genes_in_data(all_genes, corr_matrix)
    # create a submatrix for target genes
    sub_corr = corr_matrix.loc[all_genes, all_genes].copy()
    G=nx.Graph()
    for gene in all_genes:
        category = ('both' if (gene in gene_list1 and gene in gene_list2) else
                    'gene_list1' if (gene in gene_list1) else
                    'gene_list2')
        G.add_node(gene, category=category)
    # Add edges 
    triu_indices = np.triu_indices_from(sub_corr.values, k=1)
    for i, j in zip(triu_indices[0], triu_indices[1]):
        corr_val = sub_corr.iloc[i, j]
        if abs(corr_val) > threshold:
            gene1 = sub_corr.index[i]
            gene2 = sub_corr.columns[j]
            G.add_edge(gene1, gene2, weight=corr_val)
    
    # remove the isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    node_index=pd.Index(G.nodes())
    A=nx.adjacency_matrix(G, nodelist=node_index, weight='weight')
    # compute laplacian
    D=sparse.diags(np.array(A.sum(axis=1)).flatten())
    L=D-A
    return G, L, node_index


def cluster_gene_network(G, node_index, n_clusters=3):
    from sklearn.cluster import SpectralClustering
    # Create UNSIGNED adjacency matrix for clustering
    A = nx.adjacency_matrix(G, nodelist=node_index, weight='weight')
    A_unsigned = A.copy()
    A_unsigned.data = np.abs(A_unsigned.data)  # Use absolute correlation as affinity
    # Spectral clustering on unsigned adjacency
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_init=100
    )
    labels = sc.fit_predict(A_unsigned)
    # Assign to nodes
    for i, node in enumerate(node_index):
        G.nodes[node]['cluster'] = labels[i]
    return G, labels


def plot_gene_network(G, color_by="category", title='Gene Correlation Network'):
    import matplotlib.patches as mpatches
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    
    # Unified color handling
    if color_by == "category":
        node_attributes = [G.nodes[n].get('category', 'unknown') for n in G.nodes]
    elif color_by == "cluster":
        node_attributes = [G.nodes[n].get('cluster', -1) for n in G.nodes]
    else:
        node_attributes = ["single_group" for _ in G.nodes]
    
    # Auto color mapping
    unique_vals = sorted(set(node_attributes))
    palette = sns.color_palette("husl", len(unique_vals))
    color_map = dict(zip(unique_vals, palette))
    node_colors = [color_map[attr] for attr in node_attributes]
    
    # Visualization
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.8)
    # add gene names on the nodes
    labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', font_family='sans-serif')
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # Optional: Add legend
    legend_handles = []
    for val, color in color_map.items():
        legend_handles.append(mpatches.Patch(color=color, label=val))
    plt.legend(handles=legend_handles)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



# location analysis functions
# Check the location of genes in the database
def check_location_of_genes(gene_list, gene_index=None, database_file="/home/jiguo/data/data/location/subcellular_location_data.tsv", keep_uncertain=True):
    """
    Check the location of genes in the database.
    will return a metrix where rows are genes and columns are unique locations.
    values in the matrix are:
    1: Approved, 2: Supported, 3: Enhanced, 0: Uncertain, NaN: Not found in the database.
    """
    # load the databases which contain the location information from hpa, need to be .tsv file
    if database_file.endswith('.tsv'):
        location_data = pd.read_csv(database_file, sep='\t', low_memory=False, header=0)
    else:
        print("Error: The database file must be a .tsv file.")
        return None
    location_data = location_data[["Reliability", "Approved", "Supported", "Gene name","Enhanced",'Uncertain']]
    if keep_uncertain==False:
        location_data = location_data[location_data['Reliability'] != 'Uncertain']
    # if gene_list is a file, read it using load_gene_list
    if isinstance(gene_list, str):
        gene_list = load_gene_list(gene_list, gene_index=gene_index)
    # if gene_list is a list, convert it to a pandas Series
    if gene_list is not None:
        location_data = location_data[location_data['Gene name'].isin(gene_list)]
    else: # use all the genes in the database
        print("No gene list provided, using all genes in the database.")
        location_data = location_data
    if location_data.empty:
        print("No genes found in the database.")
        return None 
    # columns indeices are all the unique locations from both "Approved", "Supported", "Enhanced", "Uncertain"
    unique_locations = set()
    for index, row in location_data.iterrows():
        # drop nan values in the columns "Approved", "Supported", "Enhanced", "Uncertain"
        if pd.isna(row["Approved"]) or pd.isna(row["Supported"]) or pd.isna(row["Enhanced"]) or pd.isna(row["Uncertain"]):
            # change nan to empty string
            row["Approved"] = "" if pd.isna(row["Approved"]) else row["Approved"]
            row["Supported"] = "" if pd.isna(row["Supported"]) else row["Supported"]
            row["Enhanced"] = "" if pd.isna(row["Enhanced"]) else row["Enhanced"]
            row["Uncertain"] = "" if pd.isna(row["Uncertain"]) else row["Uncertain"]
        # split the locations by ";"
        approved_locations = row["Approved"].split(";")
        supported_locations = row["Supported"].split(";")
        enhanced_locations = row["Enhanced"].split(";")
        uncertain_locations = row["Uncertain"].split(";")
        unique_locations.update(approved_locations)
        unique_locations.update(supported_locations)
        unique_locations.update(enhanced_locations)
        unique_locations.update(uncertain_locations)
    unique_locations = sorted(unique_locations)
    unique_locations.remove("")  # remove empty string if exists
    # Create a Matrix to store the results
    results = pd.DataFrame(columns=["Gene name", "Location"])
    for location in unique_locations:
        results[location] = 0  # initialize all locations to 0
    results['Gene name'] = location_data['Gene name']  # add the gene names to the first column
    for index, row in location_data.iterrows():
        if pd.isna(row["Approved"]) or pd.isna(row["Supported"]) or pd.isna(row["Enhanced"]) or pd.isna(row["Uncertain"]):
            # change nan to empty string
            row["Approved"] = "" if pd.isna(row["Approved"]) else row["Approved"]
            row["Supported"] = "" if pd.isna(row["Supported"]) else row["Supported"]
            row["Enhanced"] = "" if pd.isna(row["Enhanced"]) else row["Enhanced"]
            row["Uncertain"] = "" if pd.isna(row["Uncertain"]) else row["Uncertain"]
        gene_name = row["Gene name"]
        approved_locations = row["Approved"].split(";")
        supported_locations = row["Supported"].split(";")
        enhanced_locations = row["Enhanced"].split(";")
        uncertain_locations = row["Uncertain"].split(";")
        # if the gene location is in the unique locations, set the value to 1
        for location in approved_locations:
            if location in unique_locations:
                results.loc[results['Gene name'] == gene_name, location] = 1
        for location in supported_locations:
            if location in unique_locations:
                results.loc[results['Gene name'] == gene_name, location] = 2
        for location in enhanced_locations:
            if location in unique_locations:
                results.loc[results['Gene name'] == gene_name, location] = 3
            for location in uncertain_locations:
                if location in unique_locations:
                    results.loc[results['Gene name'] == gene_name, location] = 0  
    return results

# Plot the location of genes in the database
def plot_location_of_genes(result_matrix, showReliability=True, xstikrotation=90):
    """
    Plot the location of genes in the database.
    """
    if showReliability:
        # Create separate counts for each reliability level (1=Approved, 2=Supported, 3=Enhanced)
        approved_counts = result_matrix.iloc[:, 1:].apply(lambda x: (x == 1).sum(), axis=0)
        supported_counts = result_matrix.iloc[:, 1:].apply(lambda x: (x == 2).sum(), axis=0)
        enhanced_counts = result_matrix.iloc[:, 1:].apply(lambda x: (x == 3).sum(), axis=0)
        uncertain_counts = result_matrix.iloc[:, 1:].apply(lambda x: (x == 0).sum(), axis=0)
        # Filter to only include localizations that have at least one gene
        locations_with_genes = result_matrix.iloc[:, 1:].apply(lambda x: (x >= 0).sum(), axis=0)
        locations_with_genes = locations_with_genes[locations_with_genes > 0]
        # print how many locations have been excluded and what is it
        print(f"Number of locations with genes: {len(locations_with_genes)}")
        print(f"Number of locations excluded: {len(result_matrix.columns) - len(locations_with_genes)}")
        print(f"Excluded locations: {set(result_matrix.columns) - set(locations_with_genes.index)}")
        # print how many locations have been included
        # Filter counts to only include these locations
        approved_counts = approved_counts[locations_with_genes.index]
        supported_counts = supported_counts[locations_with_genes.index]
        enhanced_counts = enhanced_counts[locations_with_genes.index]
        uncertain_counts = uncertain_counts[locations_with_genes.index]
        # Create stacked bar chart
        plt.figure(figsize=(30, 20))
        width = 0.8
        x_pos = range(0, len(locations_with_genes))
        plt.bar(x_pos, approved_counts, width, label='Approved', color='#8AAAE5', alpha=0.8)
        plt.bar(x_pos, supported_counts, width, bottom=approved_counts, label='Supported', color='#375E97', alpha=0.8)
        plt.bar(x_pos, enhanced_counts, width, bottom=approved_counts + supported_counts, label='Enhanced', color='#FB6542', alpha=0.8)
        plt.bar(x_pos, uncertain_counts, width, bottom=approved_counts + supported_counts + enhanced_counts, label='Uncertain', color='#FFBB00', alpha=0.8)
        # add the total count on top of each bar
        for i, (approved, supported, enhanced, uncertain) in enumerate(zip(approved_counts, supported_counts, enhanced_counts, uncertain_counts)):
            total = approved + supported + enhanced + uncertain
            plt.text(i, total + 0.1, str(total), ha='center', va='bottom',fontsize=16, color='black')
        plt.xticks(x_pos, locations_with_genes.index, rotation=xstikrotation, fontsize=18)
    else:
        localization_counts = result_matrix.iloc[:, 1:].apply(lambda x: (x >= 0).sum(), axis=0)  # Count the number of genes for each localization
        localization_counts = localization_counts[localization_counts > 0]  # Filter out localizations with zero counts
        # plot a bar graph and show the counts of each localization on the bar
        plt.figure(figsize=(30, 20))
        localization_counts.plot(kind='bar', color='skyblue')
        plt.ylim(0, localization_counts.max() + 7)  # Set y-axis limit to be slightly above the max count
        # show the value of each bar on top of the bar
        for index, value in enumerate(localization_counts):
            plt.text(index, value + 0.1, str(value), ha='center', va='bottom')
    plt.title(f"Counts of Protein Localizations", fontsize=24)
    plt.xlabel("Localization", fontsize=14)
    plt.xticks(fontsize=18)
    plt.ylabel("Number of Genes", fontsize=30)
    plt.legend(fontsize=18, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Find proteins localization in the matrix
def find_proteins_localization(location, matrix, showReliability=False, save=False, outputpath=None):
    if location not in matrix.columns:
        print(f"Location '{location}' not found in the matrix.")
        return []
    if showReliability==False:
        localized_genes = matrix[matrix[location] >= 0]['Gene name'].tolist()
        if save:
            # Save the results to a file
            output_file = f"{outputpath}/{location}_localization_results.tsv"
            with open(output_file, 'w') as f:
                f.write("Gene name\n")
                for gene in localized_genes:
                    f.write(f"{gene}\n")
            return localized_genes
        else:
            return localized_genes
    if showReliability==True:
        # Get all genes with any localization value > 0
        localized_rows = matrix[matrix[location] >= 0]
        localized_genes = localized_rows['Gene name'].tolist()
        
        # Get reliability levels for each gene
        reliability_levels = []
        for _, row in localized_rows.iterrows():
            if row[location] == 1:
                reliability_levels.append('approved')
            elif row[location] == 2:
                reliability_levels.append('supported')
            elif row[location] == 3:
                reliability_levels.append('enhanced')
            elif row[location] == 0:
                reliability_levels.append('uncertain')
    if save:
        # Save the results to a file
        output_file = f"{outputpath}/{location}_allGenes_localization_results.tsv"
        with open(output_file, 'w') as f:
            f.write("Gene name\tReliability\n")
            for gene, reliability in zip(localized_genes, reliability_levels):
                f.write(f"{gene}\t{reliability}\n")
        return list(zip(localized_genes, reliability_levels))
    else:
        return list(zip(localized_genes, reliability_levels))

# find_proteins_localization("Centrosome", results, showReliability=False, save=True, outputpath='/home/jiguo/denovo_rpe1_scrnaseq/')