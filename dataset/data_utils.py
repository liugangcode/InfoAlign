from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

import torch
import numpy as np

import copy
import pathlib
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        # atoms
        atom_features_list = []
        # atom_label = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
            # atom_label.append(atom.GetSymbol())

        x = np.array(atom_features_list, dtype=np.int64)
        # atom_label = np.array(atom_label, dtype=np.str)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)

        return graph

    except:
        return None


def read_graph_list(mol_df, keep_id=False):

    mol_list = mol_df["smiles"].tolist()
    ids_list = mol_df["mol_id"].tolist()

    graph_list = []
    total_length = len(mol_list)
    with tqdm(total=total_length, desc="Processing molecules") as pbar:
        for index, smiles_str in enumerate(mol_list):
            graph_dict = smiles2graph(smiles_str)
            if keep_id:
                graph_dict["type"] = ids_list[index]
            graph_list.append(graph_dict)
            pbar.update(1)

    pyg_graph_list = []
    print("Converting graphs into PyG objects...")
    for graph in graph_list:
        g = Data()
        g.__num_nodes__ = graph["num_nodes"]
        g.edge_index = torch.from_numpy(graph["edge_index"])
        del graph["num_nodes"]
        del graph["edge_index"]

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.from_numpy(graph["edge_feat"])
            del graph["edge_feat"]

        if graph["node_feat"] is not None:
            g.x = torch.from_numpy(graph["node_feat"])
            del graph["node_feat"]

        if graph["type"] is not None:
            g.type = graph["type"]
            del graph["type"]

        addition_prop = copy.deepcopy(graph)
        for key in addition_prop.keys():
            g[key] = torch.tensor(graph[key])
            del graph[key]

        pyg_graph_list.append(g)

    return pyg_graph_list


## utils to create prediction context (modality) graph ##

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from collections import defaultdict
from joblib import Parallel, delayed

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
from copy import deepcopy

def get_scaffold(mol):
    """Extracts the Murcko Scaffold from a molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def parallel_scaffold_computation(molecule, molecule_id):
    """Computes scaffold for a single molecule in parallel."""
    scaffold = get_scaffold(molecule)
    return scaffold, molecule, molecule_id


def cluster_molecules_by_scaffold(
    molecules, all_data_id, n_jobs=-1, remove_single=True, flatten_id=True
):
    """Clusters molecules based on their scaffolds using parallel processing."""
    # Ensure molecules and IDs are paired correctly
    paired_results = Parallel(n_jobs=n_jobs)(
        delayed(parallel_scaffold_computation)(mol, molecule_id)
        for mol, molecule_id in zip(molecules, all_data_id)
    )

    # Initialize dictionaries for batches and IDs
    batch = defaultdict(list)
    batched_data_id = defaultdict(list)

    # Process results to fill batch and batched_data_id dictionaries
    for scaffold, mol, molecule_id in paired_results:
        batch[scaffold].append(mol)
        batched_data_id[scaffold].append(molecule_id)

    # Optionally remove clusters with only one molecule
    if remove_single:
        batch = {scaffold: mols for scaffold, mols in batch.items() if len(mols) > 1}
        batched_data_id = {
            scaffold: ids for scaffold, ids in batched_data_id.items() if len(ids) > 1
        }

    # Convert dictionaries to lists for output
    scaffolds = list(batch.keys())
    batch = list(batch.values())
    batched_data_id = list(batched_data_id.values())
    if flatten_id:
        batched_data_id = [idd for batch in batched_data_id for idd in batch]
        batched_data_id = np.array(batched_data_id)

    return scaffolds, batch, batched_data_id


def calculate_mol_similarity(fingerprint1, fingerprint2):
    """Wrapper function to calculate Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fingerprint1, fingerprint2)


def pairwise_mol_similarity(mol_list, n_jobs=1):
    """Calculates the internal similarity within a cluster of molecules using multiple CPUs."""
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mol_list
    ]
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))

    # Define a task for each pair of molecules to calculate similarity
    def compute_similarity(i, j):
        if i < j:  # Avoid redundant calculations and diagonal
            return i, j, calculate_mol_similarity(fingerprints[i], fingerprints[j])
        else:
            return i, j, 0  # No calculation needed, fill with zeros

    # Use Parallel and delayed to compute similarities in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_similarity)(i, j) for i in range(n) for j in range(i, n)
    )

    # Fill the similarity matrix with results
    for i, j, sim in results:
        similarity_matrix[i, j] = similarity_matrix[j, i] = sim

    return similarity_matrix


def perform_pca_and_kmeans(features, all_data_id, n_components=2, n_clusters=100, return_pca_feature=False):
    features = np.nan_to_num(features)
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features)
    # print('explained_variance_ratio_ in perform_pca_and_kmeans', np.cumsum(pca.explained_variance_ratio_))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_features)

    batch = []
    batched_data_id = []
    for i in range(n_clusters):
        if return_pca_feature:
            batch.append(pca_features[kmeans.labels_ == i])
        else:
            batch.append(features[kmeans.labels_ == i])
        batched_data_id.append(all_data_id[kmeans.labels_ == i])
    
    batched_data_id = [idd for batch in batched_data_id for idd in batch]
    batched_data_id = np.array(batched_data_id)

    return batch, batched_data_id


# calculate the similarity between the data points in the same batch and finally merge the similarity matrix
def l1_similarity(matrix):
    matrix = np.nan_to_num(matrix)
    matrix = (matrix - matrix.mean(axis=1)[:, None]) / (matrix.std(axis=1)[:, None] + 1e-10)
    dist = distance.pdist(matrix, "cityblock")  # L1 distance
    dist = distance.squareform(dist)  # Convert to square form
    sim = 1 / (1 + dist)  # Convert distance to similarity
    np.fill_diagonal(sim, 0)  # Set diagonal elements to 0
    return sim


def l2_similarity(matrix):
    matrix = np.nan_to_num(matrix)
    matrix = (matrix - matrix.mean(axis=1)[:, None]) / (matrix.std(axis=1)[:, None] + 1e-10)
    dist = distance.pdist(matrix, "euclidean")  # L2 distance
    dist = distance.squareform(dist)  # Convert to square form
    sim = 1 / (1 + dist)  # Convert distance to similarity
    np.fill_diagonal(sim, 0)  # Set diagonal elements to 0
    return sim


def pairwise_cosine_similarity(matrix):
    matrix = np.nan_to_num(matrix)
    matrix = (matrix - matrix.mean(axis=1)[:, None]) / (matrix.std(axis=1)[:, None] + 1e-10)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0)  # Set diagonal elements to 0
    return sim


def batch_similarity(batches, similarity_func):
    # Calculate the similarity for each batch
    batch_sims = [
        similarity_func(batch)
        for batch in tqdm(batches, desc="Calculating similarities")
    ]

    # Find the total number of nonzero elements
    total_nonzero = sum(np.count_nonzero(arr) for arr in batch_sims)

    # Initialize arrays to store row, col, and data for csr_matrix construction
    rows = np.zeros(total_nonzero, dtype=np.int32)
    cols = np.zeros(total_nonzero, dtype=np.int32)
    data = np.zeros(total_nonzero, dtype=batch_sims[0].dtype)

    current_idx = 0
    current_row = 0
    current_col = 0
    # Loop through each batch similarity array
    for idx, arr in enumerate(batch_sims):
        rows_batch, cols_batch = np.nonzero(arr)
        num_nonzero = len(rows_batch)
        rows[current_idx : current_idx + num_nonzero] = rows_batch + current_row
        cols[current_idx : current_idx + num_nonzero] = cols_batch + current_col
        data[current_idx : current_idx + num_nonzero] = arr[rows_batch, cols_batch]
        current_idx += num_nonzero
        current_row += arr.shape[0]
        current_col += arr.shape[1]

    # Construct the csr_matrix
    merged_sim = csr_matrix(
        (data, (rows, cols)),
        shape=(
            sum(arr.shape[0] for arr in batch_sims),
            sum(arr.shape[1] for arr in batch_sims),
        ),
    )

    return merged_sim


def direct_similarity(matrix, similarity_func):
    sim = similarity_func(matrix)
    np.fill_diagonal(sim, 0)  # Set diagonal elements to 0
    # convert to csr_matrix
    sim = csr_matrix(sim)
    return sim

def determine_threshold(similarity, min_threshold, high_threshold=0.99, target_sparsity=0.995):
    low_threshold = min_threshold
    threshold = (high_threshold + low_threshold) / 2.0  # Start in the middle
    current_sparsity = 1 - np.count_nonzero(similarity > threshold) / similarity.size

    while low_threshold < high_threshold - 0.001:  # Continue until the interval is small
        if current_sparsity > target_sparsity:
            high_threshold = threshold  # Move the upper limit down
        else:
            low_threshold = threshold  # Move the lower limit up

        threshold = (high_threshold + low_threshold) / 2.0  # Recalculate the middle
        current_sparsity = 1 - np.count_nonzero(similarity > threshold) / similarity.size

    return threshold

def filter_similarity_and_get_ids(similarity_matrix, threshold, data_id):
    filtered_similarity = similarity_matrix.copy()
    filtered_similarity.data[filtered_similarity.data < threshold] = 0
    filtered_similarity.eliminate_zeros()

    row, col = filtered_similarity.nonzero()

    s_node_id = data_id[row]
    t_node_id = data_id[col]
    edge_weight = filtered_similarity.data
    edge_weight = np.round(edge_weight, 2)

    return s_node_id, t_node_id, edge_weight

def merge_features_and_dataframes(df1, df2, features1, features2, col_id1, col_id2, connect_col=None):
    if connect_col is None:
        df1['connect_id'] = df1[col_id1].astype(str) + '_' + df1[col_id2].astype(str)
        df2['connect_id'] = df2[col_id1].astype(str) + '_' + df2[col_id2].astype(str)
    else:
        df1['connect_id'] = df1[connect_col].astype(str)
        df2['connect_id'] = df2[connect_col].astype(str)
    index1 = f'index_{col_id1}'
    index2 = f'index_{col_id2}'
    df1[index1] = df1.index
    df2[index2] = df2.index
    
    merged_df = pd.merge(df1[['connect_id', index1, col_id1]], df2[['connect_id', index2, col_id2]], on='connect_id', how='outer')
    if connect_col is None:
        merged_df[[col_id1, col_id2]] = merged_df['connect_id'].str.split('_', expand=True)
    merged_features = []
    # Iterate over each row in the merged dataframe to concatenate features
    for _, row in merged_df.iterrows():
        feature1_row = np.nan * np.ones(features1.shape[1]) if np.isnan(row[index1]) else features1[int(row[index1])]
        feature2_row = np.nan * np.ones(features2.shape[1]) if np.isnan(row[index2]) else features2[int(row[index2])]
        merged_features.append(np.concatenate([feature1_row, feature2_row]))

    merged_features = np.vstack(merged_features)
    return merged_df, merged_features

def minmax_normalize(data, min_val=None, max_val=None):
    """
    Normalizes the data using min-max normalization, handling NaN values in min and max calculations.
    
    Parameters:
    - data: The data to be normalized (numpy array).
    - min_val: Optional. Precomputed minimum values for each feature. If not provided, computed from data.
    - max_val: Optional. Precomputed maximum values for each feature. If not provided, computed from data.
    
    Returns:
    - Normalized data, used min_val, used max_val.
    """
    # Calculate min and max values if not provided, ignoring NaN values
    if min_val is None or max_val is None:
        min_val = np.nanmin(data, axis=0)
        max_val = np.nanmax(data, axis=0)
    else:
        assert min_val.shape[0] == data.shape[1], "min_val must match the number of features in data"
        assert max_val.shape[0] == data.shape[1], "max_val must match the number of features in data"

    # Handle cases where min and max are equal
    equal_indices = min_val == max_val
    min_val[equal_indices] = min_val[equal_indices] - 1e-6
    max_val[equal_indices] = max_val[equal_indices] + 1e-6
    
    # Normalize, handling divisions by zero or where max_val equals min_val
    normalized_data = np.where(
        max_val - min_val == 0,
        0,
        (data - min_val) / (max_val - min_val)
    )

    return normalized_data, min_val, max_val

def create_nx_graph(folder, min_thres=0.6, min_sparsity=0.995, top_compound_gene_express=0.05):

    structure_df = pd.read_csv(f"{folder}/structure.csv.gz")
    mol_df = structure_df.drop_duplicates(subset="mol_id")
    smiles_list = mol_df["smiles"].tolist()
    all_mol_id = mol_df["mol_id"].tolist()
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    structure_feature = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(m, 4, nBits=1024) for m in molecules]
    )
    # load cell nodes and features
    cp_bray_df = pd.read_csv(f"{folder}/CP-Bray.csv.gz")
    cp_jump_df = pd.read_csv(f"{folder}/CP-JUMP.csv.gz")
    cp_bray_feature = np.load(f"{folder}/CP-Bray_feature.npz")["data"]
    cp_jump_feature = np.load(f"{folder}/CP-JUMP_feature.npz")["data"]
    bray_dim, jump_dim = cp_bray_feature.shape[1], cp_jump_feature.shape[1]

    cell_df, cell_feature = merge_features_and_dataframes(
        df1=cp_bray_df, df2=cp_jump_df,
        features1=cp_bray_feature, features2=cp_jump_feature,
        col_id1='cell_bid', col_id2='cell_jid',
        connect_col='mol_id'
    )
    cell_df = cell_df.rename(columns={'connect_id': 'mol_id'})
    cell_df['cell_id'] = ['c' + str(i) for i in range(1, len(cell_df) + 1)]
    cell_df = cell_df[['cell_id', 'mol_id', 'index_cell_bid', 'index_cell_jid']]

    # load gene nodes and features
    gc_df = pd.read_csv(f"{folder}/G-CRISPR.csv.gz")
    gc_feature = np.load(f"{folder}/G-CRISPR_feature.npz")["data"]
    go_df = pd.read_csv(f"{folder}/G-ORF.csv.gz")
    go_feature = np.load(f"{folder}/G-ORF_feature.npz")["data"]

    gene_df, gene_feature = merge_features_and_dataframes(
        df1=gc_df, df2=go_df,
        features1=gc_feature, features2=go_feature,
        col_id1='ncbi_gene_id', col_id2='mol_id',
        connect_col=None
    )
    gene_df['gene_id'] = gene_df['ncbi_gene_id'].apply(lambda x: 'g' + str(x))
    gene_df = gene_df[['gene_id', 'mol_id', 'index_ncbi_gene_id', 'index_mol_id']]

    # load gene expression
    express_df = pd.read_csv(f"{folder}/GE.csv.gz")
    express_feature = np.load(f"{folder}/GE_feature.npz")["data"]

    # load gene-gene interaction
    gg_df = pd.read_csv(f"{folder}/G-G.csv.gz")

    nid_to_feature_id = {}
    # Combine the loop iterations for different dataframes into a single loop
    for df, col_id in [
        (mol_df, "mol_id"),
        (express_df, "express_id"),
        (gene_df, "gene_id"),
        (cell_df, "cell_id"),
    ]:
        for i, nid in enumerate(df[col_id]):
            nid_to_feature_id[nid] = i
    
    target_mol = structure_feature
    target_gene, min_gene, max_gene = minmax_normalize(gene_feature)
    target_cell, min_cell, max_cell = minmax_normalize(cell_feature)
    target_express, min_express, max_express = minmax_normalize(express_feature)

    mol_dim, gene_dim, cell_dim, express_dim = (
        structure_feature.shape[1],
        gene_feature.shape[1],
        cell_feature.shape[1],
        express_feature.shape[1],
    )

    #### molecular similarity
    scaffold_names, batch_mol_feature, batched_mol_id = cluster_molecules_by_scaffold(
        molecules, all_mol_id
    )
    sim_mol = batch_similarity(batch_mol_feature, pairwise_mol_similarity)

    #### gene similarity
    batched_gene_feature, batched_gene_id = perform_pca_and_kmeans(
        gene_feature, gene_df["gene_id"].values
    )
    sim_gene = batch_similarity(batched_gene_feature, pairwise_cosine_similarity)

    #### cell similarity
    batched_cell_feature, batched_cell_id = perform_pca_and_kmeans(
        cell_feature[:, bray_dim:], cell_df["cell_id"].values
    )
    sim_cell = batch_similarity(batched_cell_feature, pairwise_cosine_similarity)

    #### gene expression similarity
    sim_express = direct_similarity(express_feature, pairwise_cosine_similarity)
    direct_express_id = express_df["express_id"].values

    ########## create graph ########
    G = nx.Graph()

    mol_ids = structure_df["mol_id"].values
    gene_ids = gene_df["gene_id"].values
    cell_ids = cell_df["cell_id"].values
    express_ids = express_df["express_id"].values
    
    # Add molecule nodes
    mol_nodes = [
        (
            mol_ids[idx],
            dict(
                type=mol_ids[idx],
                mol_target=target_mol[nid_to_feature_id[mol_id]],
                gene_target=np.full(gene_dim, np.nan),
                cell_target=np.full(cell_dim, np.nan),
                express_target=np.full(express_dim, np.nan)
            ),
        )
        for idx, mol_id in enumerate(mol_ids)
    ]
    G.add_nodes_from(mol_nodes)
    print("Count nodes after adding molecule nodes:", G.number_of_nodes())

    # Add gene crispr nodes
    gene_nodes = [
        (
            gene_ids[idx],
            dict(
                type=gene_ids[idx],
                mol_target=np.full(mol_dim, np.nan),
                gene_target=target_gene[nid_to_feature_id[gene_id]],
                cell_target=np.full(cell_dim, np.nan),
                express_target=np.full(express_dim, np.nan),
            ),
        )
        for idx, gene_id in enumerate(gene_ids)
    ]
    G.add_nodes_from(gene_nodes)
    print("Count nodes after adding gene nodes:", G.number_of_nodes())

    cell_nodes = [
        (
            cell_ids[idx],
            dict(
                type=cell_ids[idx],
                mol_target=np.full(mol_dim, np.nan),
                gene_target=np.full(gene_dim, np.nan),
                cell_target=target_cell[nid_to_feature_id[cell_id]],
                express_target=np.full(express_dim, np.nan),
            ),
        )
        for idx, cell_id in enumerate(cell_ids)
    ]
    G.add_nodes_from(cell_nodes)
    print("Count nodes after adding cell nodes:", G.number_of_nodes())

    # Add gene expression nodes
    express_nodes = [
        (
            express_ids[idx],
            dict(
                type=express_ids[idx],
                mol_target=np.full(mol_dim, np.nan),
                gene_target=np.full(gene_dim, np.nan),
                cell_target=np.full(cell_dim, np.nan),
                express_target=target_express[nid_to_feature_id[express_id]],

            ),
        )
        for idx, express_id in enumerate(express_ids)
    ]
    G.add_nodes_from(express_nodes)
    print("Count nodes after adding gene expression nodes:", G.number_of_nodes())

    G.add_edges_from(zip(gene_df["gene_id"], gene_df["mol_id"]), weight=1)
    print("Count of edges after adding mol-gene:", G.number_of_edges())
    G.add_edges_from(zip(cell_df["mol_id"], cell_df["cell_id"]), weight=1)
    print("Count of edges after adding mol-cell:", G.number_of_edges())
    G.add_edges_from(zip(express_df["express_id"], express_df["mol_id"]), weight=1)
    print("Count of edges after adding mol-express:", G.number_of_edges())

    # add gene-gene edges
    gene_source_prefixed = "g" + gg_df["source_id"].astype(str)
    gene_target_prefixed = "g" + gg_df["target_id"].astype(str)

    ## Filter the 'go' prefixed edges where both nodes exist in go_df["orf_id"]
    valid_gene_sources = gene_source_prefixed.isin(gene_df["gene_id"])
    valid_gene_targets = gene_target_prefixed.isin(gene_df["gene_id"])
    gene_edges_to_add = zip(
        gene_source_prefixed[valid_gene_sources & valid_gene_targets],
        gene_target_prefixed[valid_gene_sources & valid_gene_targets],
    )
    G.add_edges_from(gene_edges_to_add, weight=1)
    print("Count of edges after adding gene-gene in the graph:", G.number_of_edges())

    # Add edges between cell according to the similarity matrix
    def add_edges_with_weight(G, s_node, t_node, edge_weight):
        # G.add_edges_from(zip(s_node, t_node), weight=edge_weight)
        for i, (s, t) in enumerate(zip(s_node, t_node)):
            weight = edge_weight[i]
            G.add_edge(s, t, weight=weight)

    L1k_idmaps = pd.read_csv(f'{folder}/L1k_idmaps.csv')
    sorted_indices = np.argsort(np.abs(express_feature).flatten())
    start_index = int(len(sorted_indices) * (1-top_compound_gene_express))
    express_thre = np.abs(express_feature).flatten()[sorted_indices[start_index]]
    top_percent_indices = sorted_indices[start_index:]
    row_indices, col_indices = np.unravel_index(top_percent_indices, express_feature.shape)
    if len(row_indices) < 1000:
        top_percent_indices = sorted_indices[-1000:]
        row_indices, col_indices = np.unravel_index(top_percent_indices, express_feature.shape)

    expressed_mol = express_df['mol_id'].iloc[row_indices].values
    expressed_gene = L1k_idmaps['ncbi_gene_id'].iloc[col_indices].values
    expressed_gene = np.array([f'g{int(gid)}' for gid in expressed_gene])
    expressed_weight = np.abs(express_feature[row_indices, col_indices])
    expressed_gene_series = pd.Series(expressed_gene)
    valid_express_gene = expressed_gene_series.isin(gene_df["gene_id"]).values
    add_edges_with_weight(G, expressed_mol[valid_express_gene], expressed_gene[valid_express_gene], expressed_weight[valid_express_gene])
    print(f"Count of edges after adding top {round(top_compound_gene_express * 100, 2)} % gene-gene from expression with threshold {round(express_thre, 4)}:", G.number_of_edges())

    thre_cell = determine_threshold(sim_cell.data, min_thres, target_sparsity=min_sparsity)
    thre_gene = determine_threshold(sim_gene.data, min_thres, target_sparsity=min_sparsity)
    thre_express = determine_threshold(sim_express.data, min_thres, target_sparsity=min_sparsity)
    thre_mol = determine_threshold(sim_mol.data, min_thres, target_sparsity=min_sparsity)
    thres_dict = {'cell': thre_cell, 'gene': thre_gene, 'express': thre_express, 'mol': thre_mol}

    ## filter similarity and get ids
    mol_s_node, mol_t_node, mol_edge_weight = filter_similarity_and_get_ids(
        sim_mol, thres_dict['mol'], batched_mol_id
    )
    gene_s_node, gene_t_node, gene_edge_weight = filter_similarity_and_get_ids(
        sim_gene, thres_dict['gene'], batched_gene_id
    )
    cell_s_node, cell_t_node, cell_edge_weight = filter_similarity_and_get_ids(
        sim_cell, thres_dict['cell'], batched_cell_id
    )
    express_s_node, express_t_node, express_edge_weight = filter_similarity_and_get_ids(
        sim_express, thres_dict['express'], direct_express_id
    )
    add_edges_with_weight(G, mol_s_node, mol_t_node, mol_edge_weight)
    print(f"Count of edges after adding mol-mol sim with threshold {round(thres_dict['mol'], 2)}:", G.number_of_edges())

    add_edges_with_weight(G, gene_s_node, gene_t_node, gene_edge_weight)
    print(f"Count of edges after adding gene-gene sim with threshold {round(thres_dict['gene'], 2)}:", G.number_of_edges())

    add_edges_with_weight(G, cell_s_node, cell_t_node, cell_edge_weight)
    print(f"Count of edges after adding cell-cell sim with threshold {round(thres_dict['cell'], 2)}:", G.number_of_edges())

    add_edges_with_weight(G, express_s_node, express_t_node, express_edge_weight)
    print(f"Count of edges after adding express-express sim with threshold {round(thres_dict['express'], 2)}:", G.number_of_edges())

    nan_nodes = [n for n in G.nodes() if pd.isnull(n)]
    G.remove_nodes_from(nan_nodes)

    gene_bound = {"min": min_gene, "max": max_gene}
    cell_bound = {"min": min_cell, "max": max_cell}
    express_bound = {"min": min_express, "max": max_express}

    # add global information for ge_bound
    G.graph["gene_bound"] = gene_bound
    G.graph["cell_bound"] = cell_bound
    G.graph["express_bound"] = express_bound

    return G


import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.data import Data


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], Literal["all"]]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal["all"]]] = None,
) -> "torch_geometric.data.Data":
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or "all", optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:
        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Data` object is returned
        >>> from_networkx(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data_dict: Dict[str, Any] = defaultdict(list)
    data_dict["edge_index"] = edge_index

    node_attrs: List[str] = []
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

    edge_attrs: List[str] = []
    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())

    if group_node_attrs is not None and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs

    if group_edge_attrs is not None and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            print('i', i)
            print('feat_dict', feat_dict)
            print('node_attrs', node_attrs)
            raise ValueError("Not all nodes contain the same attributes")
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError("Not all edges contain the same attributes")
        for key, value in feat_dict.items():
            key = f"edge_{key}" if key in node_attrs else key
            data_dict[str(key)].append(value)

    for key, value in G.graph.items():
        if key == "node_default" or key == "edge_default":
            continue  # Do not load default attributes.
        key = f"graph_{key}" if key in node_attrs else key
        data_dict[str(key)] = value

    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data_dict[key] = torch.stack(value, dim=0, dtype=torch.float32)
        elif isinstance(value, (tuple, list)) and isinstance(value[0], np.ndarray):
            data_dict[key] = torch.tensor(np.stack(value), dtype=torch.float32)
        else:
            try:
                data_dict[key] = torch.as_tensor(np.array(value))
            except Exception:
                pass

    data = Data.from_dict(data_dict)

    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f"edge_{key}" if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data



##### scaffold splitting

def scaffold_split(train_df, train_ratio=0.6, valid_ratio=0.15, test_ratio=0.25):
    """Splits a dataframe of molecules into scaffold-based clusters."""
    # Get smiles from the dataframe
    train_smiles_list = train_df["smiles"]

    indinces = list(range(len(train_smiles_list)))
    train_mol_list = [Chem.MolFromSmiles(smiles) for smiles in train_smiles_list]
    scaffold_names, _, batched_id = cluster_molecules_by_scaffold(train_mol_list, indinces, remove_single=False, flatten_id=False)

    train_cutoff = int(train_ratio * len(train_df))
    valid_cutoff = int(valid_ratio * len(train_df)) + train_cutoff
    train_inds, valid_inds, test_inds = [], [], []
    inds_all = deepcopy(batched_id)
    np.random.seed(3)
    np.random.shuffle(inds_all)
    idx_count = 0
    for inds_list in inds_all:
        for ind in inds_list:
            if idx_count < train_cutoff:
                train_inds.append(ind)
            elif idx_count < valid_cutoff:
                valid_inds.append(ind)
            else:
                test_inds.append(ind)
            idx_count += 1

    return train_inds, valid_inds, test_inds