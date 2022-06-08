import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from rdkit import Chem
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)                        # 格式转换
    rowsum = np.array(adj.sum(1))                   # 计算行方向上的和（度）
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()   # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.           # 无穷处改为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)           # 对角矩阵
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()          # D^-0.5 A D^-0.5


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        """坐标，值，大小"""
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_data_gcn(dataset, node_feature_map, max_atoms, edge_feature_map):
    """
    :param dataset: 数据集路径
    :param feature_map: one-hot特征映射
    :return: adjs 矩阵的列表
             features 矩阵的列表
             y ndarry (None, 1)
    """
    df = pd.read_csv(dataset)

    """
    y
    """
    y = np.array(df['Tg']).reshape(-1, 1)

    """
    adj
    """
    adjs = []                                                       # 下标i，对应df第i行数据
    for i in range(df.shape[0]):
        adj_dict = {}
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        atoms = mol.GetAtoms()
        for atom in atoms:
            if len(atoms) <= max_atoms:
                adj_dict[atom.GetIdx()] = []
                for neighbor in atom.GetNeighbors():
                    adj_dict[atom.GetIdx()].append(neighbor.GetIdx())
            else:
                begin = int((len(atoms)-max_atoms)/2)               # [begin, end]
                end = begin+max_atoms-1
                if begin <= atom.GetIdx() <= end:                     # 节点与其邻节点均在范围内
                    adj_dict[atom.GetIdx()-begin] = []
                    for neighbor in atom.GetNeighbors():
                        if begin <= neighbor.GetIdx() <= end:
                            adj_dict[atom.GetIdx()-begin].append(neighbor.GetIdx()-begin)
        adjs.append(nx.adjacency_matrix(nx.from_dict_of_lists(adj_dict)))

    # # zero_padding至相同大小
    # print('最大原子数：%d，邻接矩阵zero_padding...' % max_atoms)
    # for i in tqdm(range(len(adjs))):
    #     mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
    #     atoms = mol.GetAtoms()
    #     width = max_atoms - len(atoms)
    #     if width > 0 and width % 2 == 0:
    #         width = int(width/2)
    #         adjs[i] = sp.csr_matrix(np.pad(adjs[i].todense(), ((width, width), (width, width)), 'constant'))
    #     elif width > 0:
    #         width = int(width/2)
    #         adjs[i] = sp.csr_matrix(np.pad(adjs[i].todense(), ((width, width+1), (width, width+1)), 'constant'))

    # features
    node_features = []
    for i in tqdm(range(df.shape[0])):
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        feature = np.zeros((max_atoms, len(node_feature_map)))
        if max_atoms >= len(mol.GetAtoms()):
            width = int((max_atoms - len(mol.GetAtoms()))/2)
            begin = 0
            end = len(mol.GetAtoms())-1
        else:
            begin = int((len(mol.GetAtoms())-max_atoms)/2)
            end = begin+max_atoms-1
            width = -begin
        for atom in mol.GetAtoms():
            if width > 0 or begin <= atom.GetIdx() <= end:
                res = atom.GetSymbol()                      # 原子类别
                feature[atom.GetIdx()+width, node_feature_map[res]] = 1

                res = atom.GetTotalNumHs()                  # 连接H原子数
                feature[atom.GetIdx()+width, node_feature_map['H%d' % res]] = 1

                res = atom.GetDegree()                      # Degree
                feature[atom.GetIdx()+width, node_feature_map['D%d' % res]] = 1

                res = atom.GetIsAromatic()                  # 芳香性
                feature[atom.GetIdx()+width, node_feature_map['A%d' % res]] = 1

                res = atom.IsInRing()                       # 是否在环上
                feature[atom.GetIdx()+width, node_feature_map['R%d' % res]] = 1
        node_features.append(sp.csr_matrix(feature))

    edge_features = []
    for i in tqdm(range(df.shape[0])):
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        feature = np.zeros((max_atoms, max_atoms, 1, len(edge_feature_map)))
        num = 1/len(edge_feature_map)
        if max_atoms >= len(mol.GetAtoms()):
            width = int((max_atoms - len(mol.GetAtoms()))/2)
            begin = 0
            end = len(mol.GetAtoms())-1
        else:
            begin = int((len(mol.GetAtoms())-max_atoms)/2)
            end = begin+max_atoms-1
            width = -begin
        for bond in mol.GetBonds():
            if width > 0 or \
                    (begin <= bond.GetBeginAtomIdx() <= end and begin <= bond.GetEndAtomIdx() <= end):
                res = bond.GetBondTypeAsDouble()
                beginIdx = bond.GetBeginAtomIdx()+width
                endIdx = bond.GetEndAtomIdx()+width
                feature[beginIdx, endIdx, 0, edge_feature_map['T'+str(res)]] = num
                feature[endIdx, beginIdx, 0, edge_feature_map['T'+str(res)]] = num

                res = bond.IsInRing()
                feature[beginIdx, endIdx, 0, edge_feature_map['R%d' % res]] = num
                feature[endIdx, beginIdx, 0, edge_feature_map['R%d' % res]] = num

                res = bond.GetIsConjugated()
                feature[beginIdx, endIdx, 0, edge_feature_map['C%d' % res]] = num
                feature[endIdx, beginIdx, 0, edge_feature_map['C%d' % res]] = num
        edge_features.append(feature)

    return adjs, node_features, y, edge_features


def encode_one_hot(df, feature):
    df_all = pd.read_csv('./dataset/dataForMMGCN.csv')
    feature_list = list(set(df_all[feature].values))

    arr = np.eye(len(feature_list))
    feature_index = {}
    for i, elem in enumerate(feature_list):
        feature_index[elem] = i

    feature_ont_hot = []
    for i in range(df.shape[0]):
        feature_ont_hot.append(arr[feature_index[df.loc[i, feature]]])

    return np.array(feature_ont_hot)


def load_data(dataset, node_feature_map, feature_name, max_atoms, edge_feature_map):
    adjs, node_features, y, edge_features = load_data_gcn(dataset, node_feature_map, max_atoms, edge_feature_map)

    df = pd.read_csv(dataset)

    # list of dicts
    discrete_features = {}
    continuous_features = []

    for name in feature_name['discrete']:
        discrete_features[name] = encode_one_hot(df, name)
    for name in feature_name['continuous']:
        continuous_features.append(np.array(list(df[name].values)))
    continuous_features = np.stack(continuous_features, axis=1)

    return adjs, node_features, y, edge_features, discrete_features, continuous_features


def preprocess_cfeatures(continuous_features):
    coldiff = continuous_features.max(0) - continuous_features.min(0)
    continuous_features = (continuous_features - continuous_features.min(0))/coldiff
    return continuous_features


def my_chebyshev_polynomials(adj, k, max_atoms):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    adj_normalized = normalize_adj(adj)                     # D^-0.5 A D^-0.5
    laplacian = sp.eye(adj.shape[0]) - adj_normalized       # I - D^-0.5 A D^-0.5
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')     # 求最大特征值
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])  # 2L/max - I

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    for i in range(len(t_k)):
        width = max_atoms - adj.shape[0]
        if width > 0 and width % 2 == 0:
            width = int(width/2)
            t_k[i] = sp.csr_matrix(np.pad(t_k[i].todense(), ((width, width), (width, width)), 'constant'))
        elif width > 0:
            width = int(width/2)
            t_k[i] = sp.csr_matrix(np.pad(t_k[i].todense(), ((width, width+1), (width, width+1)), 'constant'))

    return sparse_to_tuple(t_k)


def train_val_split_gcn(supports, node_features, edge_features, y, val_ratio, test_ratio):
    val_num = int(len(supports)*(1-val_ratio-test_ratio))
    test_num = int(len(supports)*(1-test_ratio))
    return supports[:val_num], node_features[:val_num], edge_features[:val_num], y[:val_num, :],\
           supports[val_num:test_num], node_features[val_num:test_num], edge_features[val_num:test_num],\
           y[val_num:test_num, :]


def test_split_gcn(supports, features, edge_features, y, test_ratio):
    test_num = int(len(supports)*(1-test_ratio))
    return supports[test_num:], features[test_num:], edge_features[test_num:], y[test_num:, :]


def train_val_split_mmgcn(discrete_features, continuous_features, val_ratio, test_ratio):
    val_num = int(continuous_features.shape[0]*(1-val_ratio-test_ratio))
    test_num = int(continuous_features.shape[0]*(1-test_ratio))
    discrete_features_train = {}
    discrete_features_val = {}
    for key, value in discrete_features.items():
        discrete_features_train[key] = value[:val_num]
        discrete_features_val[key] = value[val_num:test_num]
    return discrete_features_train, discrete_features_val, \
        continuous_features[:val_num], continuous_features[val_num:test_num]


def test_split_mmgcn(discrete_features, continuous_features, test_ratio):
    test_num = int(continuous_features.shape[0]*(1-test_ratio))
    discrete_features_test = {}
    for key, value in discrete_features.items():
        discrete_features_test[key] = value[test_num:]
    return discrete_features_test, continuous_features[test_num:]


def my_construct_feed_dict(gcn_feature, support, y, edge_features, con_feature, batch, D_feature_name, placeholders):
    feed_dict = {}
    # feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['features'][i]: gcn_feature[i] for i in range(len(gcn_feature))})
    feed_dict.update({placeholders['labels'][i]: y[i] for i in range(len(y))})
    feed_dict.update({placeholders['edge_features'][i]: edge_features[i] for i in range(len(edge_features))})
    feed_dict.update({placeholders['con_features'][i]: con_feature[i] for i in range(len(con_feature))})
    feed_dict.update({placeholders['num_features_nonzero'][i]: gcn_feature[i][1].shape
                      for i in range(len(gcn_feature))})

    for name in D_feature_name:
        feed_dict.update({placeholders[name][i]: batch[name][i] for i in range(len(batch[name]))})

    support_dict = {}
    for i in range(len(support)):
        for j in range(len(support[i])):
            support_dict[placeholders['support'][i][j]] = support[i][j]
    feed_dict.update(support_dict)
    return feed_dict