import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from rdkit import Chem
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def load_data_gcn(dataset, feature_map):
    """
    :param dataset: 数据集路径
    :param feature_map: one-hot特征映射
    :return: adjs 矩阵的列表
             features 矩阵的列表
             y ndarry (None, 1)
    """
    df = pd.read_csv(dataset)

    # y
    y = np.array(df['Tg']).reshape(-1, 1)

    # adj for batchsize=1, [max x max, ...]
    adjs = []
    max_atom_num = 0
    for i in range(df.shape[0]):
        adj_dict = {}
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        atoms = mol.GetAtoms()
        if len(atoms) > max_atom_num:
            max_atom_num = len(atoms)
        for atom in atoms:
            adj_dict[atom.GetIdx()] = []
            for neighbor in atom.GetNeighbors():
                adj_dict[atom.GetIdx()].append(neighbor.GetIdx())
        adjs.append(nx.adjacency_matrix(nx.from_dict_of_lists(adj_dict)))

    # zero_padding至相同大小
    print('最大原子数：%d，邻接矩阵zero_padding...' % max_atom_num)
    for i in tqdm(range(len(adjs))):
        width = max_atom_num - adjs[i].shape[0]
        if width % 2 == 0:
            width = int(width/2)
            adjs[i] = sp.csr_matrix(np.pad(adjs[i].todense(), ((width, width), (width, width)), 'constant'))
        else:
            width = int(width/2)
            adjs[i] = sp.csr_matrix(np.pad(adjs[i].todense(), ((width, width+1), (width, width+1)), 'constant'))
    """
    batch-wise没做
    """

    # features
    features = []
    for i in tqdm(range(df.shape[0])):
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        feature = np.zeros((max_atom_num, len(feature_map)))
        width = int((max_atom_num - len(mol.GetAtoms()))/2)
        for atom in mol.GetAtoms():
            res = atom.GetSymbol()                      # 原子类别
            feature[atom.GetIdx()+width, feature_map[res]] = 1

            res = atom.GetTotalNumHs()                  # 连接H原子数
            feature[atom.GetIdx()+width, feature_map['H%d' % res]] = 1

            res = atom.GetDegree()                      # Degree
            feature[atom.GetIdx()+width, feature_map['D%d' % res]] = 1

            res = atom.GetIsAromatic()                  # 芳香性
            feature[atom.GetIdx()+width, feature_map['A%d' % res]] = 1

            res = atom.IsInRing()                       # 是否在环上
            feature[atom.GetIdx()+width, feature_map['R%d' % res]] = 1
        features.append(sp.csr_matrix(feature))
    return adjs, features, y


def encode_one_hot(df, feature):
    feature_list = list(set(df[feature].values))

    arr = np.eye(len(feature_list))
    feature_index = {}
    for i, elem in enumerate(feature_list):
        feature_index[elem] = i

    feature_ont_hot = []
    for i in range(df.shape[0]):
        feature_ont_hot.append(arr[feature_index[df.loc[i, feature]]])

    return np.array(feature_ont_hot)


def load_data(dataset, feature_map, feature_name):
    adjs, features, y = load_data_gcn(dataset, feature_map)

    df = pd.read_csv(dataset)

    # list of dicts
    discrete_features = {}
    continuous_features = []

    for name in feature_name[0]:
        discrete_features[name] = encode_one_hot(df, name)
    for name in feature_name[1]:
        continuous_features.append(np.array(list(df[name].values)))
    continuous_features = np.stack(continuous_features, axis=1)

    return adjs, features, y, discrete_features, continuous_features


def train_val_split_gcn(supports, features, y, val_ratio, test_ratio):
    val_num = int(len(supports)*(1-val_ratio-test_ratio))
    test_num = int(len(supports)*(1-test_ratio))
    return supports[:val_num], features[:val_num], y[:val_num, :],\
           supports[val_num:test_num], features[val_num:test_num], y[val_num:test_num, :]

def test_split_gcn(supports, features, y, test_ratio):
    test_num = int(len(supports)*(1-test_ratio))
    return supports[test_num:], features[test_num:], y[test_num:, :]



def train_test_split_mlp(discrete_features, continuous_features, val_ratio):
    num = int(continuous_features.shape[0]*(1-val_ratio))
    discrete_features_train = {}
    discrete_features_val = {}
    for key, value in discrete_features.items():
        discrete_features_train[key] = value[:num]
        discrete_features_val[key] = value[num:]
    return discrete_features_train, discrete_features_val, \
        continuous_features[:num], continuous_features[num:]


def my_construct_feed_dict(gcn_feature, support, y, con_feature, dis_feature, index, placeholders):
    feed_dict = {}
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['features']: gcn_feature})
    feed_dict.update({placeholders['labels']: y})
    feed_dict.update({placeholders['con_features']: con_feature})
    feed_dict.update({placeholders['num_features_nonzero']: gcn_feature[1].shape})
    for key, value in dis_feature.items():
        feed_dict.update({placeholders[key]: value[index].reshape((1, -1))})
    return feed_dict