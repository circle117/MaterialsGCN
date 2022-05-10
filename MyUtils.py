import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from rdkit import Chem
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def load_data_gcn(dataset, feature_map, max_atoms):
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

    # zero_padding至相同大小
    print('最大原子数：%d，邻接矩阵zero_padding...' % max_atoms)
    for i in tqdm(range(len(adjs))):
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        atoms = mol.GetAtoms()
        width = max_atoms - len(atoms)
        if width > 0 and width % 2 == 0:
            width = int(width/2)
            adjs[i] = sp.csr_matrix(np.pad(adjs[i].todense(), ((width, width), (width, width)), 'constant'))
        elif width > 0:
            width = int(width/2)
            adjs[i] = sp.csr_matrix(np.pad(adjs[i].todense(), ((width, width+1), (width, width+1)), 'constant'))

    # features
    features = []
    for i in tqdm(range(df.shape[0])):
        mol = Chem.MolFromSmiles(df.loc[i, 'SMILES'])
        feature = np.zeros((max_atoms, len(feature_map)))
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


def load_data(dataset, feature_map, feature_name):
    adjs, features, y = load_data_gcn(dataset, feature_map)

    df = pd.read_csv(dataset)

    # list of dicts
    discrete_features = {}
    continuous_features = []

    for name in feature_name['discrete']:
        discrete_features[name] = encode_one_hot(df, name)
    for name in feature_name['continuous']:
        continuous_features.append(np.array(list(df[name].values)))
    continuous_features = np.stack(continuous_features, axis=1)

    return adjs, features, y, discrete_features, continuous_features


def preprocess_cfeatures(continuous_features):
    coldiff = continuous_features.max(0) - continuous_features.min(0)
    continuous_features = (continuous_features - continuous_features.min(0))/coldiff
    return continuous_features


def train_val_split_gcn(supports, features, y, val_ratio, test_ratio):
    val_num = int(len(supports)*(1-val_ratio-test_ratio))
    test_num = int(len(supports)*(1-test_ratio))
    return supports[:val_num], features[:val_num], y[:val_num, :],\
           supports[val_num:test_num], features[val_num:test_num], y[val_num:test_num, :]


def test_split_gcn(supports, features, y, test_ratio):
    test_num = int(len(supports)*(1-test_ratio))
    return supports[test_num:], features[test_num:], y[test_num:, :]


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


def my_construct_feed_dict(gcn_feature, support, y, con_feature, batch, D_feature_name, placeholders):
    feed_dict = {}
    # feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['features'][i]: gcn_feature[i] for i in range(len(gcn_feature))})
    feed_dict.update({placeholders['labels'][i]: y[i] for i in range(len(y))})
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