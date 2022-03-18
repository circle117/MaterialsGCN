import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from rdkit import Chem
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def load_data(dataset, feature_map, val_ratio):
    """
    :param dataset: 数据集路径
    :param feature_map: one-hot特征映射
    :return: adjs 矩阵的列表
             features 矩阵的列表
             y ndarry (None, 1)
    """
    df = pd.read_csv(dataset)
    df = df.sample(frac=1).reset_index(drop=True)           # shuffle

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
    batch-wise
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


def train_test_split(supports, features, y, val_ratio):
    val_num = int(len(supports)*(1-val_ratio))
    return supports[:val_num], features[:val_num], y[:val_num, :],\
           supports[val_num:], features[val_num:], y[val_num:, :]
