import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN
import numpy as np

tf.disable_v2_behavior()

NODE_FEATURE_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si', 'P',       # 原子类别
                     'H0', 'H1', 'H2', 'H3',                        # 连接H数量
                     'D1', 'D2', 'D3', 'D4',                        # Degree
                     'A0', 'A1',                                    # 芳香性
                     'R0', 'R1']                                    # 是否在环上

EDGE_FEATURE_LIST = ['T1.0', 'T1.5', 'T2.0', 'T3.0',                # 键类型
                     'R0', 'R1',                                    # 是否在环上
                     'C0', 'C1']                                    # 是否共轭

# Set random seed
seed = 117
np.random.seed(seed)
tf.set_random_seed(seed)

"""
Setting
"""
flags = tf.app.flags
FLAGS = flags.FLAGS
# path
flags.DEFINE_string('dataset', './dataset/pi.csv', 'Dataset string.')
flags.DEFINE_string('savepath', "./myGCN/GCN/gcn.ckpt", 'Save path string')
# val test ratio
flags.DEFINE_float('val_ratio', 0.1, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 0.1, 'Ratio of test dataset')
# model
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batchSize', 1, 'Number of batches to train')
flags.DEFINE_integer('hidden', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('num_graphs', 4, 'Number of graphs')
flags.DEFINE_integer('num_dense', 4, 'Number of units in dense layer')
flags.DEFINE_integer('max_atoms', 80, 'Number of atoms')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping_begin', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

# load_data
node_feature_map = {}                                # for embedding
for idx, feature in enumerate(NODE_FEATURE_LIST):
    node_feature_map[feature] = idx
edge_feature_map = {}
for idx, feature in enumerate(EDGE_FEATURE_LIST):
    edge_feature_map[feature] = idx
adjs, node_features, y, edge_features = load_data_gcn(FLAGS.dataset, node_feature_map, FLAGS.max_atoms, edge_feature_map)
print('数据集大小：%d，特征矩阵大小：(%d, %d)' % (len(adjs), node_features[0].shape[0], node_features[0].shape[1]))

# preprocessing
node_features = list(map(preprocess_features, node_features))
if FLAGS.model == 'gcn':
    supports = []
    for adj in adjs:
        supports.append([preprocess_adj(adj)])
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    supports = []
    print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
    for adj in adjs:
        supports.append(my_chebyshev_polynomials(adj, FLAGS.max_degree, FLAGS.max_atoms))
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))



# Define placeholders
placeholders = {
    # T_k的数量，相当于Sum的参数beta
    # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 特征：节点数, 特征数
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(node_features[0][2], dtype=tf.int64))],
    'edge_features': [tf.placeholder(tf.float32, shape=(FLAGS.max_atoms, FLAGS.max_atoms, 1, len(edge_feature_map)))
                      for _ in range(FLAGS.batchSize)],
    # 节点的label
    'labels': [tf.placeholder(tf.float32, shape=(None, y.shape[1]))],
    # 'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': [tf.placeholder(tf.int32)]
}
# support for batch train
supportBatch = []
for i in range(FLAGS.batchSize):
    support = []
    for j in range(num_supports):
        support.append(tf.sparse_placeholder(tf.float32))
    supportBatch.append(support)
placeholders['support'] = supportBatch

model = GCN(placeholders, input_dim=node_features[0][2][1],
            num_nodes=node_features[0][2][0], num_graphs=FLAGS.num_graphs,
            logging=True)


sess = tf.Session()
model.load(sess, FLAGS.savepath)

res = []
for i in range(len(node_features)):
    batch = {'feature': [node_features[i]],
             'supports': [supports[i]],
             'y': [y[i, :].reshape(-1, 1)],
             'edge_feature': [edge_features[i]]}
    feed_dict = construct_feed_dict(batch['feature'], batch['supports'], batch['y'], batch['edge_feature'],
                                    placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    out = sess.run(model.features, feed_dict=feed_dict)
    res.append(out.reshape(-1))

df = pd.read_csv(FLAGS.dataset)
from sklearn.cluster import KMeans


res_dict = {}
for k in range(3, 6):
    res_dict[k] = {}
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(res)
    y = kmeans.predict(res)
    for i in range(len(y)):
        if y[i] in res_dict[k]:
            res_dict[k][y[i]].append(df.loc[i, 'label'])
        else:
            res_dict[k][y[i]] = [df.loc[i, 'label']]

for key, value in res_dict.items():
    print("分%d类:" % key)
    for c, l in value.items():
        print(c, l)
    print("============")
