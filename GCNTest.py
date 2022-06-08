import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN
import numpy as np

tf.disable_v2_behavior()

FEATURE_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si', 'P',    # 原子类别
                'H0', 'H1', 'H2', 'H3',                     # 连接H数量
                'D1', 'D2', 'D3', 'D4',                     # Degree
                'A0', 'A1',                                 # 芳香性
                'R0', 'R1']                                 # 是否在环上

EDGE_FEATURE_LIST = ['T1.0', 'T1.5', 'T2.0', 'T3.0',                # 键类型
                     'R0', 'R1',                                    # 是否在环上
                     'C0', 'C1']                                    # 是否共轭

# Setting
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', './dataset/dataForCompare.csv', 'Dataset string.')
flags.DEFINE_string('savepath', './ModelForCompare/GCN/gcn.ckpt', 'save path string')
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
flags.DEFINE_integer('early_stopping_begin', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

# load_data
feature_map = {}                                # for embedding
for idx, feature in enumerate(FEATURE_LIST):
    feature_map[feature] = idx
edge_feature_map = {}
for idx, feature in enumerate(EDGE_FEATURE_LIST):
    edge_feature_map[feature] = idx
adjs, features, y, edge_features = load_data_gcn(FLAGS.dataset, feature_map, FLAGS.max_atoms, edge_feature_map)
print('数据集大小：%d，特征矩阵大小：(%d, %d)' % (len(adjs), features[0].shape[0], features[0].shape[1]))

# preprocessing
features = list(map(preprocess_features, features))
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

supports_train, node_features_train, edge_features_train, y_train, \
supports_val, node_features_val, edge_features_val, y_val = \
    train_val_split_gcn(supports, features, edge_features,
                        y, FLAGS.val_ratio,
                        FLAGS.test_ratio)
supports_test, node_features_test, edge_features_test, y_test = test_split_gcn(supports, features, edge_features,
                                                                               y, FLAGS.test_ratio)
print("training dataset: %d, validation dataset: %d, test dataset: %d"
      % (len(node_features_train), len(node_features_val), len(node_features_test)))

# Define placeholders
placeholders = {
    # T_k的数量，相当于Sum的参数beta
    # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 特征：节点数, 特征数
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64))],
    'edge_features': [tf.placeholder(tf.float32, shape=(FLAGS.max_atoms, FLAGS.max_atoms, 1, len(edge_feature_map)))
                      for _ in range(FLAGS.batchSize)],
    # 节点的label
    'labels': [tf.placeholder(tf.float32, shape=(None, y.shape[1]))],
    # 'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': [tf.placeholder(tf.int32)],
    'batchSize': tf.placeholder(tf.int64)
}
# support for batch train
supportBatch = []
for i in range(FLAGS.batchSize):
    support = []
    for j in range(num_supports):
        support.append(tf.sparse_placeholder(tf.float32))
    supportBatch.append(support)
placeholders['support'] = supportBatch

model = GCN(placeholders, input_dim=features[0][2][1],
            num_nodes=features[0][2][0], num_graphs=FLAGS.num_graphs,
            logging=True)


sess = tf.Session()
model.load(sess, FLAGS.savepath)


def calculateLossAndAccu(features, edge_features, supports, y, text, df):
    loss = []
    accuracy = []
    pred = []
    label = []
    for i in range(len(features)):
        batch = {'feature': [features[i]],
                 'supports': [supports[i]],
                 'y': [y[i, :].reshape(-1, 1)],
                 'edge_features': [edge_features[i]]}
        feed_dict = construct_feed_dict(batch['feature'], batch['supports'], batch['y'], batch['edge_features'],
                                        placeholders)
        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        res = sess.run([model.outputs, model.labels], feed_dict=feed_dict)
        pred.append(res[0])
        label.append(res[1])
        # print(df.loc[i, 'SMILES'], " (%d) :"%len(df.loc[i, 'SMILES']), res[0], res[1], outs[1])
        loss.append(outs[0])
        accuracy.append(outs[1])
    loss = np.mean(loss)
    accu = np.mean(accuracy)
    print(text)
    r2_1 = 0
    r2_2 = 0
    mean_y = np.mean(label)
    for i in range(len(pred)):
        r2_1 += (pred[i]-label[i])**2
        r2_2 += (label[i]-mean_y)**2
    res = 1-r2_1*1.0/r2_2
    print("loss: %.5f, accuracy: %.5f, r^2: %.5f" % (loss, accu, res))
    print("================")

df = pd.read_csv(FLAGS.dataset)

calculateLossAndAccu(node_features_train, edge_features_train, supports_train, y_train, 'Train', df)
calculateLossAndAccu(node_features_val, edge_features_train, supports_val, y_val, 'Validation', df)
calculateLossAndAccu(node_features_test, edge_features_test, supports_test, y_test, 'Test', df)
