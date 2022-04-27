import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN, MLP
import numpy as np

tf.disable_v2_behavior()

FEATURE_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si',         # 原子类别
                'H0', 'H1', 'H2', 'H3',                     # 连接H数量
                'D1', 'D2', 'D3', 'D4',                     # Degree
                'A0', 'A1',                                 # 芳香性
                'R0', 'R1']                                 # 是否在环上

# Setting
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', './dataset/dataMethod2Deleted.csv', 'Dataset string.')
flags.DEFINE_string('savepath', './myGCN/GCN_10/gcn.ckpt', 'save path string')
flags.DEFINE_float('val_ratio', 0.1, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 0.1, 'Ratio of test dataset')
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')              # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batchSize', 1, 'Number of batches to train')
flags.DEFINE_boolean('dense', False, 'dense or pooling')                # pooling每个hidden相等
flags.DEFINE_integer('hidden', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('num_graphs', 5, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

# load_data
feature_map = {}                                # for embedding
for idx, feature in enumerate(FEATURE_LIST):
    feature_map[feature] = idx
adjs, features, y = load_data_gcn(FLAGS.dataset, feature_map)
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
        supports.append(chebyshev_polynomials(adj, FLAGS.max_degree))
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    supports = []
    for adj in adjs:
        supports.append([preprocess_adj(adj)])  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


supports_test, features_test, y_test = test_split_gcn(supports, features, y, FLAGS.test_ratio)

# Define placeholders
placeholders = {
    # T_k的数量，相当于Sum的参数beta
    # 'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 特征：节点数, 特征数
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64))],
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


loss = []
accuracy = []
for i in range(len(features_test)):
    batch = {'feature': [features_test[i]],
             'supports': [supports_test[i]],
             'y': [y_test[i, :].reshape(-1, 1)]}
    feed_dict = construct_feed_dict(batch['feature'], batch['supports'], batch['y'],
                                    FLAGS.batchSize, placeholders)
    outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
    print(outs)
    loss.append(outs[0])
    accuracy.append(outs[1])

print("loss: %.5f, accuracy: %.5f" % (np.mean(loss), np.mean(accuracy)))
