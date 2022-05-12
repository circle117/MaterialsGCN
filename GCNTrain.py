import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN

import os
import time
import random

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
flags.DEFINE_string('dataset', './dataset/dataForGCN.csv', 'Dataset string.')
flags.DEFINE_string('savepath', "./myGCN/GCN_4", 'Save path string')
# val test ratio
flags.DEFINE_float('val_ratio', 0.1, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 0.1, 'Ratio of test dataset')
# model
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batchSize', 16, 'Number of batches to train')
flags.DEFINE_integer('hidden', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('num_graphs', 4, 'Number of graphs')
flags.DEFINE_integer('num_dense', 16, 'Number of units in dense layer')
flags.DEFINE_integer('max_atoms', 60, 'Number of atoms')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping_begin', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

"""
load data
"""
node_feature_map = {}                                # for embedding
for idx, feature in enumerate(NODE_FEATURE_LIST):
    node_feature_map[feature] = idx
edge_feature_map = {}
for idx, feature in enumerate(EDGE_FEATURE_LIST):
    edge_feature_map[feature] = idx
adjs, node_features, y, edge_features = load_data_gcn(FLAGS.dataset, node_feature_map, FLAGS.max_atoms, edge_feature_map)
print('数据集大小：%d，特征矩阵大小：(%d, %d)' % (len(adjs), node_features[0].shape[0], node_features[0].shape[1]))


"""
preprocessing
"""
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

temp = supports[0]

"""
train val split & shuffle
"""
supports_train, node_features_train, edge_features_train, y_train, \
supports_val, node_features_val, edge_features_val, y_val = \
    train_val_split_gcn(supports, node_features, edge_features,
                        y, FLAGS.val_ratio,
                        FLAGS.test_ratio)
list_for_shuffle = list(range(len(supports_train)))
# length = int(len(supports_val)/FLAGS.batchSize)*FLAGS.batchSize
# list_for_shuffle_val = list(range(length))
print("training dataset: %d, validation dataset: %d" % (len(list_for_shuffle), len(supports_val)))


"""
Define placeholders
"""
placeholders = {
    # 特征：节点数, 特征数
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(node_features[0][2], dtype=tf.int64))
                 for _ in range(FLAGS.batchSize)],
    'edge_features': [tf.placeholder(tf.float32, shape=(FLAGS.max_atoms, FLAGS.max_atoms, 1, len(edge_feature_map)))
                      for _ in range(FLAGS.batchSize)],
    # 节点的label
    'labels': [tf.placeholder(tf.float32, shape=(None, y.shape[1])) for _ in range(FLAGS.batchSize)],
    # 'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(FLAGS.batchSize)]
}
# support for batch train
# T_k的数量，相当于Sum的参数beta
supportBatch = []
for i in range(FLAGS.batchSize):
    support = []
    for j in range(num_supports):
        support.append(tf.sparse_placeholder(tf.float32, shape=(FLAGS.max_atoms, FLAGS.max_atoms)))
    supportBatch.append(support)
placeholders['support'] = supportBatch


"""
Create model: input_dim = features size
"""
model = model_func(placeholders, input_dim=node_features[0][2][1],
                   num_nodes=node_features[0][2][0], num_graphs=FLAGS.num_graphs,
                   logging=True)


def evaluate(features, edge_features, supports, y, placeholders):
    acc = []
    loss = []
    i = 0
    length = len(features)
    while i < length:
        batch = {'feature': [],
                 'supports': [],
                 'y': [],
                 'edge_features': []}
        if length-i < FLAGS.batchSize:
            break
        while len(batch['feature']) < FLAGS.batchSize:
            batch['feature'].append(features[i])
            batch['supports'].append(supports[i])
            batch['y'].append(y[i, :].reshape(-1, 1))
            batch['edge_features'].append(edge_features[i])
            i += 1
        feed_dict_val = construct_feed_dict(batch['feature'],
                                            batch['supports'],
                                            batch['y'],
                                            batch['edge_features'],
                                            placeholders)

        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        # print(outs)
        loss.append(outs[0])
        acc.append(outs[1])
    return np.mean(loss), np.mean(acc)


"""
Train
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer())

val_record = []
start = time.time()
for epoch in range(FLAGS.epochs):

    t = time.time()
    loss = []
    accu = []
    i = 0
    while i < len(list_for_shuffle):
        batch = {'feature': [],
                 'supports': [],
                 'y': [],
                 'edge_feature': []}
        if len(list_for_shuffle) - i < FLAGS.batchSize:
            break
        for _ in range(FLAGS.batchSize):
            batch['feature'].append(node_features_train[list_for_shuffle[i]])
            batch['supports'].append(supports_train[list_for_shuffle[i]])
            batch['y'].append(y_train[list_for_shuffle[i], :].reshape(-1, 1))
            batch['edge_feature'].append(edge_features_train[list_for_shuffle[i]])
            i += 1
        feed_dict = construct_feed_dict(batch['feature'], batch['supports'], batch['y'], batch['edge_feature'],
                                        placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # temp = sess.run(model.temp, feed_dict=feed_dict)
        loss.append(outs[1])
        accu.append(outs[2])
    loss_train = np.mean(loss)
    accu_train = np.mean(accu)

    # val
    loss_val, accu_val = evaluate(node_features_val, edge_features_val, supports_val, y_val,
                                  placeholders)
    val_record.append(accu_val)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_train),
          "train_acc=", "{:.5f}".format(accu_train), "val_loss=", "{:.5f}".format(loss_val),
          "val_acc=", "{:.5f}".format(accu_val), "time=", "{:.5f}".format(time.time() - t))

    random.shuffle(list_for_shuffle)                        # 每个epoch结束后，shuffle
    # random.shuffle(list_for_shuffle_val)

    # early stop
    if epoch > FLAGS.early_stopping_begin and val_record[-1] > np.mean(val_record[-(FLAGS.early_stopping+1):-1]):
        print('Early stopping...')
        break
    else:
        if os.path.exists(FLAGS.savepath):
            for file in os.listdir(FLAGS.savepath):
                file_path = os.path.join(FLAGS.savepath, file)
                os.remove(file_path)
        model.save(sess, FLAGS.savepath+'/gcn.ckpt')

print(model.save_path)
print("Time = %.5f" % (time.time() - start))
