import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN, MLP

import os
import time
import random

tf.disable_v2_behavior()

FEATURE_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si',         # 原子类别
                'H0', 'H1', 'H2', 'H3',                     # 连接H数量
                'D1', 'D2', 'D3', 'D4',                     # Degree
                'A0', 'A1',                                 # 芳香性
                'R0', 'R1']                                 # 是否在环上

# Set random seed
seed = 117
np.random.seed(seed)
tf.set_random_seed(seed)

"""Setting"""
flags = tf.app.flags
FLAGS = flags.FLAGS
# path
flags.DEFINE_string('dataset', './dataset/dataMethod2Deleted.csv', 'Dataset string.')
flags.DEFINE_string('savepath', "./myModel/GCN_10", 'Save path string')
# val test ratio
flags.DEFINE_float('val_ratio', 0.1, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 0.1, 'Ratio of test dataset')
# model
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('batchSize', 16, 'Number of batches to train')
flags.DEFINE_boolean('dense', False, 'dense or pooling')
flags.DEFINE_integer('hidden', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('num_graphs', 5, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')


"""load data"""
feature_map = {}                                # for embedding
for idx, feature in enumerate(FEATURE_LIST):
    feature_map[feature] = idx
adjs, features, y = load_data_gcn(FLAGS.dataset, feature_map)
print('数据集大小：%d，特征矩阵大小：(%d, %d)' % (len(adjs), features[0].shape[0], features[0].shape[1]))


"""preprocessing"""
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


"""train val split & shuffle"""
supports_train, features_train, y_train, supports_val, features_val, y_val = train_val_split_gcn(supports, features,
                                                                                                  y, FLAGS.val_ratio,
                                                                                                  FLAGS.test_ratio)
list_for_shuffle = list(range(len(supports_train)))


"""Define placeholders"""
placeholders = {
    # 特征：节点数, 特征数
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64))
                 for _ in range(FLAGS.batchSize)],
    # 节点的label
    'labels': [tf.placeholder(tf.float32, shape=(None, y.shape[1])) for _ in range(FLAGS.batchSize)],
    # 'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(FLAGS.batchSize)],
    'batchSize': tf.placeholder(tf.int64)
}
# support for batch train
# T_k的数量，相当于Sum的参数beta
supportBatch = []
for i in range(FLAGS.batchSize):
    support = []
    for j in range(num_supports):
        support.append(tf.sparse_placeholder(tf.float32))
    supportBatch.append(support)
placeholders['support'] = supportBatch


"""Create model: input_dim = features size"""
model = model_func(placeholders, input_dim=features[0][2][1],
                   num_nodes=features[0][2][0], num_graphs=FLAGS.num_graphs,
                   logging=True)


def evaluate(features, supports, y, placeholders):
    acc = []
    loss = []
    i = 0
    length = len(features)
    while i < length:
        batch = {'feature': [],
                 'supports': [],
                 'y': []}
        while len(batch['feature']) < FLAGS.batchSize and length-i >= FLAGS.batchSize-len(batch['feature']):
            batch['feature'].append(features[i])
            batch['supports'].append(supports[i])
            batch['y'].append(y[i, :].reshape(-1, 1))
            i += 1
        # 不足用任意补全，batchsize设置为有效数据大小
        while len(batch['feature']) < FLAGS.batchSize:
            batch['feature'].append(features[-1])
            batch['supports'].append(supports[-1])
            batch['y'].append(y[-1, :].reshape(-1, 1))
        feed_dict_val = construct_feed_dict(batch['feature'],
                                            batch['supports'],
                                            batch['y'],
                                            length % FLAGS.batchSize, placeholders)

        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        # print(outs)
        loss.append(outs[0])
        acc.append(outs[1])
    return np.mean(loss), np.mean(acc)


"""Train"""
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
                 'y': []}
        if len(list_for_shuffle) - i < FLAGS.batchSize:
            break
        for _ in range(FLAGS.batchSize):
            batch['feature'].append(features_train[list_for_shuffle[i]])
            batch['supports'].append(supports_train[list_for_shuffle[i]])
            batch['y'].append(y_train[list_for_shuffle[i], :].reshape(-1, 1))
            i += 1
        feed_dict = construct_feed_dict(batch['feature'], batch['supports'], batch['y'],
                                        FLAGS.batchSize, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        loss.append(outs[1])
        accu.append(outs[2])
    loss_train = np.mean(loss)
    accu_train = np.mean(accu)

    # val
    loss_val, accu_val = evaluate(features_val, supports_val, y_val, placeholders)
    val_record.append(accu_val)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_train),
          "train_acc=", "{:.5f}".format(accu_train), "val_loss=", "{:.5f}".format(loss_val),
          "val_acc=", "{:.5f}".format(accu_val), "time=", "{:.5f}".format(time.time() - t))

    random.shuffle(list_for_shuffle)                        # 每个epoch结束后，shuffle

    # early stop
    if epoch > FLAGS.early_stopping and val_record[-1] > np.mean(val_record[-(FLAGS.early_stopping+1):-1]):
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
