import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN, MMGCN

import time
import random
import os

tf.disable_v2_behavior()
tf.disable_eager_execution()

NODE_FEATURE_GCN_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si',         # atom type
                    'H0', 'H1', 'H2', 'H3',                     # number of the connected H atom
                    'D1', 'D2', 'D3', 'D4',                     # Degree
                    'A0', 'A1',                                 # aromaticity
                    'R0', 'R1']                                 # is in ring

EDGE_FEATURE_LIST = ['T1.0', 'T1.5', 'T2.0', 'T3.0',                # bond type
                     'R0', 'R1',                                    # is in ring
                     'C0', 'C1']                                    # is conjugated

FEATURE_NAME = {'discrete': ['Solvent', 'method2', 'temperature1'],
                'continuous': ['time1', 'minTemp', 'maxTemp', 'time2']}

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


"""
Settings
"""
flags = tf.app.flags
FLAGS = flags.FLAGS
# path
flags.DEFINE_string('dataset', './dataset/dataForMMGCN.csv', 'Dataset string.')
flags.DEFINE_string('savepath', './myMMGCN/MMGCN/mmgcn.ckpt', 'Save path sting')
flags.DEFINE_string('store_path', './myMMGCN/GCN/mmgcn.ckpt', 'Store path string')
flags.DEFINE_float('val_ratio', 0.1, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 0.1, 'Ratio of validation dataset')
# Model
flags.DEFINE_string('model', 'mmgcn', 'Model string.')                      # 'gcn', 'mmgen
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping_begin', 10, 'Tolerance for early stopping')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('batchSize', 16, 'Number of batch size')
flags.DEFINE_string('train_model', 'TabNet', 'the model to train')              # GCN, TabNet, All
# GCN
flags.DEFINE_integer('gcn_hidden', 64, 'Number of units in GCN hidden layer .')
flags.DEFINE_integer('num_graphs', 4, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('gcn_dense', 16, 'If do gcn dense')
flags.DEFINE_integer('max_atoms', 80, 'Number of atoms')
# Embedding
flags.DEFINE_integer('embed_dim', 1, 'Number of units in Embedding layer')
# TabNet
flags.DEFINE_integer('feature_dim', 16, 'hidden representation in feature transformation block')
flags.DEFINE_integer('output_dim', 8, 'output dimension of every decision step')
flags.DEFINE_integer('num_decision_steps', 4, 'Number of decision step')
flags.DEFINE_float('relaxation_factor', 1.5, 'Number of feature usage')


"""
load data
"""
node_feature_map = {}
for idx, feature in enumerate(NODE_FEATURE_GCN_LIST):
    node_feature_map[feature] = idx
edge_feature_map = {}
for idx, feature in enumerate(EDGE_FEATURE_LIST):
    edge_feature_map[feature] = idx
adjs, node_features, y, edge_features, discrete_features, continuous_features = \
    load_data(FLAGS.dataset, node_feature_map, FEATURE_NAME, FLAGS.max_atoms, edge_feature_map)
print('dataset size: %d，feature matrix: (%d, %d)' % (len(adjs), node_features[0].shape[0], node_features[0].shape[1]))
print('discrete data: (%d, %d),' % (len(discrete_features[FEATURE_NAME['discrete'][0]]), len(discrete_features)),
      'continuous data: (%d, %d)' % (continuous_features.shape[0], continuous_features.shape[1]))


"""
preprocessing
"""
# model
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
        supports.append(chebyshev_polynomials(adj, FLAGS.max_degree))
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'mmgcn':
    supports = []
    print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
    for adj in adjs:
        supports.append(my_chebyshev_polynomials(adj, FLAGS.max_degree, FLAGS.max_atoms))
    num_supports = 1 + FLAGS.max_degree
    model_func = MMGCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# MMGCN
continuous_features = preprocess_cfeatures(continuous_features)

"""
train val split
"""
# GCN
supports_train, node_features_train, edge_features_train, y_train, \
supports_val, node_features_val, edge_features_val, y_val = \
    train_val_split_gcn(supports,
                        node_features,
                        edge_features,
                        y,
                        FLAGS.val_ratio,
                        FLAGS.test_ratio)
# MMGCN
discrete_features_train, discrete_features_val, continuous_features_train, continuous_features_val = \
    train_val_split_mmgcn(discrete_features, continuous_features, FLAGS.val_ratio, FLAGS.test_ratio)
list_for_shuffle = list(range(len(supports_train)))
list_for_shuffle_val = list(range(len(supports_val)))


"""
Define placeholders
"""
placeholders = {
    # feature：[nodes, features]
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(node_features[0][2], dtype=tf.int64))
                 for _ in range(FLAGS.batchSize)],
    'edge_features': [tf.placeholder(tf.float32, shape=(FLAGS.max_atoms, FLAGS.max_atoms, 1, len(edge_feature_map)))
                      for _ in range(FLAGS.batchSize)],
    # label
    'labels': [tf.placeholder(tf.float32, shape=(None, y.shape[1])) for _ in range(FLAGS.batchSize)],
    # 连续特征
    'con_features': [tf.placeholder(tf.float32, shape=(None, continuous_features.shape[1]))
                     for _ in range(FLAGS.batchSize)],
    # dropout
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': [tf.placeholder(tf.int32) for _ in range(FLAGS.batchSize)]
}
# support for batch train
supportBatch = []
for i in range(FLAGS.batchSize):
    support = []
    for j in range(num_supports):
        support.append(tf.sparse_placeholder(tf.float32))
    supportBatch.append(support)
placeholders['support'] = supportBatch
# discrete features
for key, value in discrete_features.items():
    placeholders[key] = [tf.placeholder(tf.float32, shape=(None, value.shape[1])) for _ in range(FLAGS.batchSize)]
# discrete features dimension
d_feature_dim = []
for name in FEATURE_NAME['discrete']:
    d_feature_dim.append((name, discrete_features_train[name].shape[1]))

"""
Create model: input_dim = features size
"""
model = model_func(placeholders,
                   input_dim=node_features[0][2][1],
                   num_nodes=node_features[0][2][0],
                   num_graphs=FLAGS.num_graphs,
                   d_feature_dim=d_feature_dim,
                   num_features=len(FEATURE_NAME['continuous'])+len(FEATURE_NAME['discrete']),
                   feature_dim=FLAGS.feature_dim,
                   output_dim=FLAGS.output_dim,
                   num_decision_steps=FLAGS.num_decision_steps,
                   relaxation_factor=FLAGS.relaxation_factor,
                   batch_momentum=0.99,
                   virtual_batch_size=FLAGS.batchSize,
                   epsilon=0.00001,
                   is_training=True,
                   logging=True)


def evaluate(node_features, edge_features, supports, y, con_features, dis_features, placeholders, list_for_shuffle):
    loss = []
    accu = []
    i = 0
    index = 0
    while i < len(list_for_shuffle):
        batch = {'feature': [],
                 'support': [],
                 'y': [],
                 'edge_features': [],
                 'continuous_features': []}
        for name in FEATURE_NAME['discrete']:
            batch[name] = []
        for _ in range(FLAGS.batchSize):
            if i == len(list_for_shuffle):
                batch['feature'].append(node_features[list_for_shuffle[index]])
                batch['support'].append(supports[list_for_shuffle[index]])
                batch['y'].append(y[list_for_shuffle[index], :].reshape(-1, 1))
                batch['edge_features'].append(edge_features[list_for_shuffle[index]])
                batch['continuous_features'].append(con_features[list_for_shuffle[index]].reshape(1, -1))
                for name in FEATURE_NAME['discrete']:
                    batch[name].append(dis_features[name][list_for_shuffle[index]].reshape(1, -1))
                index += 1
            else:
                batch['feature'].append(node_features[list_for_shuffle[i]])
                batch['support'].append(supports[list_for_shuffle[i]])
                batch['y'].append(y[list_for_shuffle[i], :].reshape(-1, 1))
                batch['edge_features'].append(edge_features[list_for_shuffle[i]])
                batch['continuous_features'].append(con_features[list_for_shuffle[i]].reshape(1, -1))
                for name in FEATURE_NAME['discrete']:
                    batch[name].append(dis_features[name][list_for_shuffle[i]].reshape(1, -1))
                i += 1

        feed_dict_val = my_construct_feed_dict(batch['feature'],
                                               batch['support'], batch['y'],
                                               batch['edge_features'],
                                               batch['continuous_features'], batch,
                                               FEATURE_NAME['discrete'], placeholders)
        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        loss.append(outs[0])
        accu.append(outs[1])
    return np.mean(loss), np.mean(accu)


"""
Train
"""
sess = tf.Session()
if FLAGS.train_model == 'GCN' or FLAGS.train_model == 'All':
    sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())
    model.load(sess, FLAGS.store_path)

val_record = []
start = time.time()
for epoch in range(FLAGS.epochs):

    t = time.time()
    loss = []
    accu = []
    i = 0
    while i < len(list_for_shuffle):
        batch = {'feature': [],
                 'support': [],
                 'y': [],
                 'edge_features': [],
                 'continuous_features': []}
        for name in FEATURE_NAME['discrete']:
            batch[name] = []

        if len(list_for_shuffle) - i < FLAGS.batchSize:
            break
        else:
            for _ in range(FLAGS.batchSize):
                batch['feature'].append(node_features_train[list_for_shuffle[i]])
                batch['support'].append(supports_train[list_for_shuffle[i]])
                batch['y'].append(y_train[list_for_shuffle[i], :].reshape(-1, 1))
                batch['edge_features'].append(edge_features_train[list_for_shuffle[i]])
                batch['continuous_features'].append(continuous_features_train[list_for_shuffle[i]].reshape(1, -1))
                for name in FEATURE_NAME['discrete']:
                    batch[name].append(discrete_features_train[name][list_for_shuffle[i]].reshape(1, -1))
                i += 1
            feed_dict = my_construct_feed_dict(batch['feature'], batch['support'], batch['y'], batch['edge_features'],
                                               batch['continuous_features'], batch, FEATURE_NAME['discrete'],
                                               placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        loss.append(outs[1])
        accu.append(outs[2])
    loss_train = np.mean(loss)
    accu_train = np.mean(accu)

    # val
    loss_val, accu_val = evaluate(node_features_val, edge_features_val, supports_val, y_val,
                                  continuous_features_val, discrete_features_val,
                                  placeholders, list_for_shuffle_val)
    val_record.append(accu_val)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_train),
          "train_acc=", "{:.5f}".format(accu_train), "val_loss=", "{:.5f}".format(loss_val),
          "val_acc=", "{:.5f}".format(accu_val), "time=", "{:.5f}".format(time.time() - t))

    random.shuffle(list_for_shuffle)                        # 每个epoch结束后，shuffle
    random.shuffle(list_for_shuffle_val)

    if epoch > FLAGS.early_stopping_begin and \
            val_record[-1] > np.mean(val_record[-(FLAGS.early_stopping+1):-1]):
        print('Early stopping...')
        break
    else:
        if os.path.exists(FLAGS.savepath):
            for file in os.listdir(FLAGS.savepath):
                file_path = os.path.join(FLAGS.savepath, file)
                os.remove(file_path)
        model.save(sess, FLAGS.savepath + '/mmgcn.ckpt')

print("Save to path:", model.save_path)
print("Time = %.5f" % (time.time() - start))

