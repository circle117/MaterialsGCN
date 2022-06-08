import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN, MMGCN

tf.disable_v2_behavior()
tf.disable_eager_execution()

NODE_FEATURE_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si',         # 原子类别
                     'H0', 'H1', 'H2', 'H3',                     # 连接H数量
                     'D1', 'D2', 'D3', 'D4',                     # Degree
                     'A0', 'A1',                                 # 芳香性
                     'R0', 'R1']                                 # 是否在环上

EDGE_FEATURE_LIST = ['T1.0', 'T1.5', 'T2.0', 'T3.0',                # 键类型
                     'R0', 'R1',                                    # 是否在环上
                     'C0', 'C1']                                    # 是否共轭

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
flags.DEFINE_string('savepath', './myMMGCN/tmp', 'Save path string')
flags.DEFINE_string('store_path', './myMMGCN/MMGCN/mmgcn.ckpt', 'Store path string')
flags.DEFINE_float('val_ratio', 0.1, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 0.1, 'Ratio of validation dataset')
# Model
flags.DEFINE_string('model', 'mmgcn', 'Model string.')      # 'gcn', 'mmgen
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping_begin', 20, 'Tolerance for early stopping')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('batchSize', 1, 'Number of batch size')
flags.DEFINE_string('train_model', 'TabNet', 'train GCN or MMGCN')
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
for idx, feature in enumerate(NODE_FEATURE_LIST):
    node_feature_map[feature] = idx
edge_feature_map = {}
for idx, feature in enumerate(EDGE_FEATURE_LIST):
    edge_feature_map[feature] = idx
adjs, node_features, y, edge_features, discrete_features, continuous_features = \
    load_data(FLAGS.dataset, node_feature_map, FEATURE_NAME, FLAGS.max_atoms, edge_feature_map)
print('数据集大小: %d，特征矩阵大小: (%d, %d)' % (len(adjs), node_features[0].shape[0], node_features[0].shape[1]))
print('离散数据大小: (%d, %d),' % (len(discrete_features[FEATURE_NAME['discrete'][0]]), len(discrete_features)),
      '连续数据大小: (%d, %d)' % (continuous_features.shape[0], continuous_features.shape[1]))


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
        supports.append(my_chebyshev_polynomials(adj, FLAGS.max_degree, FLAGS.max_atoms))
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
supports_test, node_features_test, edge_features_test, y_test = test_split_gcn(supports, node_features, edge_features,
                                                                               y, FLAGS.test_ratio)

# MMGCN
discrete_features_train, discrete_features_val, continuous_features_train, continuous_features_val = \
    train_val_split_mmgcn(discrete_features, continuous_features, FLAGS.val_ratio, FLAGS.test_ratio)
discrete_features_test, continuous_features_test = test_split_mmgcn(discrete_features, continuous_features,
                                                                  FLAGS.test_ratio)


print("training dataset: %d, validation dataset: %d, test dataset: %d"
      % (len(node_features_train), len(node_features_val), len(node_features_test)))

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
    # 连续特征
    'con_features': [tf.placeholder(tf.float32, shape=(None, continuous_features.shape[1]))
                     for _ in range(FLAGS.batchSize)],
    # dropout的比例
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
                   is_training=False,
                   logging=True)

"""
Train
"""
sess = tf.Session()
model.load(sess, FLAGS.store_path)

df = pd.read_csv(FLAGS.dataset)

fea_temp = np.zeros((1, 7))

def calculateAccu(df, node_features, edge_features, supports, continuous_features, discrete_features, y, text):
    loss = []
    accu = []
    pred = []
    label = []
    i = 0
    global fea_temp
    model.batch_size = FLAGS.batchSize
    temp = np.zeros((1, 7))
    # train
    while i < len(node_features):
        batch = {'feature': [],
                 'support': [],
                 'y': [],
                 'edge_features': [],
                 'continuous_features': []}
        for name in FEATURE_NAME['discrete']:
            batch[name] = []
        # for _ in range(FLAGS.batchSize):
        batch['feature'].append(node_features[i])
        batch['support'].append(supports[i])
        batch['y'].append(y[i, :].reshape(-1, 1))
        batch['edge_features'].append(edge_features[i])
        batch['continuous_features'].append(continuous_features[i].reshape(1, -1))
        for name in FEATURE_NAME['discrete']:
            batch[name].append(discrete_features[name][i].reshape(1, -1))
            # i += 1
        i += 1
        feed_dict = my_construct_feed_dict(batch['feature'], batch['support'], batch['y'], batch['edge_features'],
                                           batch['continuous_features'], batch, FEATURE_NAME['discrete'],
                                           placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        res = sess.run([model.outputs, model.labels], feed_dict=feed_dict)
        temp += sess.run(model.temp, feed_dict=feed_dict)
        pred.append(res[0])
        label.append(res[1])
        loss.append(outs[0])
        accu.append(outs[1])
        # print(df.loc[i, 'SMILES'], " (%d):" % len(df.loc[i, 'SMILES']), outs[1])
        # print(outs)
    loss = np.mean(loss)
    accu = np.mean(accu)
    r2_1 = 0
    r2_2 = 0
    mean_y = np.mean(label)
    for i in range(len(pred)):
        r2_1 += (pred[i]-label[i])**2
        r2_2 += (label[i]-mean_y)**2
    res = 1-r2_1*1.0/r2_2

    print(text)
    print("loss: %.5f, accuracy: %.5f, r^2: %.5f" % (loss, accu, res))
    print(temp/len(node_features))
    fea_temp = fea_temp + temp/len(node_features)
    print("================")


calculateAccu(fea_temp, node_features_train, edge_features_train, supports_train,
              continuous_features_train, discrete_features_train,
              y_train, 'Train')

calculateAccu(fea_temp, node_features_val, edge_features_val, supports_val,
              continuous_features_val, discrete_features_val,
              y_val, 'Validate')

calculateAccu(fea_temp, node_features_test, edge_features_test, supports_test,
              continuous_features_test, discrete_features_test,
              y_test, 'Test')

print(fea_temp/3)