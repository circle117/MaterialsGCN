import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN, MMGCN

tf.disable_v2_behavior()
tf.disable_eager_execution()

FEATURE_GCN_LIST = ['*', 'C', 'N', 'O', 'F', 'S', 'Si',         # 原子类别
                    'H0', 'H1', 'H2', 'H3',                     # 连接H数量
                    'D1', 'D2', 'D3', 'D4',                     # Degree
                    'A0', 'A1',                                 # 芳香性
                    'R0', 'R1']                                 # 是否在环上

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
flags.DEFINE_string('dataset', './dataset/dataForMMGCNGeneral.csv', 'Dataset string.')
flags.DEFINE_string('savepath', './myMMGCN/tmp', 'Save path sting')
flags.DEFINE_string('store_path', './myMMGCN/MMGCN_4/mmgcn.ckpt', 'Store path string')
flags.DEFINE_float('val_ratio', 0, 'Ratio of validation dataset')
flags.DEFINE_float('test_ratio', 1, 'Ratio of validation dataset')
# Model
flags.DEFINE_string('model', 'mmgcn', 'Model string.')      # 'gcn', 'mmgen
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping_begin', 15, 'Tolerance for early stopping')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('batchSize', 1, 'Number of batch size')
flags.DEFINE_string('train_model', 'MMGCN', 'train GCN or MMGCN')
# GCN
flags.DEFINE_integer('gcn_hidden', 64, 'Number of units in GCN hidden layer .')
flags.DEFINE_integer('num_graphs', 5, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('gcn_dense', 16, 'If do gcn dense')
# Embedding
flags.DEFINE_integer('embed_dim', 1, 'Number of units in Embedding layer')
# TabNet
flags.DEFINE_integer('feature_dim', 16, 'hidden representation in feature transformation block')
flags.DEFINE_integer('output_dim', 8, 'output dimension of every decision step')
flags.DEFINE_integer('num_decision_steps', 4, 'Number of decision step')
flags.DEFINE_float('relaxation_factor', 3, 'Number of feature usage')


"""
load data
"""
feature_map = {}
for idx, feature in enumerate(FEATURE_GCN_LIST):
    feature_map[feature] = idx
adjs, gcn_features, y, discrete_features, continuous_features = load_data(FLAGS.dataset, feature_map, FEATURE_NAME)
print('数据集大小: %d，特征矩阵大小: (%d, %d)' % (len(adjs), gcn_features[0].shape[0], gcn_features[0].shape[1]))
print('离散数据大小: (%d, %d),' % (len(discrete_features[FEATURE_NAME['discrete'][0]]), len(discrete_features)),
      '连续数据大小: (%d, %d)' % (continuous_features.shape[0], continuous_features.shape[1]))


"""
preprocessing
"""
# model
gcn_features = list(map(preprocess_features, gcn_features))
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
elif FLAGS.model == 'mmgcn':
    supports = []
    print("Calculating Chebyshev polynomials up to order {}...".format(FLAGS.max_degree))
    for adj in adjs:
        supports.append(chebyshev_polynomials(adj, FLAGS.max_degree))
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
supports_train, gcn_features_train, y_train, supports_val, gcn_features_val, y_val = train_val_split_gcn(supports,
                                                                                                     gcn_features,
                                                                                                     y,
                                                                                                     FLAGS.val_ratio,
                                                                                                     FLAGS.test_ratio)
supports_test, gcn_features_test, y_test = test_split_gcn(supports, gcn_features, y, FLAGS.test_ratio)

# MMGCN
discrete_features_train, discrete_features_val, continuous_features_train, continuous_features_val = \
    train_val_split_mmgcn(discrete_features, continuous_features, FLAGS.val_ratio, FLAGS.test_ratio)
discrete_features_test, continuous_features_test = test_split_mmgcn(discrete_features, continuous_features,
                                                                  FLAGS.test_ratio)


print("training dataset: %d, validation dataset: %d, test dataset: %d"
      % (len(gcn_features_train), len(gcn_features_val), len(gcn_features_test)))

"""
Define placeholders
"""
placeholders = {
    # 特征：节点数, 特征数
    'features': [tf.sparse_placeholder(tf.float32, shape=tf.constant(gcn_features[0][2], dtype=tf.int64))
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
                   input_dim=gcn_features[0][2][1],
                   num_nodes=gcn_features[0][2][0],
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

def calculateAccu(df, gcn_features, supports, continuous_features, discrete_features, y, text):
    loss = []
    accu = []
    i = 0
    model.batch_size = FLAGS.batchSize
    # train
    while i < len(gcn_features):
        batch = {'feature': [],
                 'support': [],
                 'y': [],
                 'continuous_features': []}
        for name in FEATURE_NAME['discrete']:
            batch[name] = []
        # for _ in range(FLAGS.batchSize):
        batch['feature'].append(gcn_features[i])
        batch['support'].append(supports[i])
        batch['y'].append(y[i, :].reshape(-1, 1))
        batch['continuous_features'].append(continuous_features[i].reshape(1, -1))
        for name in FEATURE_NAME['discrete']:
            batch[name].append(discrete_features[name][i].reshape(1, -1))
            # i += 1
        i += 1
        feed_dict = my_construct_feed_dict(batch['feature'], batch['support'], batch['y'],
                                           batch['continuous_features'], batch, FEATURE_NAME['discrete'],
                                           placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
        loss.append(outs[0])
        accu.append(outs[1])
        # print(df.loc[i, 'SMILES'], " (%d):" % len(df.loc[i, 'SMILES']), outs[1])
        # print(outs)
    loss = np.mean(loss)
    accu = np.mean(accu)

    print(text)
    print("train_loss=", "{:.5f}".format(loss), "train_acc=", "{:.5f}".format(accu))
    print("================")


calculateAccu(df, gcn_features_train, supports_train,
              continuous_features_train, discrete_features_train,
              y_train, 'Train')

calculateAccu(df, gcn_features_val, supports_val,
              continuous_features_val, discrete_features_val,
              y_val, 'Validate')

calculateAccu(df, gcn_features_test, supports_test,
              continuous_features_test, discrete_features_test,
              y_test, 'Test')
