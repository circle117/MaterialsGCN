import tensorflow._api.v2.compat.v1 as tf
from MyUtils import *
from utils import *
from model import GCN, MLP
import time

tf.disable_v2_behavior()

FEATURE_LIST = ['*', 'C', 'N', 'O', 'F', 'S',               # 原子类别
                'H0', 'H1', 'H2', 'H3',                     # 连接H数量
                'D1', 'D2', 'D3', 'D4',                     # Degree
                'A0', 'A1',                                 # 芳香性
                'R0', 'R1']                                 # 是否在环上

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', './Dataset_test/data_Method2.csv', 'Dataset string.')       # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('val_ratio', 0.2, 'Ratio of validation dataset')
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')      # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 8, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 1, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')


# load_data
feature_map = {}
for idx, feature in enumerate(FEATURE_LIST):
    feature_map[feature] = idx
adjs, features, y = load_data(FLAGS.dataset, feature_map, FLAGS.val_ratio)
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


# train val split
supports_train, features_train, y_train, supports_val, features_val, y_val = train_test_split(supports, features, y, FLAGS.val_ratio)


# Define placeholders
placeholders = {
    # T_k的数量，相当于Sum的参数beta
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 特征：节点数, 特征数
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64)),
    # 节点的label
    'labels': tf.placeholder(tf.float32, shape=(None, y.shape[1])),
    # 'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model: input_dim = features size
model = model_func(placeholders, input_dim=features[0][2][1], logging=True)

sess = tf.Session()

def evaluate(features, supports, y, placeholders):
    loss = []
    accu = []
    for i in range(len(features)):
        feed_dict_val = construct_feed_dict(features[i], supports[i], y[i].reshape((-1, 1)), placeholders)
        outs = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        loss.append(outs[0])
        accu.append(outs[1])
    return np.mean(loss), np.mean(accu)

sess.run(tf.global_variables_initializer())

for epoch in range(FLAGS.epochs):

    t = time.time()

    # trian
    loss = []
    accu = []
    val_record = []
    for i in range(len(features)):
        feed_dict = construct_feed_dict(features[i], supports[i], y[i, :].reshape((-1, 1)), placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        loss.append(outs[1])
        accu.append(outs[2])
    loss_train = np.mean(loss)
    accu_train = np.mean(accu)

    #val
    loss_val, accu_val = evaluate(features_val, supports_val, y_val, placeholders)
    val_record.append(loss_val)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_train),
          "train_acc=", "{:.5f}".format(accu_train), "val_loss=", "{:.5f}".format(loss_val),
          "val_acc=", "{:.5f}".format(accu_val), "time=", "{:.5f}".format(time.time() - t))

    if epoch>FLAGS.early_stopping and val_record[-1]>np.mean(val_record[-(FLAGS.early_stopping+1):-1]):
        print('Early stopping...')
        break
