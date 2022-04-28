import platform

from layers import *
from metrics import *
import tensorflow_addons as tfa

flags = tf.app.flags
FLAGS = flags.FLAGS


def glu(act, n_units):
    """Generalized linear unit nonlinear activation."""
    return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.save_path = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self, output):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        self.save_path = saver.save(sess, path)

    def load(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        if not path:
            raise AttributeError("Save path noe provided.")
        saver = tf.train.Saver(self.vars)
        save_path = path
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense1(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense1(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self, output):
        return tf.nn.softmax(output)


class GCN(Model):
    def __init__(self, placeholders, input_dim, num_nodes, num_graphs, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim                                              # 特征数
        self.outputs = []                                                       # batch输出
        self.num_nodes = num_nodes                                              # 节点数
        self.num_graphs = num_graphs                                            # GCN层数
        self.GCN_outputs = []
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'][0].get_shape().as_list()[1]       # 分类数
        self.placeholders = placeholders
        self.labels = tf.concat(placeholders['labels'], 0)

        self.features = None

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, use_locking=True)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Mean absolute error
        self.loss += mean_absolute_error(self.outputs, self.labels)

    def _accuracy(self):
        self.accuracy = mean_absolute_error(self.outputs, self.labels)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging))

        for i in range(self.num_graphs-1):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                logging=self.logging))

        if FLAGS.dense:
            self.layers.append(Dense1(input_dim=FLAGS.hidden3,
                                      output_dim=self.output_dim,
                                      placeholders=self.placeholders,
                                      act=tf.nn.relu,
                                      dropout=True,
                                      bias=True))

            self.layers.append(Dense2(input_dim=self.output_dim,
                                      output_dim=self.num_nodes,
                                      placeholders=self.placeholders,
                                      act=lambda x: x,
                                      bias=True))
        else:
            self.layers.append(Dense1(input_dim=FLAGS.hidden,
                                      output_dim=self.output_dim,
                                      placeholders=self.placeholders,
                                      act=tf.nn.relu,
                                      dropout=True,
                                      bias=True,
                                      logging=self.logging))

            self.layers.append(Dense2(input_dim=self.output_dim,
                                      output_dim=self.num_nodes,
                                      placeholders=self.placeholders,
                                      act=lambda x: x,
                                      dropout=False,
                                      bias=True,
                                      logging=self.logging))

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        for i in range(FLAGS.batchSize):
            self.activations = [self.inputs[i]]
            self.GCN_outputs = []
            for layer in self.layers:
                hidden = layer(self.activations[-1], i)
                self.activations.append(hidden)
                if (not FLAGS.dense) and len(self.GCN_outputs) < self.num_graphs:
                    self.GCN_outputs.append(hidden)
                if (not FLAGS.dense) and len(self.activations)-1 == self.num_graphs:
                    GCN_outputs = tf.stack(self.GCN_outputs, axis=1)
                    GCN_outputs = tf.reshape(GCN_outputs, [1, self.num_graphs, self.num_nodes, FLAGS.hidden])
                    hidden = tf.nn.max_pool(GCN_outputs,
                                            ksize=[1, self.num_graphs, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID')
                    hidden = tf.reshape(hidden, [self.num_nodes, FLAGS.hidden])
                    self.features = hidden
                    self.activations.append(hidden)
            self.outputs.append(self.predict(self.activations[-1]))

        self.outputs = tf.concat(self.outputs, axis=0)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self, outputs):
        return tf.nn.softplus(outputs)


class MMGCN(Model):
    def __init__(self,
                 placeholders,
                 input_dim,
                 num_nodes,
                 num_graphs,
                 d_feature_dim,
                 num_features,
                 feature_dim,
                 output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 epsilon,
                 **kwargs):
        super(MMGCN, self).__init__(**kwargs)

        self.batch_size = len(placeholders['features'])
        self.is_training = True

        # 特征
        self.gcn_inputs = placeholders['features']                              # GCN特征
        self.mlp_inputs_c = tf.concat(placeholders['con_features'], 0)          # 连续特征
        self.mlp_inputs_d = {}                                                  # 离散特征
        self.d_feature_dim = d_feature_dim                                      # 离散特征名称及dim
        for name, dim in self.d_feature_dim:
            self.mlp_inputs_d[name] = tf.concat(placeholders[name], 0)
        self.mlp_inputs = []                                                    # mlp特征
        self.fusion_inputs = None                                               # fusion特征
        self.labels = tf.concat(placeholders['labels'], 0)

        # 层大小
        self.input_dim = input_dim                                              # gcn特征数
        self.num_nodes = num_nodes                                              # gcn节点数
        self.num_graphs = num_graphs                                            # GCN层数
        self.output_dim = placeholders['labels'][0].get_shape().as_list()[1]    # 模型输出大小
        self.mlp_input_dim = placeholders['con_features'][0].get_shape().as_list()[1] + \
                             FLAGS.embed_dim*len(self.d_feature_dim)

        # 层list
        self.gcn_layers = []                                                    # GCN层
        self.embedding_layers = {}                                              # Embedding层
        self.mlp_layers = []                                                    # MLP层
        self.fusion_layers = []                                                 # 融合层

        self.gcn_outputs = []                                                   # GCN每层输出
        self.gcn_output = None                                                  # GCN最终输出
        self.mlp_outputs = []                                                   # MLP每层输出
        self.mlp_output = None                                                  # MLP最终输出
        self.fusion_outputs = []                                                # Fusion输出
        self.outputs = []
        self.placeholders = placeholders

        # Tabnet parameters
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim_tabnet = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon
        self.reuse = False

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, use_locking=True)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.gcn_layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Mean absolute error
        self.loss += mean_absolute_error(self.outputs, self.labels)

    def _accuracy(self):
        self.accuracy = mean_absolute_error(self.outputs, self.labels)

    def _build(self):
        """
        GCN
        """
        self.gcn_layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.gcn_hidden,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                sparse_inputs=True,
                                                logging=self.logging))

        for i in range(FLAGS.num_graphs-1):
            self.gcn_layers.append(GraphConvolution(input_dim=FLAGS.gcn_hidden,
                                                    output_dim=FLAGS.gcn_hidden,
                                                    placeholders=self.placeholders,
                                                    act=tf.nn.relu,
                                                    dropout=False,
                                                    logging=self.logging))

        self.gcn_layers.append(Dense1(input_dim=FLAGS.gcn_hidden,
                                      output_dim=self.output_dim,
                                      placeholders=self.placeholders,
                                      act=tf.nn.relu,
                                      dropout=True,
                                      logging=self.logging))

        self.gcn_layers.append(Dense2(input_dim=FLAGS.gcn_dense,
                                      output_dim=self.num_nodes,
                                      placeholders=self.placeholders,
                                      act=lambda x: x,
                                      dropout=False,
                                      logging=self.logging))

        """
        Embedding
        """
        for d_feature in self.d_feature_dim:
            self.embedding_layers[d_feature[0]] = Dense1(
                input_dim=d_feature[1],
                output_dim=FLAGS.embed_dim,
                placeholders=self.placeholders,
                act=lambda x: x,
                dropout=False,
                bias=False,
                logging=self.logging
            )

        """Fusion"""
        if FLAGS.gcn_dense:
            fusion_input_dim = FLAGS.output_dim+FLAGS.gcn_dense
        else:
            fusion_input_dim = FLAGS.output_dim+self.num_nodes
        # self.fusion_layers.append(Dense1(input_dim=fusion_input_dim,
        #                                  output_dim=FLAGS.fusion_hidden1,
        #                                  placeholders=self.placeholders,
        #                                  act=tf.nn.relu,
        #                                  dropout=True,
        #                                  bias=True,
        #                                  logging=self.logging))
        #
        # self.fusion_layers.append(Dense1(input_dim=FLAGS.fusion_hidden1,
        #                                  output_dim=FLAGS.fusion_hidden2,
        #                                  placeholders=self.placeholders,
        #                                  act=tf.nn.relu,
        #                                  dropout=True,
        #                                  bias=True,
        #                                  logging=self.logging))

        self.fusion_layers.append(Dense1(input_dim=fusion_input_dim,
                                         output_dim=self.output_dim,
                                         placeholders=self.placeholders,
                                         act=lambda x: x,
                                         dropout=False,
                                         bias=True,
                                         logging=self.logging))

    def _build_tabnet(self, data, is_training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            masked_features = data
            batch_size = tf.shape(masked_features)[0]

            # Initializes decision-step dependent variables.
            output_aggregated = tf.zeros([batch_size, self.output_dim_tabnet])
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complemantary_aggregated_mask_values = tf.ones(
                [batch_size, self.num_features])
            total_entropy = 0

            if is_training:
                v_b = self.virtual_batch_size
            else:
                v_b = 1

            for ni in range(self.num_decision_steps):

                # Feature transformer with two shared and two decision step dependent
                # blocks is used below.

                reuse_flag = (ni > 0)

                transform_f1 = tf.layers.dense(
                    masked_features,
                    self.feature_dim * 2,
                    name="Transform_f1",
                    reuse=reuse_flag,
                    use_bias=False)
                transform_f1 = tf.layers.batch_normalization(
                    transform_f1,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b)
                transform_f1 = glu(transform_f1, self.feature_dim)

                transform_f2 = tf.layers.dense(
                    transform_f1,
                    self.feature_dim * 2,
                    name="Transform_f2",
                    reuse=reuse_flag,
                    use_bias=False)
                transform_f2 = tf.layers.batch_normalization(
                    transform_f2,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b)
                transform_f2 = (glu(transform_f2, self.feature_dim) +
                                transform_f1) * np.sqrt(0.5)

                transform_f3 = tf.layers.dense(
                    transform_f2,
                    self.feature_dim * 2,
                    name="Transform_f3" + str(ni),
                    use_bias=False)
                transform_f3 = tf.layers.batch_normalization(
                    transform_f3,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b)
                transform_f3 = (glu(transform_f3, self.feature_dim) +
                                transform_f2) * np.sqrt(0.5)

                transform_f4 = tf.layers.dense(
                    transform_f3,
                    self.feature_dim * 2,
                    name="Transform_f4" + str(ni),
                    use_bias=False)
                transform_f4 = tf.layers.batch_normalization(
                    transform_f4,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b)
                transform_f4 = (glu(transform_f4, self.feature_dim) +
                                transform_f3) * np.sqrt(0.5)

                if ni > 0:
                    decision_out = tf.nn.relu(transform_f4[:, :self.output_dim_tabnet])

                    # Decision aggregation.
                    output_aggregated += decision_out

                    # Aggregated masks are used for visualization of the
                    # feature importance attributes.
                    scale_agg = tf.reduce_sum(
                        decision_out, axis=1, keep_dims=True) / (
                                        self.num_decision_steps - 1)
                    aggregated_mask_values += mask_values * scale_agg

                features_for_coef = (transform_f4[:, self.output_dim_tabnet:])

                if ni < self.num_decision_steps - 1:
                    # Determines the feature masks via linear and nonlinear
                    # transformations, taking into account of aggregated feature use.
                    mask_values = tf.layers.dense(
                        features_for_coef,
                        self.num_features,
                        name="Transform_coef" + str(ni),
                        use_bias=False)
                    mask_values = tf.layers.batch_normalization(
                        mask_values,
                        training=is_training,
                        momentum=self.batch_momentum,
                        virtual_batch_size=v_b)
                    mask_values *= complemantary_aggregated_mask_values
                    mask_values = tfa.activations.sparsemax(mask_values)

                    # Relaxation factor controls the amount of reuse of features between
                    # different decision blocks and updated with the values of
                    # coefficients.
                    complemantary_aggregated_mask_values *= (
                            self.relaxation_factor - mask_values)

                    # Entropy is used to penalize the amount of sparsity in feature
                    # selection.
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(
                            -mask_values * tf.log(mask_values + self.epsilon),
                            axis=1)) / (
                                             self.num_decision_steps - 1)

                    # Feature selection.
                    masked_features = tf.multiply(mask_values, data)

            return output_aggregated

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        """Embedding"""
        self.mlp_inputs = []
        for name, dim in self.d_feature_dim:
            self.mlp_inputs.append(self.embedding_layers[name](self.mlp_inputs_d[name], 0))
        self.mlp_inputs.append(self.mlp_inputs_c)
        if self.batch_size == 1:
            self.mlp_inputs = tf.concat([self.mlp_inputs[0]], axis=1)
            self.is_training = False
        else:
            self.mlp_inputs = tf.concat(self.mlp_inputs, axis=1)
            self.is_training = True
        self.mlp_inputs = tf.concat([self.mlp_inputs], axis=1)

        # TabNet
        self.mlp_output = self._build_tabnet(self.mlp_inputs, is_training=self.is_training)
        # if not self.reuse:
        #     self.reuse = True

        for i in range(self.batch_size):
            # Build sequential layer model
            """GCN"""
            self.activations = [self.gcn_inputs[i]]
            self.gcn_outputs = []
            for layer in self.gcn_layers:
                hidden = layer(self.activations[-1], i)
                self.activations.append(hidden)
                if len(self.gcn_outputs) < self.num_graphs:
                    self.gcn_outputs.append(hidden)
                if len(self.activations)-1 == self.num_graphs:
                    gcn_output = tf.stack(self.gcn_outputs, axis=1)             # 除掉输入
                    gcn_output = tf.reshape(gcn_output, [1, self.num_graphs, self.num_nodes, FLAGS.gcn_hidden])
                    hidden = tf.nn.max_pool(gcn_output,
                                            ksize=[1, self.num_graphs, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID')
                    hidden = tf.reshape(hidden, [self.num_nodes, FLAGS.gcn_hidden])
                    self.activations.append(hidden)
            self.gcn_output = self.activations[-1]
            self.gcn_output = tf.reshape(self.gcn_output, [1, FLAGS.gcn_dense])
            # self.gcn_output = self.gcn_layers[-1](self.gcn_output, 0)

            self.fusion_inputs = tf.concat([self.gcn_output, tf.reshape(self.mlp_output[i], [1, -1])], axis=1)

            """Fusion"""
            self.fusion_outputs = [self.fusion_inputs]
            for layer in self.fusion_layers:
                hidden = layer(self.fusion_outputs[-1], i)
                self.fusion_outputs.append(hidden)
            self.outputs.append(self.predict(self.fusion_outputs[-1]))

        self.outputs = tf.concat(self.outputs, axis=0)

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self, output):
        return tf.nn.softplus(output)
