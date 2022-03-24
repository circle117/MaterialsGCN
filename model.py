import platform

from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


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

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
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

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, num_nodes, num_graphs, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim                                              # 特征数
        self.num_nodes = num_nodes                                              # 节点数
        self.num_graphs = num_graphs                                            # GCN层数
        self.GCN_outputs = []
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]       # 分类数
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Mean square error
        self.loss += mean_absolute_error(self.outputs, self.placeholders['labels'])
            # tf.losses.mean_squared_error(self.outputs, self.placeholders['labels'])
        print('a')

    def _accuracy(self):
        self.accuracy = mean_absolute_error(self.outputs, self.placeholders['labels'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
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
            self.layers.append(Dense1(input_dim=FLAGS.hidden3,
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

        # self.layers.append(MaxPooling(input_dim=FLAGS.hidden2,
        #                               output_dim=self.placeholders['labels'],
        #                               placeholders=self.placeholders,
        #                               act=lambda x: x,
        #                               logging=self.logging))

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
            if (not FLAGS.dense) and len(self.GCN_outputs) < self.num_graphs:
                self.GCN_outputs.append(hidden)
            if (not FLAGS.dense) and len(self.activations)-1 == self.num_graphs:
                GCN_outputs = tf.stack(self.GCN_outputs, axis=1)
                GCN_outputs = tf.reshape(GCN_outputs, [1, self.num_graphs, self.num_nodes, FLAGS.hidden1])
                hidden = tf.nn.max_pool(GCN_outputs,
                                        ksize=[1, self.num_graphs, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID')
                hidden = tf.reshape(hidden, [self.num_nodes, FLAGS.hidden3])
                self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softplus(self.outputs)


class MMGCN(Model):
    def __init__(self, placeholders, input_dim, num_nodes, num_graphs, **kwargs):
        super(MMGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim                                              # 特征数
        self.num_nodes = num_nodes                                              # 节点数
        self.num_graphs = num_graphs                                            # GCN层数
        self.GCN_outputs = []
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]       # 分类数
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Mean square error
        self.loss += mean_absolute_error(self.outputs, self.placeholders['labels'])
        print('a')

    def _accuracy(self):
        self.accuracy = mean_absolute_error(self.outputs, self.placeholders['labels'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
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
            # self.layers.append(Dense1(input_dim=self.num_nodes*FLAGS.hidden3,
            #                           output_dim=FLAGS.hidden4,
            #                           placeholders=self.placeholders,
            #                           act=tf.nn.relu,
            #                           dropout=True,
            #                           bias=True))

            # self.layers.append(Dense1(input_dim=FLAGS.hidden4,
            #                           output_dim=FLAGS.hidden5,
            #                           placeholders=self.placeholders,
            #                           act=tf.nn.relu,
            #                           dropout=False,
            #                           bias=True))
            #
            # self.layers.append(Dense1(input_dim=FLAGS.hidden5,
            #                           output_dim=FLAGS.hidden6,
            #                           placeholders=self.placeholders,
            #                           act=tf.nn.relu,
            #                           dropout=False,
            #                           bias=True))
            #
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
                                      dropout=False,
                                      bias=True))

        # self.layers.append(MaxPooling(input_dim=FLAGS.hidden2,
        #                               output_dim=self.placeholders['labels'],
        #                               placeholders=self.placeholders,
        #                               act=lambda x: x,
        #                               logging=self.logging))

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
            if (not FLAGS.dense) and len(self.GCN_outputs) < self.num_graphs:
                self.GCN_outputs.append(hidden)
            if (not FLAGS.dense) and len(self.activations)-1 == self.num_graphs:
                GCN_outputs = tf.stack(self.GCN_outputs, axis=1)
                GCN_outputs = tf.reshape(GCN_outputs, [1, self.num_graphs, self.num_nodes, FLAGS.hidden1])
                hidden = tf.nn.max_pool(GCN_outputs,
                                        ksize=[1, self.num_graphs, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID')
                hidden = tf.reshape(hidden, [self.num_nodes, FLAGS.hidden3])
                self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softplus(self.outputs)
