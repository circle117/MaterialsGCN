import tensorflow._api.v2.compat.v1 as tf


def mean_square_error(preds, labels):
    return tf.reduce_mean(tf.square(preds-labels))


def root_mean_square_error(preds, labels):
    res = tf.reduce_mean(tf.square(preds-labels))
    return tf.pow(res, 0.5)


def mean_absolute_error(preds, labels):
    sum = tf.Variable(0, dtype=tf.float32)
    cnt = tf.constant(len(preds), dtype=tf.float32)
    for i in range(len(preds)):
        sum = sum + tf.reduce_mean(tf.abs(preds[i]-labels[i]))
    # print(sum, cnt, sum/cnt)
    return sum/cnt


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
