import tensorflow as tf


def kl_divergence(p, p_hat):
    inv_p = tf.sub(tf.constant(1., dtype=tf.float32), p)
    inv_p_hat = tf.sub(tf.constant(1., dtype=tf.float32), p_hat)
    log_p = tf.add(logfunc(p, p_hat), logfunc(inv_p, inv_p_hat))
    return log_p


def logfunc(x, x2):
    return tf.mul(x, tf.log(tf.div(x, x2)))
