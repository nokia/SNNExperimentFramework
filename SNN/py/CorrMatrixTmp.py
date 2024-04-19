import numpy as np
import tensorflow as tf

epsilon=1e-7

y = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])

p = np.array([
        [0.7, 0.28, 0.02, 0.0],
        [0.39, 0.6, 0.01, 0.0],
        [0.025, 0.05, 0.9, 0.025],
        [0.0, 0.8, 0.0, 0.2],
        [0.8, 0.2, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0],
        [0.01, 0.02, 0.95, 0.02],
        [0.0, 0.76, 0.0, 0.24],
    ])

def cce(target, output, axis=-1):
    # scale preds so that the class probas of each sample sum to 1
    output = output / tf.reduce_sum(output, axis, True)
    # Compute cross entropy from probabilities.
    epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(target * tf.math.log(output), axis)

v_cce = cce(tf.convert_to_tensor(y), tf.convert_to_tensor(p))
s_cce = tf.reduce_mean(v_cce)

print(v_cce)
print(s_cce)

pi_t = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])

def cce_t(target, output, matrix, axis=-1):
    target = tf.linalg.matmul(target, matrix)
    output = output / tf.reduce_sum(output, axis, True)
    epsilon_ = tf.constant(epsilon, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(target * tf.math.log(output), axis)

v_cce = cce_t(tf.convert_to_tensor(y),
              tf.convert_to_tensor(p),
              tf.convert_to_tensor(pi_t))
s_cce = tf.reduce_mean(v_cce)

print(v_cce)
print(s_cce)

def __get_corr_matrix(flags, lg_pred: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    C = None
    C_nan = None
    for i in range(flags.shape[1]):
        j_tensor = None
        for j in range(flags.shape[1]):
            j_pred = lg_pred[:, j]
            i_oneH = flags[:, i]
            upper = -1*tf.gather(j_pred, tf.where(i_oneH == 1)[:, 0])
            upper = tf.reduce_sum(upper)
            lower = tf.reduce_sum(i_oneH)
            corr = upper/lower
            if upper == 0.0 and lower == 0.0:
                j_tensor = tf.constant(np.nan, shape=[flags.shape[1]])
                if C_nan is None:
                    C_nan = [i]
                else:
                    C_nan.append(i)
                break

            if j_tensor is None:
                j_tensor = tf.expand_dims(corr, 0)
            else:
                j_tensor = tf.concat([j_tensor, tf.expand_dims(corr, 0)], 0)

        if C is None:
            C = tf.expand_dims(j_tensor, 0)
        else:
            C = tf.concat([C, tf.expand_dims(j_tensor, 0)], axis=0)


    print(C)
    max_no_nan = tf.math.reduce_max(tf.where(tf.math.is_nan(C), 0.0, C))
    min_no_nan = tf.math.reduce_min(tf.where(tf.math.is_nan(C), 20.0, C))
    C = tf.where(tf.math.is_nan(C), max_no_nan, C)

    if C_nan is not None:
        mask_assign_min = np.zeros(C.shape)
        mask_assign_min[C_nan, C_nan] = 1
        mask_assign_min = tf.convert_to_tensor(mask_assign_min)
        C = tf.where(mask_assign_min == 1, min_no_nan, C)

    C = tf.map_fn(lambda x: (x-min_no_nan)/(max_no_nan - min_no_nan), C)
    return C

epsilon_ = tf.constant(1e-7, tf.double)
C = __get_corr_matrix(tf.convert_to_tensor(y),
                      tf.math.log(tf.clip_by_value(tf.convert_to_tensor(p), epsilon_, 1.0-epsilon_)))

print(C)

new_pi = tf.linalg.matmul(pi_t, 1-C)

print(new_pi)

new_one_hot = tf.linalg.matmul(y, new_pi)

print(new_one_hot)

v_cce = cce_t(tf.convert_to_tensor(y),
              tf.convert_to_tensor(p),
              new_pi)
s_cce = tf.reduce_mean(v_cce)

print(v_cce)
print(s_cce)
