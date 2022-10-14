import tensorflow as tf
from basic_ops import *


"""This script defines 2D and 3D multihead self-attention layers.
"""


def self_attention(inputs, total_key_filters, total_value_filters, output_filters,
        num_heads, training, dimension, layer_type, name, dropout_rate=0.0, use_softmax=True,
        use_bias=True, batch_att=False):
    """Multihead scaled-dot-product attention with input/output transformations.
    
    Args:
        inputs: a Tensor with shape [batch, (d,) h, w, channels]
        total_key_filters: an integer. Note that queries have the same number 
            of channels as keys.
        total_value_filters: an integer
        output_filters: an integer
        num_heads: an integer dividing total_key_filters and total_value_filters
        training: a boolean for dropout
        dimension: a string, dimension of inputs/outputs -- 2D, 3D
        layer_type: a string, type of this layer -- SAME, DOWN, UP
        name: a string
        dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
        use_softmax: a boolean deciding whether to use softmax
        use_bias: a boolean deciding whether to use the bias term in input/output transformations

    Returns:
        A Tensor of shape [batch, (_d,) _h, _w, output_filters]
    
    Raises:
        ValueError: if the total_key_filters or total_value_filters are not divisible
            by the number of attention heads.
        ValueError: if dimension is not one of ['2D', '3D'].
        ValueError: if layer_type is not one of ['SAME', 'DOWN', 'UP'].
    """
    if total_key_filters % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                        "attention heads (%d)." % (total_key_filters, num_heads))
    if total_value_filters % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                        "attention heads (%d)." % (total_value_filters, num_heads))
    if layer_type not in ['SAME', 'DOWN', 'UP', 'UP4']:
        raise ValueError("Layer type (%s) must be one of SAME, DOWN, UP." % (layer_type))

    if dimension == '2D' and not batch_att:
        compute_qkv = compute_qkv_2d
        split_heads = split_heads_2d
        unfold = unfold_2d
        combine_heads = combine_heads_2d
        output_transfrom = convolution_2D
    elif dimension == '2D' and batch_att:
        compute_qkv = compute_qkv_2d
        split_heads = split_heads_3d
        unfold = unfold_3d
        combine_heads = combine_heads_3d
        output_transfrom = convolution_2D
    elif dimension == '3D':
        compute_qkv = compute_qkv_3d
        split_heads = split_heads_3d
        unfold = unfold_3d
        combine_heads = combine_heads_3d
        output_transfrom = convolution_3D
    else:
        raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # produce q, k, v
        q, k, v = compute_qkv(inputs, total_key_filters, total_value_filters, use_bias,
                    layer_type, batch_att) # [batch, h, w, channels]
        if batch_att:
                q = tf.expand_dims(q, 0) # [1, batch, h, w, channels]
                k = tf.expand_dims(k, 0)
                v = tf.expand_dims(v, 0)

        # after splitting, shape is [batch, heads, d, h, w, channels / heads]
        q_split = split_heads(q, num_heads) # [1, heads, batch, h, w, channels / heads]
        k_split = split_heads(k, num_heads)
        v_split = split_heads(v, num_heads)

        # normalization recommended by "Attention is All You Need"
        key_filters_per_head = total_key_filters // num_heads
        q_split *= key_filters_per_head**-0.5

        output_shape = tf.concat([tf.shape(q_split)[0:-1], [v_split.shape[-1].value]], 0)

        # flatten q,k,v
        q_new = unfold(q_split)
        k_new = unfold(k_split)
        v_new = unfold(v_split)

        # attention
        o = dot_product_attention(q_new, k_new, v_new, training, dropout_rate, use_softmax)

        # putting the representations back in the right place
        o = tf.reshape(o, output_shape)

        # combine heads and perform output transformation
        o = combine_heads(o)
        if batch_att:
                o = tf.squeeze(o, axis=0)
                q = tf.squeeze(q, axis=0)

        o = output_transfrom(o, output_filters, 1, 1, use_bias, 'out_transform')

        return o, q, k
    

def get_x_lib(inputs):
    # [batch, h, w, channels]
    weights = convolution_2D(inputs, 128, 1, 1, False, 'klib') #[batch, h, w, 64]
    weights = tf.nn.sigmoid(weights)
    weights = weights/tf.reduce_sum(weights, axis=[0,1,2], keepdims=True)
    weights = tf.expand_dims(weights, 3) #[batch, h, w, 1, 64]
    inputs = tf.expand_dims(inputs, 4) #[batch, h, w, channels, 1]
    xlib = tf.reduce_sum(weights*inputs, axis=[0,1,2]) #[batch, h, w, channels, 64] --> [channels, 64]
    xlib = tf.expand_dims(tf.transpose(xlib, [1,0]), 0) #[1, 64, channels]
    return xlib


def compute_qkv_2d(inputs, total_key_filters, total_value_filters, use_bias, layer_type, batch_att):
    """Perform qkv transformations and compute query, key and value.

    Args:
        inputs: a Tensor with shape [batch, h, w, channels]
        total_key_filters: an integer
        total_value_filters: an integer
        use_bias: a boolean deciding whether to use the bias term in qkv transformations
        layer_type: a string, type of this layer -- SAME, DOWN, UP
    
    Returns:
        q: a Tensor with shape [batch, _h, _w, total_key_filters]
        k: a Tensor with shape [batch, h, w, total_key_filters]
        v: a Tensor with shape [batch, h, w, total_value_filters]
    """
    # transformation for q
    if layer_type == 'SAME':
        q = convolution_2D(inputs, total_key_filters, 1, 1, use_bias, 'q_transform')
#         q = tf.get_variable('query', [1, 64, 64, total_key_filters])
#         q = tf.tile(q, [tf.shape(inputs)[0], 1, 1, 1])
    elif layer_type == 'DOWN':
        q = convolution_2D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
    elif layer_type == 'UP':
        q = transposed_convolution_2D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
#       q = convolution_2D(inputs, total_key_filters, 1, 1, False, 'q_transform')
#       q = tf.image.resize_nearest_neighbor(q, tf.concat([tf.shape(inputs)[1:2]*2, tf.shape(inputs)[2:3]*2],0))

    k = convolution_2D(inputs, total_key_filters, 1, 1, use_bias, 'k_transform')
    v = convolution_2D(inputs, total_value_filters, 1, 1, use_bias, 'v_transform')
#         diff = tf.reduce_mean(xlibrary-library, axis=[0,1])
        
#   k = convolution_2D(inputs, total_key_filters, 1, 1, use_bias, 'k_transform')
#   v = convolution_2D(inputs, total_value_filters, 1, 1, use_bias, 'v_transform')

    return q, k, v


def compute_qkv_3d(inputs, total_key_filters, total_value_filters, use_bias, layer_type):
    """Perform qkv transformations and compute query, key and value.

    Args:
        inputs: a Tensor with shape [batch, d, h, w, channels]
        total_key_filters: an integer
        total_value_filters: an integer
        use_bias: a boolean deciding whether to use the bias term in qkv transformations
        layer_type: a string, type of this layer -- SAME, DOWN, UP
    
    Returns:
        q: a Tensor with shape [batch, _d, _h, _w, total_key_filters]
        k: a Tensor with shape [batch, d, h, w, total_key_filters]
        v: a Tensor with shape [batch, d, h, w, total_value_filters]
    """
    # transformation for q
    if layer_type == 'SAME':
        q = convolution_3D(inputs, total_key_filters, 1, 1, use_bias, 'q_transform')
    elif layer_type == 'DOWN':
        q = convolution_3D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
    elif layer_type == 'UP':
        q = transposed_convolution_3D(inputs, total_key_filters, 3, 2, use_bias, 'q_transform')
    elif layer_type == 'UP4':
        q = tf.reshape(inputs, tf.concat([tf.shape(inputs)[0:1]*tf.shape(inputs)[1:2], tf.shape(inputs)[2:]],0))
        q = tf.image.resize_nearest_neighbor(q, tf.concat([tf.shape(inputs)[2:3]*4, tf.shape(inputs)[3:4]*4],0))
        q = tf.reshape(q, tf.concat([tf.shape(inputs)[:2], tf.shape(q)[1:]], 0))

    # linear transformation for k
    k = convolution_3D(inputs, total_key_filters, 1, 1, use_bias, 'k_transform')

    # linear transformation for v
    v = convolution_3D(inputs, total_value_filters, 1, 1, use_bias, 'v_transform')

    return q, k, v


def reshape_range(tensor, i, j, shape):
    """Reshapes a tensor between dimensions [i,j)."""
    target_shape = tf.concat(
            [tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
            axis=0)
    return tf.reshape(tensor, target_shape)


def unfold_2d(x):
    x_shape = tf.shape(x)
    # [batch, heads, length, channels], length = h*w
    x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])
    return x


def unfold_3d(x):
    x_shape = tf.shape(x)
    # [batch, heads, length, channels], length = d*h*w
    x = reshape_range(x, 2, 5, [tf.reduce_prod(x_shape[2:5])])
    return x


def dot_product_attention(q, k, v, training, dropout_rate, use_softmax):
    """Dot-product attention.

    Args:
        q: a Tensor with shape [batch, heads, length_q, channels_k]
        k: a Tensor with shape [batch, heads, length_kv, channels_k]
        v: a Tensor with shape [batch, heads, length_kv, channels_v]
        training: a boolean for dropout
        dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
        use_softmax: a boolean deciding whether to use softmax

    Returns:
        A Tensor with shape [batch, heads, length_q, channels_v]
    """
    # normalize attention
    if use_softmax:
        # [batch, num_heads, length_q, length_kv]
        attention_weights = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_weights, name='softmax')
        # dropping out the attention links for each of the heads
        if dropout_rate != 0.0:
            attention_weights = tf.layers.dropout(attention_weights, dropout_rate, training)
        atted = tf.matmul(attention_weights, v)

    else:
        kv = tf.matmul(k, v, transpose_a=True)
        atted = tf.matmul(q, kv)/tf.cast(tf.shape(k)[2], tf.float32)
#       reduced_v = tf.reduce_sum(v, [2], keepdims=True)
#       p = tf.get_variable(
#               '/weights', [v.shape[1].value, 1, q.shape[3].value],
#               initializer=tf.contrib.layers.xavier_initializer())
#       reduced_q = tf.multiply(q, p)
#       reduced_q = tf.reduce_sum(reduced_q, [-1], keepdims=True)
#       qv = tf.matmul(reduced_q, reduced_v)
#       reduced_k = tf.multiply(k, p)
#       reduced_k = tf.reduce_mean(reduced_k, [-1], keepdims=True)
#       kv = tf.matmul(reduced_k, v, transpose_a=True)
#       atted = tf.add(qv, kv)/tf.cast(tf.shape(q)[2], tf.float32)
    return atted


def split_heads_2d(x, num_heads):
    """Split channels (last dimension) into multiple heads (becomes dimension 1).
    
    Args:
        x: a Tensor with shape [batch, h, w, channels]
        num_heads: an integer
    
    Returns:
        a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def split_heads_3d(x, num_heads):
    """Split channels (last dimension) into multiple heads (becomes dimension 1).
    
    Args:
        x: a Tensor with shape [batch, d, h, w, channels]
        num_heads: an integer
    
    Returns:
        a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 4, 1, 2, 3, 5])


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.

    Args:
        x: a Tensor with shape [..., m]
        n: an integer.

    Returns:
        a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret


def combine_heads_2d(x):
    """Inverse of split_heads_2d.

    Args:
        x: a Tensor with shape [batch, num_heads, h, w, channels / num_heads]

    Returns:
        a Tensor with shape [batch, h, w, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def combine_heads_3d(x):
    """Inverse of split_heads_3d.

    Args:
        x: a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]

    Returns:
        a Tensor with shape [batch, d, h, w, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 4, 1, 5]))


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.

    Args:
        x: a Tensor with shape [..., a, b]

    Returns:
        a Tensor with shape [..., a*b]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret
