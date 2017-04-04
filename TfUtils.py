import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np

def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)


def reduce_avg(reduce_target, lengths, dim):
    """
    Args:
        reduce_target : shape(d_0, d_1,..,d_dim, .., d_k)
        lengths : shape(d0, .., d_(dim-1))
        dim : which dimension to average, should be a python number
    """
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1 :
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    if rank_diff!=0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keep_dims=False)
    red_avg = red_sum / (tf.to_float(lengths_reshape) + 1e-30)
    return red_avg


def batch_embed_lookup(embedding, ids):
    '''
        embedding: shape(b_sz, tstp, emb_sz)
        ids : shape(b_sz, k)
    '''
    input_shape = tf.shape(embedding)
    time_steps = input_shape[0]
    def _create_ta(name, dtype):
        return tf.TensorArray(dtype=dtype,
                              size=time_steps,
                              tensor_array_name=name)
    input_ta = _create_ta('input_ta', embedding.dtype)
    fetch_ta = _create_ta('fetch_ta', ids.dtype)
    output_ta = _create_ta('output_ta', embedding.dtype)
    input_ta = input_ta.unstack(embedding)
    fetch_ta = fetch_ta.unstack(ids)

    def loop_body(time, output_ta):
        embed = input_ta.read(time) #shape(tstp, emb_sz) type of float32
        fetch_id = fetch_ta.read(time) #shape(tstp) type of int32
        out_emb = tf.nn.embedding_lookup(embed, fetch_id)
        output_ta = output_ta.write(time, out_emb)

        next_time = time+1
        return next_time, output_ta
    time = tf.constant(0)
    _, output_ta = tf.while_loop(cond=lambda time, *_: time < time_steps,
                  body=loop_body, loop_vars=(time, output_ta),
                  swap_memory=True)
    ret_t = output_ta.stack() #shape(b_sz, tstp, embd_sz)
    return ret_t


def self_attn(inputs, inputs_lengths):
    '''
    Args:
        inputs: shape(b_sz, tstp, rep_sz)
        input_lengths: shape(b_sz,)
    '''

    attn_sz = np.int(inputs.get_shape()[-1])
    inputs_shape = tf.shape(inputs)
    tstp = inputs_shape[1]
    b_sz = inputs_shape[0]
    small_num = -np.Inf

    mask = mkMask(inputs_lengths, tstp)  # shape(b_sz, tstp)
    attn_mask = tf.expand_dims(mask, 1)  # shape(b_sz, 1, tstp)
    attn_mask = tf.tile(attn_mask, [1, tstp, 1])    # shape(b_sz, tstp, tstp)

    attn_matrix_1 = last_dim_linear(inputs, attn_sz, bias=False,
                                    scope='self_Attn_W1')  # shape(b_sz, tstp, attn_sz)
    attn_matrix_2 = last_dim_linear(inputs, attn_sz, bias=False,
                                    scope='self_Attn_W2')  # shape(b_sz, tstp, attn_sz)
    attn_matrix_1 = tf.expand_dims(attn_matrix_1, axis=2)    # shape(b_sz, tstp, 1, attn_sz)
    attn_matrix_2 = tf.expand_dims(attn_matrix_2, axis=1)    # shape(b_sz, 1, tstp, attn_sz)
    attn_matrix_1 = tf.tile(attn_matrix_1, [1, 1, tstp, 1])     # shape(b_sz, tstp, tstp, attn_sz)
    attn_matrix_2 = tf.tile(attn_matrix_2, [1, tstp, 1, 1])     # shape(b_sz, tstp, tstp, attn_sz)

    attn_matrix = tf.tanh(attn_matrix_1+attn_matrix_2)  #shape(b_sz, tstp, tstp, attn_sz)
    attn_logits = last_dim_linear(attn_matrix, 1, bias=False,
                                  scope='self_Attn_V')    #shape(b_sz, tstp, tstp, 1)
    attn_logits = tf.squeeze(attn_logits, [3])  #shape(b_sz, tstp, tstp)
    attn_logits = tf.where(attn_mask, attn_logits, tf.ones_like(attn_logits)*small_num)
    attn_prob = tf.nn.softmax(attn_logits, dim=-1)  # shape(b_sz, tstp, tstp)
    attn_prob = tf.expand_dims(attn_prob, 3)    # shape(b_sz, tstp, tstp, 1)
    attn_input = tf.expand_dims(inputs, 1)  # shape(b_sz, 1, tstp, rep_sz)
    attn_input = tf.tile(attn_input, [1, tstp, 1, 1])   # shape(b_sz, tstp, tstp, rep_sz)

    attn_content = tf.reduce_sum(attn_input * attn_prob,
                                 axis=[2])     # shape(b_sz, tstp, rep_sz)
    return attn_content


def last_dim_linear(inputs, output_size, bias, scope):
    '''
    Args:
        input: shape(b_sz, ..., rep_sz)
        output_size: a scalar, python number
    '''
    bias_start=0.0
    input_shape = tf.shape(inputs)
    out_shape = tf.concat(axis=0, values=[input_shape[:-1], [output_size]])
    input_size = np.int(inputs.get_shape()[-1])
    unbatch_input = tf.reshape(inputs, shape=[-1, input_size])

    unbatch_output = linear(unbatch_input, output_size, bias=bias,
                                            bias_start=bias_start, scope=scope)
    batch_output = tf.reshape(unbatch_output, shape=out_shape)

    return batch_output     # shape(b_sz, ..., output_size)


def seq_loss(logits, label, lengths):
    """
    Args
        logits: shape (b_sz, tstp, c_sz)
        label: shape (b_sz, tstp)
        lengths: shape(b_sz)
    Return
        loss: A scalar tensor, mean error
    """

    loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(  # shape(b_sz, tstp), step level
        logits=logits, labels=label, name='seq_loss')
    loss_avg = reduce_avg(loss_all, lengths, dim=1)    # shape(b_sz) example level
    loss = tf.reduce_mean(loss_avg)
    return loss


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope) as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)
