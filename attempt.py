import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope

def fcl(input, nodes, name):
    # Pass through to conv_layer. renamed function for easier readability
    return conv_layer(input, [1, 1, input.shape[3], nodes], name , padding='VALID', stride=1, pooling=False)

def conv_layer(input_layer, weights, name, padding, stride=1, pooling=True):
    # with tf.variable_scope(name) as scope:
    kernel = tf.get_variable(name+"_kernel", shape=weights, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    conv = conv2d(input_layer, kernel, padding, stride)
    init = tf.constant(1., shape=[weights[-1]], dtype=tf.float32)
    bias = tf.get_variable(name+"_bias",  dtype=tf.float32, initializer=init)
    preactivation = tf.nn.bias_add(conv, bias)
    conv_relu = tf.nn.relu(preactivation, name=name)

    if pooling:
        out = create_max_pool_layer(conv_relu)
    else:
        out = conv_relu
    return out


def conv2d(input_layer, W, pad, stride=1):
    return tf.nn.conv2d(input=input_layer,
                        filter=W,
                        strides=[1, stride, stride, 1],
                        padding=pad)

def create_max_pool_layer(input):
    return tf.nn.max_pool(value=input,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
def network(input):
    net1, end_points1 = resnet_v2.resnet_v2_101(input, None, is_training=True, global_pool=False, output_stride=16)
    arranged = tf.reshape(net1, shape=[-1, 1, 1, 3 * 4 * 2048])
    fc1 = fcl(arranged, 1024, "fc1")
    fc2 = fcl(fc1, 2048, "fc2")
    fc3 = fcl(fc2, 512, "fc3")
    return fc3

with arg_scope(resnet_v2.resnet_arg_scope()):
    with tf.variable_scope("core", reuse=tf.AUTO_REUSE) as scope:
        inputs1 = tf.placeholder(tf.float32, shape = (None,48, 64, 3))
        inputs2 = tf.placeholder(tf.float32, shape = (None,48, 64, 3))

        out1 = network(inputs1)
        scope.reuse_variables()
        out2 = network(inputs2)

        diff = tf.norm(tf.subtract(out1, out2))

saver = tf.train.Saver()
with tf.Session() as sess:
    #variables_can_be_restored = list(set(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)).intersection(tf.train.list_variables("./model/")))
    #variables_can_be_restored = list(tf.train.list_variables("./model/"))
    g = tf.Graph()
    variables_can_be_restored = tf.train.list_variables("./model/")
    vars = []
    for v in variables_can_be_restored:
        name = v[0]
        var = g.get_tensor_by_name(name)
        vars.append(var)
    #for v in variables_can_be_restored: # tf.global_variables():
    #    print (v)
    temp_saver = tf.train.Saver(vars)
    """
    ckpt_state = tf.train.get_checkpoint_state('./model/')
    print('Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    temp_saver.restore(sess, ckpt_state.model_checkpoint_path)"""

    writer = tf.summary.FileWriter("log/", sess.graph)
    writer.close()
    tf.initialize_all_variables().run()
    test1 = np.ones([1, 48, 64, 3])
    test2 = np.zeros([1, 48, 64, 3])
    test = sess.run(diff,feed_dict={inputs1:test1, inputs2:test2})
    print(test)
    print("Fin")
