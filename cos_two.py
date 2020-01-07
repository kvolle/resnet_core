import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from matplotlib import pyplot as plt
import glob

image_width = 320 #240
image_height = 240 #192

class network:
    def __init__(self, input1, input2, match, na, nb, s):
        with arg_scope(resnet_v2.resnet_arg_scope()) as scope:
            with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
                #self.inputs1 = tf.placeholder(tf.float32, shape=(None, 240, 192, 3))
                #self.inputs2 = tf.placeholder(tf.float32, shape=(None, 240, 192, 3))
                #self.match = tf.placeholder(tf.bool, shape=(None))
                self.x1 = tf.scalar_mul(0.003922, tf.cast(input1, dtype=tf.float32))
                self.x2 = tf.scalar_mul(0.003922, tf.cast(input2, dtype=tf.float32))
                self.match = tf.reshape(tf.cast(match, dtype=tf.float32),shape=[-1, 1],name="reshape")
                
                self.num_a = na
                self.num_b = nb
                self.datasets = s

                self.out1 = self.side(self.x1)
                scope.reuse_variables()
                self.out2 = self.side(self.x2)
                self.len_a = tf.norm(self.out1, axis=1)
                self.len_b = tf.norm(self.out2, axis=1)

                self.a_norm = tf.nn.l2_normalize(self.out1, axis=1)
                self.b_norm = tf.nn.l2_normalize(self.out2, axis=1)
#                self.dot = tf.reduce_sum (tf.multiply(self.a_norm, self.b_norm, name="inner_prod_mult"), 1, keep_dims=True, name="inner_prod_sum")
                self.dot = tf.reduce_sum (tf.multiply(self.a_norm, self.b_norm, name="inner_prod_mult"), 1, keep_dims=True, name="inner_prod_sum")
                self.diff = tf.subtract(self.out1, self.out2)
                self.dist = tf.norm(self.diff, axis=1, name="get_distance_between_vecs")
                self.margin = 10.0
                self.pos_fcn = self.cos_pos()  # self.l2_pos()
                self.neg_fcn = self.cos_neg()  # self.l2_neg()
                self.loss = self.loss_fcn()
                self.acc = self.acc_fcn()
                self.hist = self.hist_summary()

    def fcl(self, input_layer, nodes, in_name, keep_rate=1., norm=False):
        # Pass th#rough to conv_layer. renamed function for easier readability
        layer = self.conv_layer(input_layer, [1, 1, input_layer.shape[3], nodes], in_name, padding='VALID', stride=1, pooling=False)
        #out = tf.nn.dropout(layer, keep_prob=keep_rate)
        if norm:
            relu = tf.nn.relu(layer, in_name+"_relu")
            out = tf.layers.batch_normalization(relu, name=in_name+"_batch")
        else:
            out = layer
        return out

    def conv_layer(self, input_layer, weights, name, padding, stride=1, pooling=True):
        # with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name+"_kernel", shape=weights, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        conv = self.conv2d(input_layer, kernel, padding, stride)
        init = tf.constant(1., shape=[weights[-1]], dtype=tf.float32)
        bias = tf.get_variable(name+"_bias",  dtype=tf.float32, initializer=init)
        preactivation = tf.nn.bias_add(conv, bias, name=name+"_bias_add")
        conv_relu = tf.nn.relu(preactivation, name=name)

        if pooling:
            out = self.create_max_pool_layer(conv_relu)
        else:
            out = conv_relu
        return out


    def conv2d(self, input_layer, W, pad, stride=1):
        return tf.nn.conv2d(input=input_layer,
                            filter=W,
                            strides=[1, stride, stride, 1],
                            padding=pad)

    def create_max_pool_layer(self, input):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
    def side(self, input_layer):
        net1, end_points1 = resnet_v2.resnet_v2_50(input_layer, None, is_training=True, global_pool=False, output_stride=16)
        #net1, end_points1 = resnet_v2.resnet_v2_101(input, None, is_training=False, global_pool=False, output_stride=16)
        self.arranged = tf.reshape(net1, shape=[-1, 1, 1, 16*2048], name="arrange_for_fcl") 
        self.fc1 = self.fcl(self.arranged, 3072, "fc1", 1.0, True)#0.70)  # 1024
        self.fc2 = self.fcl(self.fc1, 1536, "fc2", 1.0, True)#0.90)  # 2048
        self.fc3 = self.fcl(self.fc2, 770, "fc3", 1.00, False)   # 512
        self.out_reshape = tf.reshape(self.fc3, shape=[-1, 770], name="arrange_for_norm")
        return self.out_reshape

    def hist_summary(self):
        return [tf.summary.histogram("arranged", self.arranged), tf.summary.histogram("Layer1", self.fc1), tf.summary.histogram("Layer2", self.fc2), tf.summary.histogram("Layer3", self.fc3)]
        #return [tf.summary.histogram("Layer1", self.out_1), tf.summary.histogram("Layer2", self.out_2), tf.summary.histogram("Layer3", self.out_3), tf.summary.histogram("Layer4", self.out_4), tf.summary.histogram("Layer5", self.out_5), tf.summary.histogram("Layer6", self.out_6), tf.summary.histogram("Layer7", self.out_7), tf.summary.histogram("Layer8", self.out_8)]

    def loss_fcn(self):
        return self.pos_fcn + self.neg_fcn

    def l2_pos(self):
        self.distance_matching = tf.multiply(self.match, self.dist,name="distance_matching")
        return tf.reduce_sum(self.distance_matching)

    def l2_neg(self):
        self.match_compliment = tf.subtract(tf.constant(1, dtype=tf.float32), self.match)
        non_match = tf.nn.relu(tf.subtract(tf.constant(self.margin, dtype=tf.float32, name="margin"), self.dist, name="margin_minus_diff"))
        self.distance_unmatched = tf.multiply(non_match, self.match_compliment, name = "distance_unmatched")
        return tf.reduce_sum(self.distance_unmatched)

    def cos_pos(self):
        self.distance_matching_cos = tf.multiply(self.match, tf.subtract(tf.constant(1, dtype=tf.float32, name="one_minus_cos"), self.dot), name="cos_matching")
        return tf.reduce_sum(self.distance_matching_cos, name="dist_match_sum")

    def cos_neg(self):
        self.match_compliment = tf.subtract(tf.constant(1, dtype=tf.float32,name="comp_const"), self.match, name="match_compliment")
        self.distance_unmatched_cos = tf.multiply(self.match_compliment, self.dot, name="cos_non")
        return tf.reduce_sum(self.distance_unmatched_cos, name="dist_unmatch_sum")

#    def cosine_loss(self):
#        mag = tf.multiply(self.a_len, self.b_len, name="magnitudes")
#        cos = tf.divide(self.dot, mag, "cosine_dist")
#        return self.cos_pos() + self.cos_neg()

    def acc_fcn(self):
        return [tf.summary.scalar("loss", self.loss_fcn()), tf.summary.scalar("Pos_loss", self.pos_fcn), tf.summary.scalar("Neg_loss", self.neg_fcn), tf.summary.scalar("Match", tf.reduce_sum(self.match))]


def _parse_function(example_proto):
    feature = {
        'img_a': tf.FixedLenFeature([], tf.string),
        'img_b': tf.FixedLenFeature([], tf.string),
        'num_a': tf.FixedLenFeature([], tf.int64),
        'num_b': tf.FixedLenFeature([], tf.int64),
        'set'  : tf.FixedLenFeature([], tf.string),
        'match': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, feature)
    # Convert the image data from string back to the numbers
    image_a = tf.image.decode_jpeg(features['img_a'], channels=3, name="Steve")
    image_b = tf.image.decode_jpeg(features['img_b'], channels=3, name="Greg")
    num_a = tf.cast(features['num_a'], tf.int32)
    num_b = tf.cast(features['num_b'], tf.int32)
    dataset = features['set']
    match = tf.cast(features['match'], tf.int32)

    # Reshape image data into the original shape
    #image_a = tf.reshape(image_a, [image_width, image_height, 3])
    #image_b = tf.reshape(image_b, [image_width, image_height, 3])

    return image_a, image_b, num_a, num_b, dataset, match

def write_img_pair(left, right, value, folder, i):
    left_cast = tf.cast(tf.scalar_mul(255., tf.cast(left, dtype=tf.float32)), dtype=tf.uint8)
    rite_cast = tf.cast(tf.scalar_mul(255., tf.cast(right, dtype=tf.float32)), dtype=tf.uint8)
    left_image = tf.image.encode_jpeg(left_cast, format='rgb', quality=100)
    rite_image = tf.image.encode_jpeg(rite_cast, format='rgb', quality=100)
    spec = tf.Session()
    [data_left, data_rite] = spec.run([left_image, rite_image])
    with open(folder+'img'+str(i)+'_'+str(value).replace('.','_')+'l.jpg', 'wb') as fd:
        fd.write(data_left)
    with open(folder+'img'+str(i)+'_'+str(value).replace('.','_')+'r.jpg', 'wb') as fd:
        fd.write(data_rite)

def histogram(sess, net, dataset, step=""):
    n_bins = 150 
    bin_max = 750
    dist_diff=[]
    dist_same=[]
    file = open('dist_log'+step+'.csv','w')
    for i in range(150):
        [mb, dist, sets, x1, x2, dots, len_a, len_b] = sess.run([net.match, net.dist, net.datasets, net.num_a, net.num_b, net.dot, net.len_a, net.len_b])
        for (match, value, ds, num_a, num_b, dot, la, lb) in zip(mb, dist, sets, x1, x2, dots, len_a, len_b):
            file.write('%d, %f, %s, %d, %d, %f, %f, %f\n' % (match, value, ds.decode("utf-8"), num_a, num_b, dot, la, lb))
            if (match):
                dist_same.append(min(value, bin_max-0.001))
            else:
                dist_diff.append(min(value, bin_max-0.001))
    file.close()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # Fix the range
    axs[0].hist(dist_same, bins=n_bins, range=[0., bin_max])
    axs[0].set_title("Same Image Dist")
    axs[0].set_ylim(top=2000)
    axs[1].hist(dist_diff, bins=n_bins, range=[0., bin_max])
    axs[1].set_title("Diff Image Dist")
    axs[1].set_ylim(top=500)
    if step == "":
        plt.show()
    else:
        #plt.savefig('Upped-'+step+'.png')
        plt.close()

# prepare data and tf.session
data_path = ['datasets/training_gibson_small.tfrecords']
dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.shuffle(buffer_size=180000)
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(25)
iterator = dataset.make_initializable_iterator()
[x1, x2, na, nb, s, y] = iterator.get_next()

network = network(x1, x2, y, na, nb, s)

#with tf.InteractiveSession() as sess:
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    #tf.initialize_all_variables().run()
    sess.run(iterator.initializer)
    tf_saver = tf.train.Saver(name="saver")
    """
    if tf.train.checkpoint_exists("./model/Final_NOPE"):
        print("Loading from model")
        tf_saver.restore(sess,'./model/Final')
        list_after_load = [v.name for v in tf.global_variables()]
        print(list_after_load)

    else:
        print("Loading from pretrained")
        variables_can_be_restored = tf.train.list_variables("./pretrained_model/resnet_50/")#50/")
        list_of_variables = []
        for v in variables_can_be_restored:
            list_of_variables.append(v[0]+":0")
            print(v[0])
        global_vars=[]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for v in tf.global_variables():
            global_vars.append(v.name)
            print(v.name)
        reduced_list = list(set(global_vars).intersection(list_of_variables))
        pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=reduced_list)
        tf_pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
        tf_pretrained_saver.restore(sess, "./pretrained_model/resnet_50/model.ckpt-225207")
        #tf_pretrained_saver.restore(sess, "./pretrained_model/model.ckpt-30452")
    #    pass
    """
    writer = tf.summary.FileWriter("log/", sess.graph)

    start = 00
    N = 10000 
    batch_n = 00 # 1500 # Used to lock the normalization values
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(network.loss)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    for step in range(start, N):
        _, loss_v = sess.run([train_step, network.loss])
        if step % 50 == 0:
            #  print(str(step) + ", " +str(loss_v))
            [ll, lp, ln, m] = sess.run(network.acc)
            writer.add_summary(ll, step)
            writer.add_summary(lp, step)
            writer.add_summary(ln, step)
            writer.add_summary(m, step)
            [h1, h2, h3, h4] = sess.run(network.hist)
            writer.add_summary(h1, step)
            writer.add_summary(h2, step)
            writer.add_summary(h3, step)
            writer.add_summary(h4, step)
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            tf_saver.save(sess, 'model/Final')
            quit()
        if step % 20000 == 0:
            tf_saver.save(sess, 'model/intermediate', global_step=step)
        if step % 2000 == 0:
            histogram(sess, network, dataset, str(step))
    writer.close()
    for i in range(batch_n):
        _ = sess.run([update_op])
    tf_saver.save(sess, 'model/Final')
    histogram(sess, network, dataset, str(N))
    list_after_train = [v.name for v in tf.global_variables()]
    print(list_after_train)
    print("Fin")
