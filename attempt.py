import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from matplotlib import pyplot as plt
import glob

image_width = 320 #240
image_height = 240 #192

class network:
    def __init__(self, input1, input2, match):
        with arg_scope(resnet_v2.resnet_arg_scope()) as scope:
            with tf.variable_scope("", reuse=tf.AUTO_REUSE) as scope:
                #self.inputs1 = tf.placeholder(tf.float32, shape=(None, 240, 192, 3))
                #self.inputs2 = tf.placeholder(tf.float32, shape=(None, 240, 192, 3))
                #self.match = tf.placeholder(tf.bool, shape=(None))
                self.x1 = tf.scalar_mul(0.003922, tf.cast(input1, dtype=tf.float32))
                self.x2 = tf.scalar_mul(0.003922, tf.cast(input2, dtype=tf.float32))
                self.match = tf.cast(match, dtype=tf.float32)

                out1 = self.side(self.x1)
                scope.reuse_variables()
                out2 = self.side(self.x2)

                self.diff = tf.reshape(tf.subtract(out1, out2), [-1,out2.shape[3]])
                self.dist = tf.norm(self.diff, axis=1, name="get_distance_between_vecs")
                self.margin = 2.0
                self.loss = self.loss_fcn()
                self.acc = self.acc_fcn()

    def fcl(self, input_layer, nodes, name, keep_rate=1.):
        # Pass th#rough to conv_layer. renamed function for easier readability
        layer = self.conv_layer(input_layer, [1, 1, input_layer.shape[3], nodes], name, padding='VALID', stride=1, pooling=False)
#        out = tf.nn.dropout(layer, rate=#drop_rate)
        out = tf.nn.dropout(layer, keep_prob=keep_rate)
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
    def side(self, input):
        net1, end_points1 = resnet_v2.resnet_v2_101(input, None, is_training=True, global_pool=False, output_stride=16)
        shrunk = tf.nn.max_pool(value=net1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
        arranged = tf.reshape(shrunk, shape=[-1, 1, 1, 7 * 5 * 2048], name="arrange_for_fcl") #40*30
        self.fc1 = self.fcl(arranged, 256, "fc1", 0.70)  # 1024
        self.fc2 = self.fcl(self.fc1, 512, "fc2", 0.90)  # 2048
        self.fc3 = self.fcl(self.fc2, 128, "fc3", 1.00)   # 512
        self.norm = tf.nn.l2_normalize(self.fc3, axis=1)
        return self.norm

    def loss_fcn(self):
        self.distance_matching = tf.multiply(self.match, self.dist,name="distance_matching")
        self.match_compliment = tf.subtract(tf.constant(1, dtype=tf.float32), self.match)
        non_match = tf.nn.relu(tf.subtract(tf.constant(self.margin, dtype=tf.float32, name="margin"), self.dist, name="margin_minus_diff"))
        self.distance_unmatched = tf.multiply(non_match, self.match_compliment, name = "distance_unmatched")

        return tf.reduce_sum(self.distance_matching)+ tf.reduce_sum(self.distance_unmatched)

    def acc_fcn(self):
        return tf.summary.scalar("loss", self.loss_fcn())


def _parse_function(example_proto):
    feature = {
        'img_a': tf.FixedLenFeature([], tf.string),
        'img_b': tf.FixedLenFeature([], tf.string),
        'match': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, feature)
    # Convert the image data from string back to the numbers
    image_a = tf.image.decode_jpeg(features['img_a'], channels=3, name="Steve")
    image_b = tf.image.decode_jpeg(features['img_b'], channels=3, name="Greg")
    match = tf.cast(features['match'], tf.int32)

    # Reshape image data into the original shape
    #image_a = tf.reshape(image_a, [image_width, image_height, 3])
    #image_b = tf.reshape(image_b, [image_width, image_height, 3])

    return image_a, image_b, match

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
    n_bins = 50
    bin_max = 2
    dist_diff=[]
    dist_same=[]
    file = open('dist_log.csv','w')
    for i in range(500):
        [mb, dist, x1, x2] = sess.run([net.match, net.dist, net.x1, net.x2])
        for (match, value, left, right) in zip(mb, dist, x1, x2):
            if step=="":
                file.write('%d, %f\n' % (match, value))
            if (match):
                dist_same.append(min(value, bin_max-0.001))
                if (value < 2.):
                   write_img_pair(left, right, value, 'true_pos/', i)
                   continue 
            else:
                dist_diff.append(min(value, bin_max-0.001))
                if (value < 2. and value > 0.):
                    write_img_pair(left, right, value, 'false_pos/', i)
    file.close()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # Fix the range
    axs[0].hist(dist_same, bins=n_bins, range=[0., bin_max])
    axs[0].set_title("Same Image Dist")
    axs[0].set_ylim(top=2000)
    axs[1].hist(dist_diff, bins=n_bins, range=[0., bin_max])
    axs[1].set_title("Diff Image Dist")
    axs[1].set_ylim(top=2000)
    if step == "":
        plt.show()
    else:
        plt.savefig('Upped-'+step+'.png')

# prepare data and tf.session
data_path = glob.glob('datasets/training*.tfrecords')
#data_path = glob.glob('datasets/valid_*.tfrecords')
dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(15)
iterator = dataset.make_initializable_iterator()
[x1, x2, y] = iterator.get_next()

network = network(x1, x2, y)

#with tf.InteractiveSession() as sess:
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    sess.run(iterator.initializer)
    tf_saver = tf.train.Saver(name="saver")
    if tf.train.checkpoint_exists("./model/Final"):
        print("Loading from model")
        tf_saver.restore(sess,'./model/Final')
    else:
        print("Loading from pretrained")
        variables_can_be_restored = tf.train.list_variables("./pretrained_model/")
        list_of_variables = []
        for v in variables_can_be_restored:
            list_of_variables.append(v[0]+":0")

        globals=[]
        for v in tf.global_variables():
            globals.append(v.name)

        reduced_list = list(set(globals).intersection(list_of_variables))
        pretrained_vars = tf.contrib.framework.get_variables_to_restore(include=reduced_list)

        tf_pretrained_saver = tf.train.Saver(pretrained_vars, name="pretrained_saver")
        tf_pretrained_saver.restore(sess, "./pretrained_model/model.ckpt-30452")

    writer = tf.summary.FileWriter("log/", sess.graph)

    N = 50000
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(network.loss)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in range(N):
        _, loss_v = sess.run([train_step, network.loss])
        if step % 100 == 0:
            #  print(str(step) + ", " +str(loss_v))
            ll = sess.run(network.acc)
            writer.add_summary(ll, step)
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            tf_saver.save(sess, 'model/Final')
            quit()
        if step % 20000 == 0:
            tf_saver.save(sess, 'model/intermediate', global_step=step)
        if step % 5000 == 0:
            histogram(sess, network, dataset, str(step))
    writer.close()
    tf_saver.save(sess, 'model/Final')
    histogram(sess, network, dataset)
    print("Fin")
