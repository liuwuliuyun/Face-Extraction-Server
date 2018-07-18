import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework
import tensorflow as tf


def arg_scope(is_training, wd):
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected, layers.separable_conv2d], activation_fn=None,
        weights_initializer=layers.variance_scaling_initializer(mode='FAN_OUT'),
        weights_regularizer=layers.l2_regularizer(wd)):
        with framework.arg_scope(
            [layers.conv2d, layers.separable_conv2d], biases_initializer=None):
            with framework.arg_scope(
                [layers.batch_norm], is_training=is_training,
                scale=True, decay=0.9, epsilon=2e-5) as scope:
                    return scope

def prelu(x, name='prelu'):
    with tf.variable_scope(name):
        alpha = tf.get_variable(
            'alpha', x.get_shape()[-1],
            initializer=tf.constant_initializer(0.25))
        return tf.maximum(x, 0.0) + alpha * tf.minimum(x, 0.0)


def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None):
    assert stride[0] == stride[1]
    stride = stride[0]
    assert (kernel == (3, 3) and pad == (1, 1)) or (kernel == (1, 1) and pad == (0, 0))
    pad = 'SAME'
    assert num_group == 1 or num_group == data.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        if num_group == 1:
            y = layers.conv2d(data, num_filter, kernel, stride, 'SAME')
        else:
            y = layers.separable_conv2d(data, None, kernel, 1, stride, 'SAME')
        y = layers.batch_norm(y)
        y = prelu(y)
        return y


def network(x, is_training=False, embedding_size=512, wd=7.5e-4, keep_prob=0.6, reduce_dim=True, gap=False, gmp=False):
    with tf.variable_scope('extractor//mobilenet', reuse=tf.AUTO_REUSE):
        with framework.arg_scope(arg_scope(is_training, wd)):
            candidates = []
            x = (x - 127.5) *  0.0078125  
            #x = x [:, : ,: ,::-1]
            conv_1 = Conv(x, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_1") 
            conv_2_dw = Conv(conv_1, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_2_dw") # 112/112
            conv_2 = Conv(conv_2_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_2") # 112/112
            conv_3_dw = Conv(conv_2, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_3_dw") # 112/56
            conv_3 = Conv(conv_3_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_3") # 56/56
            candidates.append(conv_3)
            conv_4_dw = Conv(conv_3, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_4_dw") # 56/56
            conv_4 = Conv(conv_4_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_4") # 56/56
            conv_5_dw = Conv(conv_4, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_5_dw") # 56/28
            conv_5 = Conv(conv_5_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_5") # 28/28
            conv_6_dw = Conv(conv_5, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_6_dw") # 28/28
            candidates.append(conv_6_dw)
            conv_6 = Conv(conv_6_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6") # 28/28
            conv_7_dw = Conv(conv_6, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_7_dw") # 28/14
            conv_7 = Conv(conv_7_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_7") # 14/14
            conv_8_dw = Conv(conv_7, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_8_dw") # 14/14
            conv_8 = Conv(conv_8_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_8") # 14/14
            candidates.append(conv_8)
            conv_9_dw = Conv(conv_8, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_9_dw") # 14/14
            conv_9 = Conv(conv_9_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_9") # 14/14
            conv_10_dw = Conv(conv_9, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_10_dw") # 14/14
            conv_10 = Conv(conv_10_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_10") # 14/14
            conv_11_dw = Conv(conv_10, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_11_dw") # 14/14
            candidates.append(conv_11_dw)
            conv_11 = Conv(conv_11_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_11") # 14/14
            conv_12_dw = Conv(conv_11, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_12_dw") # 14/14
            conv_12 = Conv(conv_12_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_12") # 14/14
            conv_13_dw = Conv(conv_12, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_13_dw") # 14/7
            candidates.append(conv_13_dw)
            conv_13 = Conv(conv_13_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_13") # 7/7
            conv_14_dw = Conv(conv_13, num_group=1024, num_filter=1024, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv_14_dw") # 7/7
            conv_14 = Conv(conv_14_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7/7 
            if reduce_dim:
                conv_14 = Conv(conv_14, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='reducedim')
            var = None
            if gap or gmp:
                mean, var = tf.nn.moments(conv_14, axes=[1, 2])
                var = tf.reduce_mean(conv_14, axis=1)
            if gap:
                conv_14 = tf.reduce_mean(conv_14, axis=[1,2])
            if gmp:
                conv_14 = tf.reduce_max(conv_14, axis=[1,2])
            y = layers.batch_norm(conv_14)
            y = layers.dropout(y, keep_prob, is_training=is_training)
            y = layers.flatten(y)
            y = layers.fully_connected(y, embedding_size)
            y = layers.batch_norm(y, scale=False)
            for candidate in candidates:
                tf.add_to_collection('checkpoints', candidate)
                return tf.nn.l2_normalize(y, 1) 


class extractor:
    def __init__(self, session, devices, batch_size):
        self.session = session
        assert batch_size % len(devices) == 0
        batch_size_per_device = batch_size // len(devices)
        assert len(devices) == 1
        device = devices[0]
        with tf.name_scope('extractor'):
            with tf.device('/cpu:0'):
                self.images = tf.placeholder(
                    tf.float32, (None, 112, 112, 3), 'images')
                embeddings = []
            with tf.device(device):
                local_embeddings = network(self.images)
                embeddings.append(local_embeddings)
            with tf.device('cpu:0'):
                self.embeddings = embeddings[0] 
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='extractor//mobilenet'))
        saver.restore(session, '/root/server/yshenFace/faceExtractor/checkpoints/extractor')
        #output_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [self.embeddings.name[:-2]])
        #tflite_model = tf.contrib.lite.toco_convert(output_graph_def, [self.images], [self.embeddings])
        #open("converteds_model.tflite", "wb").write(tflite_model)
    def extract(self, images):
        e = self.session.run(self.embeddings, feed_dict={self.images: images})
        return [i for i in e]

        
