from data_processing import data_process
import tensorflow as tf
import numpy as np
import os


class deep_coder(object):
    """docstring for deep_coder"""
    def __init__(self):
        self.num_input = 3 #number of inputs(parameters) of a program
        self.embedding_size = 20#Dimension of the integer embedding
        self.integer_max = 256 #range of all possible outputs of a program
        self.integer_min = -255
        self.num_example = 5
        self.max_list_len = 20 #maximum length of a input/output list
        self.num_hidden_layer = 3
        self.hidden_layer_size = 256
        self.attribute_size = 34
        self.batch_size = 1
        self.num_epoch = 100
        self.learning_rate = 0.001
        np.set_printoptions(precision=3)
        

    def build_model(self):
        #data(x)
        #num_input + num_output = num_input + 1 
        #max_list_len + type_vec_len = max_list_len + 2
        self.prog_data = tf.placeholder(tf.int32, shape=[
            None, self.num_example, self.num_input + 1, self.max_list_len + 2],name='input')
        #对于每个program，有num_example的input-output pairs,每个pair最多有
        #num_input个输入和1个输出，每个输入输出的长度为max_list_len + type vec

        #target (y)
        self.attribute = tf.placeholder(tf.float32, shape=[None, self.attribute_size],name='attributes')

        #trainable variable
        #integer embedding 
        self.integer_embedding = tf.Variable(tf.random_normal([
            -self.integer_min + self.integer_max + 1, self.embedding_size]), name='integer_embedding')
        #对于每个integer in integer range，embedding size 大小的vector表示
        #负数？
        
        #main network
        self.encoded = self.encoder(self.prog_data)
        self.decoded = self.decoder(self.encoded)
        

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.decoded, labels=self.attribute), name='loss')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.prediction = tf.sigmoid(self.decoded)
        
        self.summary_writer = tf.summary.FileWriter(logdir='./train/', graph=tf.get_default_graph())
        s1 = tf.summary.histogram('Integer_Embedding', self.integer_embedding)
        s2 = tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge([s1, s2])



    #将value和type分割，并对value进行embedding， 
    #然后连接embedding后的tensor和type，然后输入到三层全连接网络
    def encoder(self, x):
        #split type tensor and value tensor
        #input: [batch_size, num_example, num_input + 1, max_list_len + 2]
        #output:type:[batch_size, num_example, num_input + 1, 2]
        #       value:[batch_size, num_example, num_input + 1, max_list_len]
        types, values = tf.split(x, [2, self.max_list_len], axis=3)
        sess = tf.Session()
        constant_add = tf.constant(
            value=-self.integer_min,shape=[
            self.batch_size,self.num_example,self.num_input+1,self.max_list_len],dtype=tf.int32)

        #output:[batch_size, num_example, num_input + 1, max_list_len, embedding_size]

        values = tf.add(values,constant_add)

        value_embeded = tf.nn.embedding_lookup(self.integer_embedding, values)
        value_embeded_reduced = tf.reshape(value_embeded,
            [-1, self.num_example, self.num_input + 1, self.max_list_len * self.embedding_size])

        types = tf.cast(types, tf.float32)

        #output: [batch_size, num_Example, num_input + 1, max_list_len * embedding_size + 2]
        x_embeded = tf.concat([types, value_embeded_reduced], axis=3)
        output = x_embeded
        #hidden dense layers
        #output: [batch_size, num_example, num_input + 1, hidden_layer_size]
        for i in range(self.num_hidden_layer):
            with tf.variable_scope('layer_{}'.format(i)) as scope:
                output = tf.layers.dense(
                    inputs=output, units=self.hidden_layer_size, activation=tf.sigmoid)
        return output


        #对输入进行平均池化，然后连接一层输出nn
    def decoder(self, encoded):
        #average pooling by reducing in examples
        #input: [batch_size, num_example, num_input+1, hidden_layer_size]
        #output: [batch_size, num_example, (num_input + 1) * hidden_layer_size]
        reduced = tf.reshape(
            encoded, [-1, self.num_example, (self.num_input + 1) * self.hidden_layer_size])

        #output : [batch_size, (num_input + 1) * hidden_layer_size]
        pooled = tf.reduce_mean(reduced, axis=1, name='pooling')
        result = tf.layers.dense(
            inputs=pooled, units=self.attribute_size, activation=tf.sigmoid,name='decoded')
        return result

    def train(self, data, target):
        #ratio of train test split
        split_idx = len(data)//10 * 9
        #shuffle training data, unfortunately shuffle cannot take two arrays
        #a random seed is created randomly and is used to see both shffles
        some_seed = np.random.randint(1000000)
        np.random.seed(some_seed)
        np.random.shuffle(data)
        np.random.seed(some_seed)
        np.random.shuffle(target)

        #generate train and test data
        train_data, train_target = data[:split_idx], target[:split_idx]
        test_data, test_target = data[split_idx:], target[split_idx:]

        #random batch
        def get_batch(d, t):
            idx = np.random.choice(d.shape[0], size=self.batch_size)
            return d[idx], t[idx]

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            feed = {}
            for ep in range(1, self.num_epoch + 1): #101
                for i in range(len(data) // self.batch_size):
                    train_data_batch, train_target_batch = get_batch(train_data, train_target)
                    feed = {self.prog_data:train_data_batch, self.attribute:train_target_batch}
                    sess.run(self.optimizer, feed)
                    loss, summary = sess.run([self.loss, self.summary_op], feed)

                #report loss every epoch
                    if i%10000 == 0:
                        print('epoch: ',ep, 'loss: ', loss)


                #evaluate model every 10 epoches
                if ep%10 == 0:
                    test_data_batch, test_target_batch = get_batch(test_data, test_target)
                    test = sess.run(self.loss, feed_dict={
                        self.prog_data:test_data_batch, self.attribute:test_target_batch
                        })
                    self.summary_writer.add_summary(summary, ep)
                    print('epoch: ',ep, 'Test loss: ',test)
                    print("tensor is \n{}".format(sess.run(self.decoded, feed)))
                    print("attribute is\n {}".format(sess.run(self.attribute, feed)))



                #save model every 20 epoches
                if ep%20 == 0:
                    saver.save(sess, os.path.join('./','model/model'), global_step=ep)
    def predict(self, data):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver = tf.train.import_meta_graph(os.path.join(
                self.save_dir, 'model/model-{}.meta'.format(self.num_epoch)))
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(self.save_dir, 'model/')))
            return sess.run(self.predict_op, feed_dict={self.prog_data:data})



if __name__ == '__main__':
    
    print("data processing is beginning")
    x_dataset, y_dataset = data_process()
    deep_coder_impl = deep_coder()
    deep_coder_impl.build_model()
    deep_coder_impl.train(x_dataset, y_dataset)







  