import json
import tensorflow as tf
import numpy as np
import random



list_length = 20
integer_max = 256
integer_min = -256
input_number = 3
attribute_length = 34
example_numbers = 5
tpye_length = 2

hidden_units = 256




def data_process():
    f = open('testdataset.json')
    dataset = json.load(f)
    input_examples = []
    output_examples = []
    attributes = []
    for data in dataset:
        input_example,output_example,attribute = each_data_deal(data)
        input_examples.append(input_example)
        output_examples.append(output_example)
        attributes.append(attribute)
    input_examples = np.array(input_examples,dtype=np.int32)
    output_examples = np.array(output_examples,dtype = np.int32)
    attributes = np.array(attributes,dtype = np.int32)
   # print(type(input_examples))
    # print(attributes.shape)
    # print(attributes)
    print(input_examples.shape)
    print(output_examples.shape)

    attributes = attributes.reshape(-1,34)#reshape the 3-Dimensions to 2-Dimensions
    input_examples = input_examples.reshape(-1,input_number*(list_length+tpye_length))
    output_examples = output_examples.reshape(-1,list_length+tpye_length)
    

    print(input_examples.shape)
    print(output_examples.shape)
    # print(attributes.shape)
    # print(attributes)
    return input_examples,output_examples,attributes

def each_data_deal(data):
    ret_input_example = []
    ret_output_example = []
    ret_attribute = []
    example = data["examples"]# example a dicticonary list

    for exa in example:   #every example has five input-output dic, and input may be integer,array or both, output may be integer or array
        input_ = exa["input"] #return a list

        if len(input_)<input_number:
            for i in range(len(input_),input_number):
                input_.append([])


        input_element = input_element_deal(input_) # process each element in input list

        # nparray = np.array(input_element)
        # print(np.shape(nparray))
        output_ = exa["output"]#return a integer or array
        output_shape = np.array(output_)

        output_element = output_element_deal(output_)
        
        ret_input_example.append(input_element)
        ret_output_example.append(output_element)
    ret_attribute.append(data["attribute"])

    return ret_input_example,ret_output_example,ret_attribute# the length of ret_example is two times of ret_attribute
    
def output_element_deal(element):
    re_elements = []
    
    type_ = type_find(element)
    if type_ == [0,0]:
        re_elements.append(element)

    else:
        re_elements.extend(element)
    if len(re_elements)<list_length:
        re_elements.extend([integer_max]*(list_length-len(re_elements)))
    re_elements.extend(type_)
    return re_elements

def input_element_deal(elements):
    re_elements = []
    for element in elements:#deal with each input in inputs 
        type_ = type_find(element)#return the type of this input, encode by one-hot-encoding
        if type_ == [0,0]:#for the input type which is integer, convert integer into list with one element
            element = [element]
        if type_ == [0,1]:# [0,1] means the input type is function
            element = []
        if len(element)<list_length:#for the elements which is short than defined list length, append integer_max inorder to get the same element length
            element.extend([integer_max]*(list_length-len(element)))
        element.extend(type_)# add input type into element, now, the element is a list with 12 value, the last two one is type
        re_elements.extend(element)# the length of re_elements will be N*12, N means the number of input (1,2,3)
    return re_elements


def type_find(input_):
    if input_ == []:# [1,1] means this input does not exist, just to make sure the shape of ML's input should be same 
        return [1,1]
    if isinstance(input_,int):
        return [0,0]
    elif isinstance(input_,list):
        return [1,0]
    else:
        return [0,1]# [0,1] means this input is function







 











class ML_Model(object):
    """docstring for ML_model"""


    def __init__(self,input_data1,input_data2,output_data):
        #self.input_data1 = input_data1
        #self.input_data2 = input_data2
        self.output_data = []
        self.input_data = []
        self.batch_size = 1
        np.set_printoptions(precision=3)
        for x,y in zip(input_data1,input_data2):
            self.input_data.append(np.append(x,y))
        for data in output_data:
            for i in range(0,5):
                self.output_data.append(data)

        print(len(self.output_data))
        print(len(self.input_data))
        self.model_begin()


    def get_weight(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def get_bias(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def avg_pool_layer(self,tensor):
        return tf.nn.avg_pool(value=tensor,ksize=[1,1,1,1],strides=[1,1,1,1],padding='SAME')

    def full_connection(self,fc_w,fc_b,tensor):
        fc_c = tf.matmul(tensor,fc_w)+fc_b
        return tf.nn.sigmoid(fc_c)

    def output_layer(self,fc_w,fc_b,tensor):
        fc_c = tf.matmul(tensor,fc_w)+fc_b
        return fc_c   #最后一层不需要使用softmax分类

        



    def model_begin(self):
        input_x = tf.placeholder(dtype=tf.float32,shape=[None,22*4])

        fc_w1 = self.get_weight([22*4,hidden_units])
        fc_b1 = self.get_bias([hidden_units])
        fc_z1 = self.full_connection(fc_w1,fc_b1,input_x)
        tf.summary.histogram('fc_w1',fc_w1)
        tf.summary.histogram('fc_b1',fc_b1)


        fc_w2 = self.get_weight([hidden_units,hidden_units])
        fc_b2 = self.get_bias([hidden_units])
        fc_z2 = self.full_connection(fc_w2,fc_b2,fc_z1)
        tf.summary.histogram('fc_w2',fc_w2)
        tf.summary.histogram('fc_b2',fc_b2)


        fc_w3 = self.get_weight([hidden_units,hidden_units])
        fc_b3 = self.get_bias([hidden_units])
        fc_z3 = self.full_connection(fc_w3,fc_b3,fc_z2)
        tf.summary.histogram('fc_w3',fc_w3)
        tf.summary.histogram('fc_b3',fc_b3)


        z_pool = tf.reshape(fc_z3,[-1,256,1,1])

        pool_tensor = self.avg_pool_layer(z_pool)

        out_pool = tf.reshape(pool_tensor,[-1,256])

        output_w = self.get_weight([hidden_units,attribute_length])
        output_b = self.get_bias([attribute_length])
        output_z = self.output_layer(output_w,output_b,out_pool)
        tf.summary.histogram('output_w',output_w)
        tf.summary.histogram('output_b',output_b)



        y_label = tf.placeholder(dtype=tf.float32,shape=[None,attribute_length])

     #   cross_entropy = -tf.reduce_sum(y_label*tf.log(output_z))
     #   square_loss = tf.reduce_sum(tf.sqrt(tf.pow((y_label-output_z),2)))

        re_lambda = 1e-3
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=output_z,labels=y_label))#+\
           # re_lambda*tf.nn.l2_loss(fc_w1) + re_lambda*tf.nn.l2_loss(fc_w2) + re_lambda*tf.nn.l2_loss(fc_w3)


        tf.summary.scalar('cross_entropy',cross_entropy)

        #add learning rate decay
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.5,global_step,1000,0.9)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step = global_step)
        
        #change the loss function to calculate all the elements in vector
        correct_prediction = tf.equal(tf.argmax(output_z,1),tf.argmax(y_label,1))



        index = tf.argmax(y_label,1)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


     


        with tf.Session() as sess:
            #training
            writer = tf.summary.FileWriter('/home/dongfang/coding/My_DeepCoder/train',sess.graph)
            merged = tf.summary.merge_all()
            
            sess.run(tf.global_variables_initializer())


            for i in range(0,50000):
                input_batch = random.sample(self.input_data,self.batch_size)
                output_batch = random.sample(self.output_data,self.batch_size)
                feed = {input_x:input_batch,y_label:output_batch}
                _,summary = sess.run([train_step,merged],feed_dict=feed)

                writer.add_summary(summary,i)


                if i%1000 == 0:

                    cross_entropy.eval(feed_dict=feed)
                    correct_prediction.eval(feed_dict=feed)
                    train_accuracy = accuracy.eval(feed_dict=feed)

                    print('step:', i)
                  #  print("step:%d,accuracy:%g"%(i,train_accuracy))
                    print("cross_entropy:{}".format(cross_entropy.eval(feed_dict=feed)))
                    print('tensor is \n',sess.run(output_z, feed))
                    print('attributes is \n', sess.run(y_label, feed))

            writer.close()
            print("ML training finished")





if __name__ == '__main__':
    
    print("data processing is beginning")
    input_example_dataset,output_example_dataset, output_attrs = data_process()#both input/output example dataset are the input of ML model


    print("ML model is beginning")
    ml_model = ML_Model(input_example_dataset,output_example_dataset,output_attrs)
    # print("this is the input_example_dataset")
    # print(input_example_dataset)
    # print("this is the output_example_dataset")
    # print(output_example_dataset)
    # print("this is the output_attributes")
    # print(output_attrs)
    # print(len(input_example_dataset))

    #print(len(output_example_dataset))
    # print(len(output_attrs))