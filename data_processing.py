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
    x_dataset = []
    y_dataset = []

    for data in dataset:
        x_data = each_x_data_deal(data)
        x_dataset.append(x_data)
        y_dataset.append(data['attribute'])


    x_dataset = np.array(x_dataset)
    y_dataset = np.array(y_dataset)
    return x_dataset, y_dataset


def each_x_data_deal(data):
    ret_in_out_pairs_lists = []
    example = data["examples"]# example a dicticonary list
    for exa in example: #every example has five input-output dic, and input may be integer,array or both, output may be integer or array
        input_ = exa["input"] #return a list
        if len(input_)<input_number:
            for i in range(len(input_),input_number):
                input_.append([])
        
        input_output_pairs = [] #存放一个example的pairs size=4
        for input_array in input_:
            input_array = input_output_array_deal(input_array)
            input_output_pairs.append(input_array)
        

        output_ = exa["output"]
        output_array = input_output_array_deal(output_)
        input_output_pairs.append(output_array)

        ret_in_out_pairs_lists.append(input_output_pairs)
    #print(len(ret_in_out_pairs_lists))

    return ret_in_out_pairs_lists
    
    


def input_output_array_deal(element):
    type_ = type_find(element)#return the type of this input, encode by one-hot-encoding
    if type_ == [0,0]:#for the input type which is integer, convert integer into list with one element
        element = [element]
    if type_ == [0,1]:# [0,1] means the input type is function
        element = []
    if len(element)<list_length:#for the elements which is short than defined list length, append integer_max inorder to get the same element length
        element.extend([integer_max]*(list_length-len(element)))
    element.extend(type_)# add input type into element, now, the element is a list with 12 value, the last two one is type
    return element


def type_find(input_):
    if input_ == []:# [1,1] means this input does not exist, just to make sure the shape of ML's input should be same 
        return [1,1]
    if isinstance(input_,int):
        return [0,0]
    elif isinstance(input_,list):
        return [1,0]
    else:
        return [0,1]# [0,1] means this input is function



if __name__ == '__main__':
    
    print("data processing is beginning")
    x_dataset, y_dataset = data_process()
    print(np.shape(x_dataset))
    print(np.shape(y_dataset))
    # x_dataset:[
    # num_program, 
    # num_example_in_each_program(5),
    # num_input_in_each_example(3)+num_output_in_each_example(1),
    # length_of_input_array(20)+length_type_vector(2)
    # ]
    # y_dataset:[
    # num_program,
    # num_attributes(34)
    # ]



    