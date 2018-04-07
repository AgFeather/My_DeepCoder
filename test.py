import copy
import numpy as np

#this is a dfs test doc


def scanl1_func(f, xs):
	ys = []
	ys.append(xs[0])
	for i in range(1, len(xs)):

		ys.append(f(ys[i-1], xs[i]))
	return ys

                       #number，function_name，lambda expression，lambda input type，lambda output type: 
                       #     0 means int; 1 means array; 2 means lambda function(for higher_order_fucntion); 3 means boolean
first_order_function = {1:['head', lambda xs: xs[0] if len(xs)>0 else None, [1], [0]],
						2:['last', lambda xs: xs[-1] if len(xs)>0 else None, [1], [0]],
						3:['take', lambda n, xs: xs[:n], [0, 1], [1]],
						4:['drop', lambda n, xs: xs[n:], [0, 1], [1]],
						5:['access', lambda n, xs: xs[n] if n >=0 and len(xs)>n else None, [0, 1], [0]],
						6:['minimum', lambda xs: min(xs) if len(xs)>0 else None, [1], [0]],
						7:['maximum', lambda xs: max(xs) if len(xs)>0 else None, [1], [0]],
						8:['reverse', lambda xs: list(reversed(xs)), [1], [1]],
						9:['sort', lambda xs: sorted(xs), [1], [1]],
						10:['sum', lambda xs: sum(xs), [1], [0]]
						}

higher_order_function = {11:['map', lambda f, xs: [f(x) for x in xs], [2, 1], [1]],
						 12:['filter', lambda f, xs: [x for x in xs if f(x)], [2, 1], [1]],
						 13:['count', lambda f, xs: len([x for x in xs if f(x)]), [2, 1], [0]],
						 14:['zipwith', lambda f, xs ,ys: [f(x, y) for (x, y) in zip(xs, ys)], [2, 1, 1], [1]],
						 15:['scanl1', lambda f, xs: scanl1_func(f, xs), [2, 1], [1]]
					#	 scanl1 given a lambda function f mapping integer pairs to integers, and an array xs,
					#	returns an array ys of the same length as xs and with its content defined by the recurrence 
					#	ys[0] = xs[0], ys[n] = f(ys[n-1], xs[n]) for n>=1
						 }


basic_function = {16:['>0', lambda x: True if x>0 else False, [0], [3]],
				  17:['<0', lambda x: True if x<0 else False, [0], [3]],
				  18:['%2==0', lambda x: True if x%2==0 else False, [0], [3]],
				  19:['%2==1', lambda x: True if x%2==1 else False, [0], [3]],
				  20:['+1', lambda x: x+1, [0], [0]],
				  21:['-1', lambda x: x-1, [0], [0]],
				  22:['*(-1)', lambda x: -x, [0], [0]],
				  23:['*2', lambda x: 2*x, [0], [0]],
				  24:['*3', lambda x: 3*x, [0], [0]],
				  25:['*4', lambda x: 4*x, [0], [0]],
				  26:['/2', lambda x: x/2, [0], [0]],
				  27:['/3', lambda x: x/3, [0], [0]],
				  28:['/4', lambda x: x/4, [0], [0]],
				  29:['**2', lambda x: x**2, [0], [0]],
				  30:['+', lambda x, y: x+y, [0, 0], [0]],
				  31:['-', lambda x, y: x-y, [0, 0], [0]],
				  32:['*', lambda x, y: x*y, [0, 0], [0]],
				  33:['min', lambda x,y: x if x<y else y, [0, 0], [0]],
				  34:['max', lambda x,y: x if x>y else y, [0, 0], [0]]
				  }



probabilities = [0.1, 0.35, 0.2, 0.1, 0.5, 0.4, 0.3, 0.921, 0.92, 0.1, 
				0.94, 0.92, 0.87, 0.1, 0.1, 0.3, 0.962, 0.81, 0.25, 0.54,
				0.88, 0.14, 0.12, 0.21, 0.942, 0.2, 0.7, 0.1, 0.1, 0.3,
				0.23, 0.12, 0.22, 0.31] 

probabilities = np.array(probabilities)
np.random.shuffle(probabilities)
print(probabilities)


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
max_input_number = 3 #一个程序的最多输入参数个数
# program_input_list = [[-17, -3, 4, 11, 0, -5, -9, 13, 6 ,6, -8, 11]]
# program_output = [-12, -20, -32, -36, -68]
# attributes = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
program_input_list = [[8,5,7,2,5]]
program_output = 5


def find_functions():
	probabilities_tuple = []
	for i in range(1,35):
		probabilities_tuple.append((probabilities[i-1], i))
	func_prob_tuples = sorted(probabilities_tuple, key=lambda x:x[0], reverse=True)
	# sort with the probilities and [0] means the most probable func
	print(func_prob_tuples)

	#DFS method to find a program
	for i in range(0, 34):
		function1 = func_prob_tuples[i][1]
		for j in range(i+1, 34):
			function2 = func_prob_tuples[j][1]
			for k in range(j+1, 34):
				function3 = func_prob_tuples[k][1]
				for m in range(k+1, 34):
					function4 = func_prob_tuples[m][1]
					for n in range(m+1, 34):
						function5 = func_prob_tuples[n][1]
						for y in range(n+1, 34):
							function6 = func_prob_tuples[y][1]


							function_list = []
							function_list.append(function1)
							function_list.append(function2)
							function_list.append(function3)
							function_list.append(function4)
							function_list.append(function5)
							function_list.append(function6)
						#	print(function_list)
							judge, program = create_program(function_list)
							if judge == True:
								return True, program
							else:
								print('this is a wrong fucntion combination, lets check next')

	return False, None




import itertools
def create_program(function_list):
	
	if not check_vaild(function_list):
		return False
	#do permutations operations for all functions in the list
	program_list = list(itertools.permutations(function_list, len(function_list))) #get all possible programs with given functions
#	print(program_list)


	for program in program_list:
		print('\nwe will check program: ', program)
		flag, func_list = judge_program(program)
		if flag:
			print('we find the program: {}'.format(func_list))
			return True, func_list
		else:
			print('this is a wrong program, lets check next')
	return False, None
	

def check_vaild(function_list):
	print(function_list)
	#check is there higher order function in function_list
	#if we have higher_order_functions but we don not have 
	#lambda expression for it, return False
	count_higher_function = 0
	count_basic_lambda = 0
	for func in function_list:
		if func>=16 and func<=34:
			count_basic_lambda += 1
		if func>=11 and func<=15:
			count_higher_function += 1
	#if count_basic_lambda == 0 and count_higher_function != 0:
	if count_basic_lambda != count_higher_function:
		print('there is no basic lambda expression for higher_order_function, error')
		return False, None

	return True


def build_input_temp_value_map():
	input_temp_values_map = {0:[], 1:[], 2:[], 3:[]}# 0 int; 1 array; 2 lambda function(for higher_order); 3 boolean
#	input_variable_number=[0,0,0]
#	print(program_input_list)
	for i in program_input_list:
		if isinstance(i, list):
			input_temp_values_map[1].append(i)
	#		input_variable_number[0]+=1
		if isinstance(i, int):
			input_temp_values_map[0].append(i)
	#		input_variable_number[1]+=1
#	print(input_temp_values_map)
	return input_temp_values_map





def judge_program(program):
	func_list = []
	input_temp_values_map = build_input_temp_value_map()

	# add all lambda expression to input_temp_values_map first
	for func in program:
		if func>=16 and func<=34:
			#means this function is a lambda expression for high_order_function
			input_temp_values_map[2].append(func)
			func_name = basic_function[func][0]
			func_list.append(func_name)
#	print('input_temp_values_map: ',input_temp_values_map)


	for func in program:
#		print('function number: ',func)
		if func<=10 and func>=1:
			#means this function is a first_order_function

			func_name = first_order_function[func][0]
			func_lambda = first_order_function[func][1]
			input_type = first_order_function[func][2]
			output_type = first_order_function[func][3]
			func_list.append(func_name)
			print('func_name: ', func_name)
		#	print(str(func_lambda))

			if len(input_type) == 1:
				input_1 = input_type[0] #get the input type(only one) of this function
				if len(input_temp_values_map[input_1]) == 0:
					# means there is not any input_temp_value satisifing the input type of this fucntion
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_temp_values_map[input_1])
					for values in values_list_1:
						try:
							output_ = func_lambda(values)
						except:
							return False, None
					#	print('input of lambda expression: ', values)
					#	print('output of lambda expression: ', output_)
						
						if output_ == program_output:
							return True, func_list
						else:
							
							input_temp_values_map[output_type[0]].append(output_)
				#			print(input_temp_values_map[output_type[0]])


			if len(input_type) == 2:
				input_1 = input_type[0]
				input_2 = input_type[1]
				if len(input_temp_values_map[input_1]) == 0 or len(input_temp_values_map[input_2]) == 0:
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_temp_values_map[input_1])
					values_list_2 = copy.deepcopy(input_temp_values_map[input_2])
					for values_1 in values_list_1:
						for values_2 in values_list_2:
							try:
						
								output_ = func_lambda(values_1, values_2)
							except:
								return False, None
						#	print('input of lambda expression: ', values_1, values_2)
						#	print('output of lambda expression: ', output_)
							if output_ == program_output:
								return True, func_list
							else:
								input_temp_values_map[output_type[0]].append(output_)
				







		if func>10 and func<=15:
			#means this function is a high_order_function

			func_name = higher_order_function[func][0]
			func_lambda = higher_order_function[func][1]
			input_type = higher_order_function[func][2]
			output_type = higher_order_function[func][3]
			func_list.append(func_name)
			print('func_name: ', func_name)
		#	print('input_type:', input_type)
		#	print('output_type:', output_type)



			if len(input_type) == 2:
				input_1 = input_type[0] # lambda function
				input_2 = input_type[1]
				if len(input_temp_values_map[input_1]) == 0 or len(input_temp_values_map[input_2]) == 0:
					print('error?')
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_temp_values_map[input_1])
					values_list_2 = copy.deepcopy(input_temp_values_map[input_2])
					for values_1 in values_list_1:
						for values_2 in values_list_2:
						#	print('values_1', basic_function[values_1][0], 'values_2', values_2)

							try:
								output_ = func_lambda(basic_function[values_1][1], values_2)
							except:
							#	print('exception')
								return False, None
							# print('output_: ',output_, 'program_output: ',program_output)
							# print('input of lambda expression: ', values_1, values_2)
							# print('output of lambda expression: ', output_)
							if output_ == program_output:
								return True, func_list
							else:
								input_temp_values_map[output_type[0]].append(output_)

			if len(input_type) == 3:
				input_1 = input_type[0]
				input_2 = input_type[1]
				input_3 = input_type[2]
				if len(input_temp_values_map[input_1]) == 0 or len(input_temp_values_map[input_2]) == 0:
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_temp_values_map[input_1])
					values_list_2 = copy.deepcopy(input_temp_values_map[input_2])
					values_list_3 = copy.deepcopy(input_temp_values_map[input_3])
					for values_1 in values_list_1:
						for values_2 in values_list_2:
							for values_3 in values_list_3:
								try:
									output_ = func_lambda(values_1, values_2, values_3)
								except:
									return False, None
								# print('input of lambda expression: ', values_1, values_2, values_3)
								# print('output of lambda expression: ', output_)
								if output_ == program_output:
									return True, func_list
								else:
									input_temp_values_map[output_type[0]].append(output_)

#	print(input_temp_values_map)
	# print('\n finish judgement, this is a wrong program')	
	return False, None









def check_pro_test():
	input_temp_values_map = {}
	env = {}







if __name__ == '__main__':
	


	# program_input_list = [[-17, -3, 4, 11, 0, -5, -9, 13, 6 ,6 -8, 11]]
	# program_output = [-12, -20, -32, -36, -68]





	flag1, program1 = find_functions()
#	flag1, program1 = create_program([8,9,11,12,17,25])
#	flag1, program1 = judge_program([27, 31, 33, 10, 13, 15])

	if flag1:
		print('\nright, it is the correct answer', program1)
	else:
		print('wrong')
