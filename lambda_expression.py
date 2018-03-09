import copy



def scanl1_func(f, xs):
	ys = []
	ys[0] = xs[0]
	for i in range(1, len(xs)):
		ys[i] = f(xs[i-1], xs[i])

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

higher_order_function = {11:['map', lambda f, xs: [f(x) for x in xs], [1, 2], [1]],
						 12:['filter', lambda f, xs: [x for x in xs if f(x)], [1, 2], [1]],
						 13:['count', lambda f, xs: len([x for x in xs if f(x)]), [1, 2], [0]],
						 14:['zipwith', lambda f, xs ,ys: [f(x, y) for (x, y) in zip(xs, ys)], [1, 1, 2], [1]],
						 15:['scanl1', lambda f, xs: scanl1_func(f, xs), [1, 2], [1]]
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


probabilities = [0.1,0.3,0.2,0.1,0.5,0.4,0.7,0.4,0.2,0.1,0.4,0.2,0.7,0.1,0.1,0.3,0.2,0.1,0.5,0.4,0.7,0.4,0.2,0.1,0.4,0.2,0.7,0.1,
					0.1,0.3,0.2,0.1] 

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
program_input_list = [1,5,[0,2,2,10]]
program_output = 12
attributes = [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



def find_functions():
	probabilities_tuple = []
	for i in range(1,32):
		probabilities_tuple.append((probabilities[i], i))
	func_prob_tuples = sorted(probabilities_tuple, key=lambda x:x[0], reverse=True)
	# sort with the probilities and [0] means the most probable func
	print(func_prob_tuples)
	
	#DFS method to find a program
	for i in range(0, 31):
		function1 = func_prob_tuples[i][1]
		for j in range(i+1, 31):
			function2 = func_prob_tuples[j][1]
			for k in range(j+1, 31):
				function3 = func_prob_tuples[k][1]
				for m in range(k+1, 31):
					function4 = func_prob_tuples[m][1]
					for n in range(m+1, 31):
						function5 = func_prob_tuples[n][1]
						function_list = []
						function_list.append(function1)
						function_list.append(function2)
						function_list.append(function3)
						function_list.append(function4)
						function_list.append(function5)
						print(function_list)
						judge, program = create_program(function_list)
						if judge == True:
							return program







import itertools
def create_program(function_list):
	print(function_list)

	#do permutations operations for all functions in the list
	program_list = list(itertools.permutations(function_list, len(function_list))) #get all possible programs with given functions
#	print(program_list)

	for program in program_list:
		flag, func_list = judge_program(program)
		if flag:
			print('we find the program: {}'.format(func_list))
			return True, func_list
		

def build_input_temp_value_map():
	input_variable_map = {0:[], 1:[], 2:[], 3:[]}# 0 int; 1 array; 2 lambda function(for higher_order); 3 boolean
	input_variable_number=[0,0,0]
	for i in program_input_list:
		if isinstance(i, list):
			input_variable_map[1].append(i)
			input_variable_number[0]+=1
		if isinstance(i, int):
			input_variable_map[0].append(i)
			input_variable_number[1]+=1
	return input_variable_map


def judge_program(program):
	func_list = []
	input_variable_map = build_input_temp_value_map()
#	print('input_variable_map: ',input_variable_map)
	# add all lambda expression to input_variable_map first
	for func in program:
		if func>=16 and func<=34:
			#means this function is a lambda expression for high_order_function
			input_variable_map[3].append(func)
#	print('input_variable_map: ',input_variable_map)
	for func in program:
		
		if func<=10 and func>=1:
			#means this function is a first_order_function

			func_name = first_order_function[func][0]
			func_lambda = first_order_function[func][1]
			input_type = first_order_function[func][2]
			output_type = first_order_function[func][3]
			func_list.append(func_name)
			print('func_name: ', func_name)
			print(str(func_lambda))
			print('input_type:', input_type)
			print('output_type:', output_type)


			if len(input_type) == 1:
				input_1 = input_type[0] #get the input type(only one) of this function
				print(input_1)
				if len(input_variable_map[input_1]) == 0:
					# means there is not any input_temp_value satisifing the input type of this fucntion
					return False, None
				else:
					for values in input_variable_map[input_1]:
						print(values)
						output_ = func_lambda(values)
						print(output_, ' ?==', program_output)
						if output_ == program_output:
							return True, func_list
						else:
							
							input_variable_map[output_type[0]].append(output_)
							print(input_variable_map[output_type[0]])


			if len(input_type) == 2:
				input_1 = input_type[0]
				input_2 = input_type[1]
				if len(input_variable_map[input_1]) == 0 or len(input_variable_map[input_2]) == 0:
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_variable_map[input_1])
					values_list_2 = copy.deepcopy(input_variable_map[input_2])
					for values_1 in values_list_1:
						for values_2 in values_list_2:
						#	print('int:', values_1, 'list', values_2)
							output_ = func_lambda(values_1, values_2)
							if output_ == program_output:
								return True, func_list
							else:
								input_variable_map[output_type[0]].append(output_)
				







		if func>10 and func<=15:
			#means this function is a high_order_function

			func_name = higher_order_function[func][0]
			func_lambda = higher_order_function[func][1]
			input_type = higher_order_function[func][2]
			output_type = higher_order_function[func][3]
			func_list.append(func_name)
			print('func_name: ', func_name)
			print(str(func_lambda))
			print('input_type:', input_type)
			print('output_type:', output_type)



			if len(input_type) == 2:
				input_1 = input_type[0] # lambda function
				input_2 = input_type[1]
				if len(input_variable_map[input_1]) == 0 or len(input_variable_map[input_2]) == 0:
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_variable_map[input_1])
					values_list_2 = copy.deepcopy(input_variable_map[input_2])
					for values_1 in values_list_1:
						for values_2 in values_list_2:
							output_ = func_lambda(values_1, values_2)
							if output_ == program_output:
								return True, func_list
							else:
								input_variable_map[output_type[0]].append(output_)

			if len(input_type) == 3:
				input_1 = input_type[0]
				input_2 = input_type[1]
				input_3 = input_type[2]
				if len(input_variable_map[input_1]) == 0 or len(input_variable_map[input_2]) == 0:
					return False, None
				else:
					values_list_1 = copy.deepcopy(input_variable_map[input_1])
					values_list_2 = copy.deepcopy(input_variable_map[input_2])
					values_list_3 = copy.deepcopy(input_variable_map[input_3])
					for values_1 in values_list_1:
						for values_2 in values_list_2:
							for values_3 in values_list_3:
								output_ = func_lambda(values_1, values_2, values_3)
								if output_ == program_output:
									return True, func_list
								else:
									input_variable_map[output_type[0]].append(output_)


	print('finish')	
	return False, None












if __name__ == '__main__':
	
#	find_functions()

#	create_program([6, 12, 20, 26, 4])
#	create_program([1,2,3,4,5])
	flag, program = judge_program([11,12,13,14,15])
	if flag:
		print('finish', program)
	else:
		print('error')