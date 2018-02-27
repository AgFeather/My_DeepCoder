



first_order_function = {1:['head', lambda xs: xs[0] if len(xs)>0 else None],
						2:['last', lambda xs: xs[-1] if len(xs)>0 else None],
						3:['take', lambda n, xs: xs[:n]],
						4:['drop', lambda n, xs: xs[n:]],
						5:['access', lambda n, xs: xs[n] if n >=0 and len(xs)>n else None],
						6:['minimum', lambda xs: min(xs) if len(xs)>0 else None],
						7:['maximum', lambda xs: max(xs) if len(xs)>0 else None],
						8:['reverse', lambda xs: list(reversed(xs))],
						9:['sort', lambda xs: sorted(xs)],
						10:['sum', lambda xs: sum(xs)]
						}

higher_order_function = {11:['map', lambda f, xs: [f(x) for x in xs]],
						 12:['filter', lambda f, xs: [x for x in xs if f(x)]],
						 13:['count', lambda f, xs: len([x for x in xs if f(x)])],
						 14:['zipwith', lambda f, xs ,ys: [f(x, y) for (x, y) in zip(xs, ys)]]
					#	 15:['scanl1', lambda]
						 }

basic_function = {16:['>0', lambda x: True if x>0 else False],
				  17:['<0', lambda x: True if x<0 else False],
				  18:['%2==0', lambda x: True if x%2==0 else False],
				  19:['%2==1', lambda x: True if x%2==1 else False],
				  20:['+1', lambda x: x+1],
				  21:['-1', lambda x: x-1],
				  22:['*(-1)', lambda x: -x],
				  23:['*2', lambda x: 2*x],
				  24:['*3', lambda x: 3*x],
				  25:['*4', lambda x: 4*x],
				  26:['/2', lambda x: x/2],
				  27:['/3', lambda x: x/3],
				  28:['/4', lambda x: x/4],
				  29:['**2', lambda x: x**2],
				  30:['min', lambda x,y: x if x<y else y],
				  31:['max', lambda x,y: x if x>y else y]
				  }


probabilities = [0.1,0.3,0.2,0.1,0.5,0.4,0.7,0.4,0.2,0.1,0.4,0.2,0.7,0.1,0.1,0.3,0.2,0.1,0.5,0.4,0.7,0.4,0.2,0.1,0.4,0.2,0.7,0.1,
					0.1,0.3,0.2,0.1] 


def find_functions():
	probabilities_tuple = []
	for i in range(1,32):
		probabilities_tuple.append((probabilities[i], i))
	func_prob_tuples = sorted(probabilities_tuple, key=lambda x:x[0], reverse=True)
	print(func_prob_tuples)
	for i in range(0, 31):
		function1 = func_prob_tuples[i]
		for j in range(i, 31):
			function2 = func_prob_tuples[j]
			for k in range(j, 31):
				function3 = func_prob_tuples[k]
				for m in range(k, 31):
					function4 = func_prob_tuples[m]
					for n in range(m, 31):
						function5 = func_prob_tuples[n]



find_functions()

#print(first_order_function[1][1]([5,2,3]))