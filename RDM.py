import networkx as nx
import numpy as np

def convert_binary_to_graph(bits, n):
	'''
	converts bits to graph and 
	displays results for representation
	'''
	G = np.identity(n, dtype=np.bool_)
	# for b in bits:
	row, col, count = 0, 1, n-1
	
	for b in bits:
		G[row, col] = int(b) - int("0")
		G[col, row] = ~G[row, col] # careful of this, 2s complement!!

		col += 1

		if col > count:
			row += 1
			col = row + 1 
			count -= 1
	# print(G.astype('uint8'))

	return G

def calculate_prob(bits, G, n, ht={}):
	'''
	params: G: graph
	        n: number of nodes
	'''

	# need to calculate probability for each node 
	m = int(n*(n-1)/2) # total number of matches
	
	if n == 2:
		if int(bits, 2) == 0:
			return np.array([0, 1])
		else:
			return np.array([1, 0])

	# get probability each node is not eliminated this round
	prob = [0.0] * n

	# print(G.astype('uint8'))
	row, col, count = 0, 1, n-1
	inds = np.arange(n)
	for i in range(m):
		# TODO: get actual u, v, wrong answer rn
		u, v = row, col 
		res = bits[i]

		print(u, v, "res: %s" % res)

		# if res == 0:
			# new_bits = bits[]
		if res:
			temp_G = np.delete(G, u, 0)
			new_G = np.delete(temp_G, u, 1)
			print(new_G)

		new_bits = "".join(x[i] for i in new_G.flatten())
		# # print("test ", G[:i] + G[i+1:], "i is: ", i)
		print(new_bits)

		# ht[G[:i] + G[i+1:]] = calculate_prob(G[:i] + G[i+1:], n-1, ht)
		# test = 1/m * ht[G[:i] + G[i+1:]]

		# need to somehow put it in the hash table
		# print("????" , test, ht)
		col += 1

		if col > count:
			row += 1
			col = row + 1 
			count -= 1
	print("-------------------")
	pass

def generate_graphs(n):
	'''
	n: number of nodes in graph
	return: list of all possible directed graphs
	'''
	e = int(n*(n-1)/2)
	return [np.binary_repr(i, width=e) for i in range(2**e)]


if __name__ == "__main__":
	x = generate_graphs(2)
	y = generate_graphs(3)

	ht = {}
	
	# for graph in x:
	# 	ht[graph] = calculate_prob(graph, 2)
	
	for bitgraph in y:
		G = convert_binary_to_graph(bitgraph, 3)
		calculate_prob(bitgraph, G, 3)
		
	# convert_binary_to_graph(y[1], 3)
	print(ht)