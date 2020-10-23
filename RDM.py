import networkx as nx
import numpy as np
from tqdm import tqdm 
from timer import Timer

def convert_binary_to_graph(bits, n):
	'''
	converts bits to graph and 
	displays results for representation
	'''
	G = np.identity(n, dtype=np.bool_)
	# for b in bits:
	row, col, count = 0, 1, n-1
	
	for b in bits:
		# print(row, col, count, n)
		G[row, col] = int(b) - int("0")
		G[col, row] = ~G[row, col] # careful of this, 2s complement!!

		col += 1

		if col > count:
			row += 1
			col = row + 1 
			# count -= 1
	return G

def calculate_prob(bits, G, n, ht={}):
	'''
	params: G: graph
			n: number of nodes
	'''

	# need to calculate probability for each node 
	m = int(n*(n-1)/2) # total number of matches

	# print("Calculating prob for %s with n = %d" % (bits, n))
	if n == 2:
		if int(bits, 2) == 0:
			return np.array([0, 1])
		else:
			return np.array([1, 0])

	# get probability each node is not eliminated this round
	prob = np.zeros((n), dtype=np.float32)

	row, col, count = 0, 1, n-1
	inds = np.arange(n)
	for i in range(m):
		u, v = row, col 
		res = bits[i]

		# print(u, v, "res: %s" % res)

		if res == "0":
			elim = u
		else:
			elim = v
		
		# eliminate the node 
		# print("in match %d w n=%d, which is %d vs %d, %d eliminated" %(i, n, u, v, elim))
		temp_G = np.delete(G, elim, 0)
		new_G = np.delete(temp_G, elim, 1)

		triu_inds = np.triu_indices(n-1, 1)
		new_bits = "".join(x[i] for i in new_G[triu_inds])
	
		# first check ht
		if not new_bits in ht:
			# print("FIRST TIME CALLING ", new_bits)
			ht[new_bits] = calculate_prob(new_bits, new_G, n-1, ht)

		# update the probability array with these probabilities
		# print(ht[new_bits] )
		prob[inds != elim] += 1/m * ht[new_bits]
		
		col += 1
		if col > count:
			row += 1
			col = row + 1 
			# count -= 1

	ht[bits] = prob
	# print("-------------------")
	return prob

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
	z = generate_graphs(4)


	n = 4
	graphs = generate_graphs(n)

	ht = {}
	time = Timer()

	for bitgraph in tqdm(graphs):
		time.tic()
		print(bitgraph)
		G = convert_binary_to_graph(bitgraph, n)
		calculate_prob(bitgraph, G, n, ht)
		time.toc()
	
		
	# convert_binary_to_graph(y[1], 3)
	for k, v in ht.items():
		print(k, v)
	print(len(ht))
	print("AVG TIME: %f" %time.average_time)
	print("Total Time: %f" %time.total_time)