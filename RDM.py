# import networkx as nx
import numpy as np
import argparse
import itertools

import collections

from tqdm import tqdm 
from utils.graph_utils import *
from utils.timer import Timer

def calculate_prob(bits, G, n, ht={}):
	'''
	params: G: graph
			n: number of nodes
	'''
	# TODO: look up equivalent graphs??

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

		if res == "0":
			elim = u
		else:
			elim = v
		
		# eliminate the node 
		temp_G = np.delete(G, elim, 0)
		new_G = np.delete(temp_G, elim, 1)

		triu_inds = np.triu_indices(n-1, 1)
		new_bits = "".join(str(i) for i in new_G[triu_inds].astype("uint8"))
	
		# first check ht
		if not new_bits in ht:
			ht[new_bits] = calculate_prob(new_bits, new_G, n-1, ht)

		# update the probability array with these probabilities
		prob[inds != elim] += 1/m * ht[new_bits]
		
		col += 1
		if col > count:
			row += 1
			col = row + 1 

	ht[bits] = prob
	return prob

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	parser.add_argument('-s', type=int, default=3)
	args = parser.parse_args()

	n, s = args.n, args.s
	time = Timer()
	graphs, manip = get_all_graphs(n, s)
	print("Finished generating graphs")
	print("%d unique graphs, %d manipulated graphs" % (len(graphs), len(manip)))


	ht = {}
	time = Timer()
	for bitgraph in tqdm(graphs+manip):
		time.tic()
		G = convert_binary_to_graph(bitgraph, n)
		calculate_prob(bitgraph, G, n, ht)
		time.toc()
	
	# ordering for convience sake
	ht = collections.OrderedDict(sorted(ht.items(), key=lambda x:len(x[0])))

	# for k, v in ht.items():
	# 	print(k, v)

	print("%d entries in table" % len(ht))
	print("Avg Time: %f sec per graph" %time.average_time)
	print("Total Time: %f sec" %time.total_time)
	
	gain = get_manipulability(graphs, n, ht, s=s)
	print("Total gain: ", gain)