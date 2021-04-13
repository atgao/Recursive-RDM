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

def find_termination(n, s): 
	gains = []

	start_n = n
	ht = {}

	while True: 
		time = Timer()

		if n < 9:
			graphs = get_all_graphs(n, s)
			count = 0
			for bitgraph, subsets in tqdm(graphs.items()):
				time.tic()
				G = convert_binary_to_graph(bitgraph, n)
				calculate_prob(bitgraph, G, n, ht)

				count += 1
				
				for subset, manips in subsets.items():
					for manip in manips:
						G = convert_binary_to_graph(manip, n)
						calculate_prob(manip, G, n, ht)
						count += 1
		else: 
			colluders = generate_graphs(s)
			non_colluders = generate_graphs(n-s)
			time.tic()
			g, graphs = connect_two_graphs(colluders, non_colluders, 8, n-s)
			time.toc()
			print("Total time to connect %fs" % time.total_time)

			count = 0
			for bitgraph, manips in tqdm(graphs.items()):
				time.tic()
				G = convert_binary_to_graph(bitgraph, n)
				calculate_prob(bitgraph, G, n, ht)
				count += 1

				for m in manips: 
					G = convert_binary_to_graph(m, n)
					calculate_prob(m, G, n, ht)
					count += 1
				time.toc()

		
		# ordering for convience sake
		ht = collections.OrderedDict(sorted(ht.items(), key=lambda x:len(x[0])))
		print("%d entries in table" % count)

		print("Avg Time: %f sec per graph" %time.average_time)
		print("Total Time: %f sec" %time.total_time)
		
		if n < 9:
			gain = get_manipulability(graphs, n, ht, s=s)
		else:
			gain = get_manipulability_higher_nodes(g, graphs, n, ht, s=s)
		print("Total gain for %d nodes: %f" %(n, gain))
		gains.append(gain)

		n += 1 # increase n
		
		if n > 10: 
			break



	start = start_n 
	for gain in gains:
		print("Gained %f for n=%d" % (gain, start))
		start += 1
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	parser.add_argument('-s', type=int, default=3)
	parser.add_argument('-t', type=str, default="True")
	args = parser.parse_args()


	n, s, terminating = args.n, args.s, args.t
	time = Timer()
	ht = {}

	if terminating.lower() == "true":
		find_termination(n, s)
	else:

		if n < 9:
			graphs = get_all_graphs(n, s)
			time = Timer()
			count = 0
			for bitgraph, subsets in tqdm(graphs.items()):
				time.tic()
				G = convert_binary_to_graph(bitgraph, n)
				calculate_prob(bitgraph, G, n, ht)

				count += 1
				
				for subset, manips in subsets.items():
					for manip in manips:
						G = convert_binary_to_graph(manip, n)
						calculate_prob(manip, G, n, ht)
						count += 1

				time.toc()
			
			# ordering for convience sake
			ht = collections.OrderedDict(sorted(ht.items(), key=lambda x:len(x[0])))
			print(count)

			# for k, v in ht.items():
			# 	print(k, v)
			print("Avg Time: %f sec per graph" %time.average_time)
			print("Total Time: %f sec" %time.total_time)
			
			gain = get_manipulability(graphs, n, ht, s=s)
			print("Total gain for %d nodes: %f" %(n, gain))
		else: 
			colluders = generate_graphs(s)
			non_colluders = generate_graphs(n-s)
			time.tic()
			g, graphs = connect_two_graphs(colluders, non_colluders, 8, n-s)
			time.toc()
			print("Total time to connect %fs" % time.total_time)

			for bitgraph, manips in tqdm(graphs.items()):
				time.tic()
				G = convert_binary_to_graph(bitgraph, n)
				calculate_prob(bitgraph, G, n, ht)

				for m in manips: 
					G = convert_binary_to_graph(m, n)
					calculate_prob(m, G, n, ht)
				time.toc()
			
			gain = get_manipulability_higher_nodes(g, graphs, n, ht, s=s)
			print("Total gain for %d nodes: %f" %(n, gain))

		

	