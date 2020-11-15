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

def generate_graphs(n, all=False):
	'''
	n: number of nodes in graph
	return: list of all possible directed graphs
	'''
	e = int(n*(n-1)/2)
	graphs = []
	nodes = {}

	if all:
		return [np.binary_repr(i, width=e) for i in range(2**e)]

	for i in range(2**e):
		bitgraph = np.binary_repr(i, width=e)
		G = convert_binary_to_graph(bitgraph, n)

		key = tuple(np.sort(np.sum(G, axis=0)))

		if key not in nodes:
			print("adding ... ", bitgraph, key)
			nodes[key] = True
			graphs.append(bitgraph)
	return graphs

def get_manipulability(graphs, n, ht, s=3):
	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s)) 
	
	# current graphs for n
	bitgraphs = graphs

	# ways the subset can manipulate 
	subset_bitgraphs = generate_graphs(s, all=True)[1:]
	count = s-1

	maxGain = float('-inf')

	for bits in tqdm(bitgraphs):
		cur = ht[bits] # the current probability
		for subset in subsets:
			for sb in subset_bitgraphs: # tries all possible manipulations
				# need to set the matches here..
				manipulation = list(bits)
				i, j = 0, 1 # keep track of which indices so can access matches
				for match in sb:
					if int(match) == 1: 
						u, v = subset[i], subset[j]
						idx = get_idx_for_match(u, v, n)
						if manipulation[idx] == "0":
							manipulation[idx] = "1"
						else:
							manipulation[idx] = "0"
					j += 1
					if j > count:
						i += 1
						j = i + 1 

				# now get new prob
				new_key = "".join(manipulation)
				if ht.get(new_key) is None:
					for g in bitgraphs:
						if check_permutation(g, new_key, n):
							ht[new_key] = permute_probs(g, new_key, ht[g], n)
							break
				# temporary fix...
				if ht.get(new_key) is None:
					ht[new_key] = calculate_prob(new_key, convert_binary_to_graph(new_key, n), n, ht)
				new_prob = ht[new_key]
				diff = new_prob[list(subset)] - cur[list(subset)] 
				gain = np.sum(diff) # np.max instead??
				if gain > maxGain:
					G = convert_binary_to_graph(bits, n)
					print("orig: ", bits)
					print(np.sum(G, axis=1)[list(subset)])
					G_mod = convert_binary_to_graph(new_key, n)
					print("new: ", new_key)
					print(np.sum(G_mod, axis=1)[list(subset)])
					print("new gain: ", gain)
					maxGain = gain
	return maxGain

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	args = parser.parse_args()

	n = args.n
	time = Timer()
	graphs = generate_graphs(n)
	print("finished generating graphs", len(graphs))


	ht = {}
	for bitgraph in tqdm(graphs):
		time.tic()
		# print(bitgraph)
		G = convert_binary_to_graph(bitgraph, n)
		calculate_prob(bitgraph, G, n, ht)
		time.toc()
	
	ht = collections.OrderedDict(sorted(ht.items(), key=lambda x:len(x[0])))

	for k, v in ht.items():
		print(k, v)
	print(len(ht))
	prev_total = time.total_time
	print("Avg Time: %f sec per graph" %time.average_time)
	print("Total Time: %f sec" %time.total_time)
	
	time.tic()
	gain = get_manipulability(graphs, n, ht)
	print("gain: ", gain)
	time.toc()
	print("manipulability total time: %f"% (time.total_time - prev_total))