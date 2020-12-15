import networkx as nx
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
		prob[inds != elim] += 1.0/m * ht[new_bits]
		
		col += 1
		if col > count:
			row += 1
			col = row + 1 

	ht[bits] = prob
	return prob

def get_manipulability(graphs, n, ht, s=3, subsets=None):
	inds = np.arange(n)
	if subsets is None:
		subsets = list(itertools.combinations(inds, s)) 
	
	# current graphs for n
	bitgraphs = graphs

	# ways the subset can manipulate 
	subset_bitgraphs = generate_graphs(s)[1:]
	count = s-1

	maxGain = float('-inf')

	for bits in bitgraphs:
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
				new_prob = ht[new_key]
				diff = new_prob[list(subset)] - cur[list(subset)] 
				gain = np.sum(diff) # np.max instead??
				if gain > maxGain:
					maxGain = gain
	return maxGain

def get_2R_bound(graphs, n, ht, s=3):
	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s)) 

	# ways the subset can manipulate 
	subset_bitgraphs = generate_graphs(s-1)[1:]
	count = s-2

	maxGain = float('-inf')
	# current graphs for n
	bitgraphs = graphs
	for bits in bitgraphs:
		G = convert_binary_to_graph(bits, n)
		for subset in subsets[:1]:
			for u in subset:
				elim = u 

				temp_G = np.delete(G, elim, 0)
				new_G = np.delete(temp_G, elim, 1)

				triu_inds = np.triu_indices(n-1, 1)
				new_bits = "".join(str(i) for i in new_G[triu_inds].astype("uint8"))

				colluding = list(subset)
				colluding.remove(u)

				cur = ht[new_bits]
				for sb in subset_bitgraphs: # tries all possible manipulations
					# need to set the matches here..
					manipulation = list(new_bits)
					i, j = 0, 1 # keep track of which indices so can access matches
					for match in sb:
						if int(match) == 1: 
							u, v = colluding[i], colluding[j]
							idx = get_idx_for_match(u, v, n-1)
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
					new_prob = ht[new_key]
					diff = cur[list(colluding)] - new_prob[list(colluding)]
					gain = np.sum(diff) # np.max instead??
					if gain > maxGain:
						maxGain = gain
	
	print(maxGain)

	pass

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	args = parser.parse_args()

	n = args.n
	time = Timer()
	graphs = generate_graphs(n)
	print("finished generating graphs", len(graphs))


	ht = {}
	time = Timer()

	for bitgraph in tqdm(graphs):
		time.tic()
		# print(bitgraph)
		G = convert_binary_to_graph(bitgraph, n)
		calculate_prob(bitgraph, G, n, ht)
		time.toc()
	
	# ordering for convience sake
	ht = collections.OrderedDict(sorted(ht.items(), key=lambda x:len(x[0])))

	for k, v in ht.items():
		print(k, v)
	print(len(ht))
	print("AVG TIME: %f" %time.average_time)
	print("Total Time: %f" %time.total_time)

	gain = get_manipulability(graphs, n, ht, s=2)
	print("Total Gain ", gain)

	# get_2R_bound(graphs, n, ht)