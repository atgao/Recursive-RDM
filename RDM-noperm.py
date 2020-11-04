import networkx as nx
import numpy as np
import argparse
import itertools

import collections

from tqdm import tqdm 
from timer import Timer


def get_idx_for_match(u, v, n):

	# find starting position
	starting_pos = 0
	for i in range(u):
		starting_pos += n - i - 1

	idx = starting_pos + v - u - 1
	return idx

# TODO: you should probably write this function later...
def idx_to_match(idx, n):
	u, v = 0, 1
	match_idx = 0
	while match_idx < idx:
		pass 
	pass 

def convert_binary_to_graph(bits, n):
	'''
	converts bits to graph and 
	displays results for representation
	'''
	G = np.identity(n, dtype=np.bool_)

	row, col, count = 0, 1, n-1
	
	for b in bits:
		G[row, col] = int(b) - int("0")
		G[col, row] = ~G[row, col] # careful of this, 2s complement!!

		col += 1

		if col > count:
			row += 1
			col = row + 1 
	return G

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

def generate_graphs(n):
	'''
	n: number of nodes in graph
	return: list of all possible directed graphs
	'''
	e = int(n*(n-1)/2)
	# graphs = []
	# nodes = {}

	# for i in range(2**e):
	# 	bitgraph = np.binary_repr(i, width=e)
	# 	G = convert_binary_to_graph(bitgraph, n)

	# 	key = tuple(np.sort(np.sum(G, axis=0)))

	# 	if key not in nodes:
	# 		print("adding ... ", bitgraph, key)
	# 		nodes[key] = True
	# 		graphs.append(bitgraph)
	return [np.binary_repr(i, width=e) for i in range(2**e)]

def get_manipulability(graphs, n, ht, s=3):
	inds = np.arange(n)
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
				diff = cur[list(subset)] - new_prob[list(subset)]
				gain = np.sum(diff) # np.max instead??
				if gain > maxGain:
					maxGain = gain
	return maxGain

def check_permutation(bg1, bg2, n):
	G1 = convert_binary_to_graph(bg1, n)
	G2 = convert_binary_to_graph(bg2, n)

	# same graphs have equal # in/out 
	# graphs for each node
	sum1 = np.sort(np.sum(G1, axis=0))
	sum2 = np.sort(np.sum(G2, axis=0))
	return np.all(sum1 == sum2)

def permute_probs(bg1, bg2, prob, n):
	G1 = convert_binary_to_graph(bg1, n)
	G2 = convert_binary_to_graph(bg2, n)

	sum1 = np.sum(G1, axis=0)
	sum2 = np.sum(G2, axis=0)

	new_prob = np.zeros((n), dtype=np.float32)

	for i in range(n):
		if sum1[i] == sum2[i]:
			new_prob[i] = prob[i]
		else:
			# get first occurence
			idx = np.where(sum1 == sum2[i])[0][0]
			new_prob[i] = prob[idx]

	return new_prob

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
	
	ht = collections.OrderedDict(sorted(ht.items(), key=lambda x:len(x[0])))

	for k, v in ht.items():
		print(k, v)
	print(len(ht))
	print("AVG TIME: %f" %time.average_time)
	print("Total Time: %f" %time.total_time)

	gain = get_manipulability(graphs, n, ht)
	print(gain)