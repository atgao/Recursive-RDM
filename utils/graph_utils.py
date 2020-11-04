import networkx as nx
import numpy as np
import argparse
import itertools

import collections

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

def generate_graphs(n):
	'''
	n: number of nodes in graph
	return: list of all possible directed graphs
	'''
	e = int(n*(n-1)/2)
	return [np.binary_repr(i, width=e) for i in range(2**e)]

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