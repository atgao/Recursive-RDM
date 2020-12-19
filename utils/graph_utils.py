# import networkx as nx
import numpy as np
import argparse
import itertools
# import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

import collections
from tqdm import tqdm 
import subprocess

from timer import Timer


def draw_graph(bitgraphs, n):

	for i in range(len(bitgraphs)):
		G = convert_binary_to_graph(bitgraphs[i], n)
		G = nx.from_numpy_matrix(G, create_using=nx.DiGraph)
		pos = nx.layout.spring_layout(G)
		nodes = nx.draw_networkx_nodes(G, pos)
		edges = nx.draw_networkx_edges(G, pos, arrowstyle="->")
		nx.draw_networkx_labels(G, pos, font_color="w")

		# plt.draw() 
		# plt.figure()
		plt.title(bitgraphs[i])
		plt.draw()
		plt.savefig("graphs/rdm/%s.jpg"%bitgraphs[i])
		plt.clf()

def get_num_edges(n):
	return int(n*(n-1)/2)

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

def generate_graphs(n, unique=True):
	'''
	n: number of nodes in graph
	return: list of all possible directed graphs
	'''

	if not unique:
		e = get_num_edges(n)
		return [np.binary_repr(i, width=e) for i in range(2**e)]

	return gentourng(n)

def permute_probs(bg1, bg2, prob, n):
	G1 = convert_binary_to_graph(bg1, n)
	G2 = convert_binary_to_graph(bg2, n)

	perms = list(itertools.permutations(np.arange(n)))[1:]

	new_prob = np.zeros((n), dtype=np.float32)
	P = np.zeros((n, n), dtype=np.uint8)
	inds = np.arange(n)

	for perm in perms:
		P[inds, perm] = 1
		if np.all(P @ (G1 @ P.T) == G2):
			break
		P[inds, perm] = 0 # reset
	
	# set the indices 
	for i in range(n):
		new_prob[i] = prob[perm[i]]
	return new_prob

def get_num_beat(bits, subset, n):
	G = convert_binary_to_graph(bits, n)

	num_beat = ~G[subset, :]

	# can only count ones outside
	num_beat[:, subset] = False 

	return np.sum(num_beat)

def get_all_graphs(n, s=3):
	bitgraphs = generate_graphs(n)
	bitgraphs_to_delete = set()
	res = []
	res.extend(bitgraphs)
	manip = []

	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s))

	if s == 2:
		subset_bitgraphs = generate_graphs(s)
	else:
		subset_bitgraphs = generate_graphs(s, unique=False)
		# subset_bitgraphs = generate_graphs(s)
	count = s-1

	for bits in bitgraphs:
		for subset in subsets:
			if get_num_beat(bits, subset, n) > 9:
				# print("continuing???? on ", bits, get_num_beat(bits, subset, n))
				bitgraphs_to_delete.add(bits)
				continue
			for sb in subset_bitgraphs:
				manipulation = list(bits)
				i, j = 0, 1 # keep track of which indices so can access matches
				for match in sb:
					# if int(match) == 1: 
					# 	u, v = subset[i], subset[j]
					# 	idx = get_idx_for_match(u, v, n)
					# 	if manipulation[idx] == "0":
					# 		manipulation[idx] = "1"
					# 	else:
					# 		manipulation[idx] = "0"
					u, v = subset[i], subset[j]
					idx = get_idx_for_match(u, v, n)
					manipulation[idx] = match
					j += 1
					if j > count:
						i += 1
						j = i + 1 

				# now get new prob
				new_key = "".join(manipulation)
				if get_num_beat(new_key, subset, n) > 9:
					continue
				manip.append(new_key)
	# print(res)
	# print(2**get_num_edges(n), len(res), len(bitgraphs)) # comparison of how much we r saving
	print("bitgraphs to delete: ", len(bitgraphs_to_delete))
	res = set(res) - bitgraphs_to_delete
	return list(res), manip

def get_manipulability(graphs, n, ht, s=3):
	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s)) 
	
	# current graphs for n
	bitgraphs = graphs

	# ways the subset can manipulate 
	if s == 2:
		subset_bitgraphs = generate_graphs(s)
	else:
		subset_bitgraphs = generate_graphs(s, unique=False)
		# subset_bitgraphs = generate_graphs(s)
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
					# if int(match) == 1: 
					# 	u, v = subset[i], subset[j]
					# 	idx = get_idx_for_match(u, v, n)
					# 	if manipulation[idx] == "0":
					# 		manipulation[idx] = "1"
					# 	else:
					# 		manipulation[idx] = "0"
					u, v = subset[i], subset[j]
					idx = get_idx_for_match(u, v, n)
					manipulation[idx] = match
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
					print("maxgain: %f, gain:%f, manip %s for subset %s" % (maxGain, gain, sb, subset))
					maxGain = gain
	return maxGain

def gentourng(n):
	res = subprocess.check_output("./nauty27r1/gentourng %d" % n, shell=True)
	return res.decode().splitlines()

def get_all_graphs_higher_nodes(n, s=3):
	m = n-s
	bitgraphs = generate_graphs(m)
	subsets = generate_graphs(s, unique=True)

	final_graphs = []

	# now connect them
	# actually need to do this in a loop up to 
	# k = 9???
	conns = kbits(m*s, 9)
	for i in range(len(conns)):
		arr = np.fromstring(conns[i], dtype='u1').reshape((s, m)) - ord('0')
		conns[i] = arr
	print("finished converting conns...")

	zeros = np.zeros((m, m), dtype=np.bool_)
	conns_zeros = np.zeros((s, m), dtype=np.bool_)
	
	time = Timer()

	for subset in subsets:
		G = np.identity(n, dtype=np.bool_)
		G[:s, :s] = convert_binary_to_graph(subset, s)
		
		for bitgraph in bitgraphs: 
			time.tic()
			G[s:, s:] = convert_binary_to_graph(bitgraph, m)

			for conn in conns:
				
				# technically should also set G[s:, :s]
				# but it's okay bc of way graph is built 
				G[:s, s:] = conn
				
				# build the new graph and append it
				triu_inds = np.triu_indices(n-1, 1)
				new_bits = "".join(str(i) for i in G[triu_inds].astype("uint8"))
				final_graphs.append(new_bits)
				
				# clear out connections
				G[:s, s:] = conns_zeros

			# clears out the mxm graph for new mxm graph
			G[s:, s:] = zeros
			time.toc()
	print("Total Time: %f sec" %time.total_time)
	print("Avg Time: %f sec per graph" %time.average_time)
	print(len(final_graphs))
	return final_graphs

# taken from https://stackoverflow.com/questions/1851134/generate-all-binary-strings-of-length-n-with-k-bits-set
def kbits(n, k):
	result = []
	for bits in itertools.combinations(range(n), k):
		s = ['0'] * n
		for bit in bits:
			s[bit] = '1'
		result.append(''.join(s))
	return result

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	args = parser.parse_args()

	n = args.n

	# test = kbits(18, 9)
	# print(test)
	# print(len(test))
	get_all_graphs_higher_nodes(9)