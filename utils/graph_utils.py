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
	graphs_and_subsets = collections.defaultdict(list)
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
				continue
			graphs_and_subsets[bits].append(subset) # consider these subsets for the graph
			
			# form the originally colluding nodes
			orig_colluding = []
			for s in subset:
				orig_colluding.append(bits[s])
			orig_colluding = "".join(orig_colluding)

			for sb in subset_bitgraphs:

				manipulation = list(bits)
				i, j = 0, 1 # keep track of which indices so can access matches
				for match in sb:
					u, v = subset[i], subset[j]
					idx = get_idx_for_match(u, v, n)
					manipulation[idx] = match
					j += 1
					if j > count:
						i += 1
						j = i + 1 

				# now get new prob
				new_key = "".join(manipulation)
				
				# stricter checks to eliminate graphs
				manip_diff = count_difference(orig_colluding, sb)
				if manip_diff == 3 and get_num_beat(new_key, subset, n) > 9:
					continue 
				if manip_diff == 2 and get_num_beat(new_key, subset, n) > 6:
					continue 
				if manip_diff == 1 and get_num_beat(new_key, subset, n) > 3: 
					continue
				manip.append(new_key)
	return graphs_and_subsets, manip

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
		for subset in bitgraphs[bits]: # only use subsets we know satisfy conditions
			
			# form the originally colluding nodes
			orig_colluding = []
			for s in subset:
				orig_colluding.append(bits[s])
			orig_colluding = "".join(orig_colluding)

			for sb in subset_bitgraphs: # tries all possible manipulations

				manipulation = list(bits)
				i, j = 0, 1 # keep track of which indices so can access matches
				for match in sb:

					u, v = subset[i], subset[j]
					idx = get_idx_for_match(u, v, n)
					manipulation[idx] = match
					j += 1
					if j > count:
						i += 1
						j = i + 1 

				# now get new prob
				new_key = "".join(manipulation)

				# stricter checks to eliminate graphs
				manip_diff = count_difference(orig_colluding, sb)
				if manip_diff == 3 and get_num_beat(new_key, subset, n) > 9:
					continue 
				if manip_diff == 2 and get_num_beat(new_key, subset, n) > 6:
					continue 
				if manip_diff == 1 and get_num_beat(new_key, subset, n) > 3: 
					continue
				
				new_prob = ht[new_key]
				diff = new_prob[list(subset)] - cur[list(subset)] 
				gain = np.sum(diff) # np.max instead??
				if gain > maxGain:
					# print("maxgain: %f, gain:%f, manip %s for subset %s" % (maxGain, gain, sb, subset))
					# print("current graph ", bits, ht[bits])
					# print("manipulated graph ", new_key, ht[new_key])
					# print("diff: ", diff)
					# print("-------------")
					maxGain = gain
	return maxGain

def get_manipulability_higher_nodes(graphs, manips, n, ht, s=3):
	subset = [i for i in range(s)]

	maxGain = float('-inf')

	for graph, manip in zip(graphs, manip):
		diff = ht[manip][subset] - ht[graph][subset]
		gain = np.sum(diff) # np.max instead??
		if gain > maxGain:
			# print("maxgain: %f, gain:%f, manip %s for subset %s" % (maxGain, gain, sb, subset))
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
	manips = []

	# now connect them
	# actually need to do this in a loop up to 
	for k in range(9):
		conns = kbits(m*s, k)
		for i in range(len(conns)):
			# should technically ~conns[i] since 1 = win, 0 = beat...
			arr = np.fromstring(conns[i], dtype='u1').reshape((s, m)) - ord('0')
			conns[i] = ~arr # can convert it to ~conns[i] here
		print("finished converting conns... for k = %d" %k)

		zeros = np.zeros((m, m), dtype=np.bool_)
		conns_zeros = np.zeros((s, m), dtype=np.bool_)
		
		# time = Timer()

		for subset in subsets:
			G = np.identity(n, dtype=np.bool_)
			G[:s, :s] = convert_binary_to_graph(subset, s)
			
			for bitgraph in bitgraphs: 
				# time.tic()
				G[s:, s:] = convert_binary_to_graph(bitgraph, m)

				for conn in conns:
					
					# technically should also set G[s:, :s]
					# but it's okay bc of way graph is built 
					G[:s, s:] = conn
					
					# build the new graph and append it
					triu_inds = np.triu_indices(n, 1)
					new_bits = "".join(str(i) for i in G[triu_inds].astype("uint8"))
					if subset == subsets[-1]:
						manips.append(new_bits)
					else:
						final_graphs.append(new_bits)
					
					
					# clear out connections
					G[:s, s:] = conns_zeros

				# clears out the mxm graph for new mxm graph
				G[s:, s:] = zeros
				# time.toc()

	# print("Total Time: %f sec" %time.total_time)
	# print("Avg Time: %f sec per graph" %time.average_time)
	print(len(final_graphs), len(manips))
	return final_graphs, manips

# taken from https://stackoverflow.com/questions/1851134/generate-all-binary-strings-of-length-n-with-k-bits-set
def kbits(n, k):
	result = []
	for bits in itertools.combinations(range(n), k):
		s = ['0'] * n
		for bit in bits:
			s[bit] = '1'
		result.append(''.join(s))
	return result

def count_difference(bit1, bit2):
	count = 0
	for b1, b2 in zip(bit1, bit2):
		if b1 != b2: 
			count += 1
	return count

def connect_two_graphs(colluders, graphs, k, n, s=3):
	res = []

	colluding_groups = [determine_groups(G, s) for G in colluders]

	colluding_combos = {}
	for group, G in zip(colluding_groups, colluders):
		combos = []
		for i in range(s+1):
			combos.append(list(itertools.combinations(group.values(), i))) # TODO: make sure to go thru keys
		colluding_combos[G] = combos 

	connected_graphs = []
	G = np.identity(n, dtype=np.bool_)
	for graph in graphs[:5]: 
		group = determine_groups(graph, n)
		print(graph, group)
		combos = []
		for i in range(n+1):
			combos.append(list(itertools.combinations(group.values(), i))) # TODO: make sure to go thru keys

		# conenct it to each type of graph 
		for colluding_G in colluders:
			print(colluding_G)
			print(colluding_combos[colluding_G])
			temp = list(itertools.product(combos, colluding_combos[colluding_G])) 
		# print(print(new_graph) for new_graph in temp)
		# for combo in combos:
		# 	print(combo)
		# 	print("*******")
		print("????????")
		print(combos)
		
		print("---------------------------------")
	return res 

def check_nodes_in_cycle(nodes, G):
	stack = [nodes[0]]
	nodes = set(nodes[1:])
	
	while stack:
		curr = stack.pop() 
		wins = np.argwhere(G[curr] == 1)[0] if len(np.argwhere(G[curr] == 1)) > 0 else []

		for w in wins:
			if w in nodes:
				stack.append(w)
				nodes.remove(w)
		if len(nodes) == 0: 
			return True 
	return False

def determine_groups(G, n):
	# 1. start by partitioning sets by degree 
	graph = convert_binary_to_graph(G, n)
	degrees = np.sum(graph, axis=1) # degree = number of wins 

	# 2. partition into initial sets by degree
	groups = {} # key = degree, value = nodes in group
	for node, degree in enumerate(degrees):
		group = groups.get(degree, [])
		group.append(node)
		groups[degree] = group 
	
	# 3. refine groups
	np.fill_diagonal(graph, 0) # do this to help with indexing later
	new_group_num = np.max(degrees) + 1 

	while True:
		print(groups)
		new_groups = {}
		for k, v in groups.items():
			nodes_beat = set() 

			if len(v) > 1 and len(v) % 2 != 0 and check_nodes_in_cycle(v, graph) is True:
				continue

			losers = np.argwhere(graph[v[0]] == 1)[0] if len(np.argwhere(graph[v[0]] == 1)) > 0 else []
			for l in losers: nodes_beat.add(l)
			new_group_nodes = []
		
			for node in v[1:]:
				losers = np.argwhere(graph[node] == 1)[0]

				for l in losers:
					if l not in nodes_beat:
						# split this into new group
						new_group_nodes.append(node)
						break 
			
			# add new groups and fix old ones
			if len(new_group_nodes) > 0:
				groups[k] = list(set(v) - set(new_group_nodes))
				new_groups[new_group_num] = new_group_nodes
				new_group_num += 1
		
		if len(new_groups) == 0: break
		# update groups dict with new groups 
		groups.update(new_groups)	
		print("another update round", new_groups)
	print("----------")
	return groups

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	args = parser.parse_args()

	n = args.n

	# colluders = generate_graphs(3)
	# graphs = generate_graphs(n-3)

	# print(colluders)

	# graphs = connect_two_graphs(colluders, graphs, 9, n-3)
	
	# # to help with fixing the groups
	# graphs = generate_graphs(n)
	# groups = [determine_groups(G, n) for G in graphs]
	# for group in groups:
	# 	print(group)
