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

from collections import defaultdict

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
					# print("diff: ", diff, gain)
					# print("-------------")
					maxGain = gain
	return maxGain

def get_manipulability_higher_nodes(graphs, manips, n, ht, s=3):
	subset = [i for i in range(s)]

	maxGain = float('-inf')

	print(len(manips.values()), len(graphs))
	for graph, v in manips.items():
		for manip in v:
			diff = ht[manip][subset] - ht[graph][subset]

			gain = np.sum(diff)
			
			if gain > maxGain:
				maxGain = gain
	return maxGain

def gentourng(n):
	res = subprocess.check_output("./nauty27r1/gentourng %d" % n, shell=True)
	return res.decode().splitlines()

def count_difference(bit1, bit2):
	count = 0
	for b1, b2 in zip(bit1, bit2):
		if b1 != b2: 
			count += 1
	return count

def connect_two_graphs(colluders, graphs, k, n, s=3):
	connected_graphs = set()
	manip_connected_graphs = defaultdict(set)

	colluding_groups = [determine_groups(G, s) for G in colluders]

	colluding_combos = {}
	for group, G in zip(colluding_groups, colluders):
		combos = []
		for i in range(1, s+1):
			combos.append(list(itertools.combinations(group.values(), i))) # TODO: make sure to go thru keys
		colluding_combos[G] = combos 
	all_colluders = generate_graphs(s, unique=False) 

	# print(colluding_groups)
	# print("colluding combos: ", colluding_combos)

	conns = np.ones((s, n), dtype=np.bool_) # to clear the set nodes later on
	for graph in graphs: 
		G = np.ones((n+s, n+s), dtype=np.bool_)
		G_manip = np.ones((n+s, n+s), dtype=np.bool_)

		G[s:, s:] = convert_binary_to_graph(graph, n)
		G_manip[s:, s:] = convert_binary_to_graph(graph, n)

		group = determine_groups(graph, n)
		group_combos = []

		for i in range(1, n+1):
			group_combos.append(list(itertools.combinations(group.values(), i)))

		for combos in group_combos:
			if not combos: continue

			for combo in combos: # further unpacks the combo again
				(*unpacked_combo,) = combo
				if len(unpacked_combo) > 1:
					unpacked_combo.append(list(itertools.chain.from_iterable(unpacked_combo))) # <--- double check this later...
				# print("UPDATED OTHER GRAPH UNPACKED ", unpacked_combo)

				for colluder in colluders:
					group_combos_to_beat = colluding_combos[colluder]
					G[:s, :s] = convert_binary_to_graph(colluder, s) # set up colluder graph 

					# one graph where NONE of colluders are beat
					# print("GROUP COMBOS TO BEAAAAAAT ", group_combos_to_beat)
					triu_inds = np.triu_indices(s+n, 1)
					new_bits = "".join(str(b) for b in G[triu_inds].astype("uint8"))
					connected_graphs.add(new_bits)

					for manip in all_colluders: # test ??
						if manip != colluder:
							G_manip[:s, :s] = convert_binary_to_graph(manip, s)
							manip_bits = "".join(str(b) for b in G_manip[triu_inds].astype("uint8"))
							manip_connected_graphs[new_bits].add(manip_bits)
					
					for combos_to_beat in group_combos_to_beat:
						# print("COMBOS TO BEAT ", combos_to_beat)
						if not combos_to_beat: continue
						for combo_to_beat in combos_to_beat: # further unpack the combos 
							(*unpacked_combo_to_beat,) = combo_to_beat
							if len(unpacked_combo_to_beat) > 1:
								unpacked_combo_to_beat.append(list(itertools.chain.from_iterable(unpacked_combo_to_beat)))
							# print("unbacked combos to beat ", unpacked_combo_to_beat)
							for unpacked_group_to_beat in unpacked_combo_to_beat:
								# print("TO BEAT ", unpacked_group_to_beat)
								for unpacked_group in unpacked_combo:
									# print("CURRENT UNPACKED GROUP ", unpacked_group)
									for i in range(1, len(unpacked_group)+1):
										for j in range(1, len(unpacked_group_to_beat)+1):
											if i * j >= k: 
												# print("CONTINUING...")
												continue
											non_colluders = np.array(unpacked_group[:i])
											beaten_colluders = np.array(unpacked_group_to_beat[:j])

											G[np.ix_((beaten_colluders), (s+non_colluders))] = False
											G_manip[np.ix_((beaten_colluders), (s+non_colluders))] = False

											# build the new graph and append it
											triu_inds = np.triu_indices(s+n, 1)
											new_bits = "".join(str(b) for b in G[triu_inds].astype("uint8"))
											connected_graphs.add(new_bits)

											# build the manip graph
											for manip in all_colluders: # test ??
												if manip != colluder:
													G_manip[:s, :s] = convert_binary_to_graph(manip, s)
													manip_bits = "".join(str(b) for b in G_manip[triu_inds].astype("uint8"))
													manip_connected_graphs[new_bits].add(manip_bits)

											# print(s+non_colluders, " beats ", beaten_colluders, " with graph ", new_bits)
											G[:s, s:] = conns
											G_manip[:s, s:] = conns
											# print("AFTER CLEAR")
											# print(G.astype('uint8'))
											# print("------------")
			# print("---------------------------------")

	for k, v in manip_connected_graphs.items():
		manip_connected_graphs[k] = list(v)

	return list(connected_graphs), manip_connected_graphs

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
		new_groups = {}
		for k, v in groups.items():
			losers = np.argwhere(graph[v[0]] == 1)[:, 0] if len(np.argwhere(graph[v[0]] == 1)) > 0 else []
			nodes_beat = set(losers) - set(v)

			new_group_nodes = []
		
			for node in v[1:]:
				losers = np.argwhere(graph[node] == 1)[:, 0]
				# if the losers to node aren't in the set or aren't one of the nodes already in the group
				if set(losers) - set(v) != nodes_beat:
					new_group_nodes.append(node)

			# add new groups and fix old ones
			if len(new_group_nodes) > 0:
				groups[k] = list(set(v) - set(new_group_nodes))
				new_groups[new_group_num] = new_group_nodes
				new_group_num += 1
		
		if len(new_groups) == 0: break
		# update groups dict with new groups 
		groups.update(new_groups)

	# sanity check 
	# for k, group in groups.items(): 
	# 	all_nodes_beat = set(np.argwhere(graph[group[0]] == 1)[:, 0]) - set(group)
	# 	for node in group[1:]:
	# 		losers = np.argwhere(graph[node] == 1)[:, 0]	
	# 		if set(losers) - set(group) != all_nodes_beat:
	# 			print("ERROR on graph ", G)
	return groups

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	args = parser.parse_args()

	n = args.n

	colluders = generate_graphs(3)
	graphs = generate_graphs(n-3)

	# print(colluders)

	graphs, manip = connect_two_graphs(colluders, graphs, 8, n-3)
	print(len(graphs))
	print(graphs)

	# for bitgraph in graphs+list(manip.values()):
	# 	print(bitgraph)
	
	# # # to help with fixing the groups
	# graphs = generate_graphs(n)
	# groups = [determine_groups(G, n) for G in graphs]
	# for graph, group in zip(graphs, groups):
	# 	print(graph, ": ", group)

	# # G = "1100110111"
	# # print(determine_groups(G, 5))

	# colluders = generate_graphs(3)
	# graphs = generate_graphs(n-3)

	# graphs, manip = connect_two_graphs(colluders, graphs, 8, n-3)
	# for k, v in manip.items():
	# 	print(k, " : ", len(v), v)
	# print(graphs)
	# print(len(manip.keys()), len(manip.values()))
	# ht = {}

	# # for k, v in manip.items():
	# # 		# time.tic()
	# # 		G = convert_binary_to_graph(k, n)
	# # 		calculate_prob(k, G, n, ht)

	# # 		for bitgraph in v:
	# # 			G = convert_binary_to_graph(bitgraph, n)
	# # 			calculate_prob(bitgraph, G, n, ht)
	# # 		# time.toc()
	# # gain = get_manipulability_higher_nodes(graphs, manip, n, ht, s)
	# # print("GAIN FOR n = %d is %d " % (n, gain))

