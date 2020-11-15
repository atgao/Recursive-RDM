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
    # need to calculate probability for each node 

    # print("Calculating prob for %s with n = %d" % (bits, n))
    if n == 2:
        if int(bits, 2) == 0:
            return np.array([0, 1])
        else:
            return np.array([1, 0])

    # get probability each node is not eliminated this round
    prob = np.zeros((n), dtype=np.float32)
    # go through each node and play them
    for i in range(n):
        round = G[i]
        keep = ~round 

        if np.all(round):
            keep[i] = True

        # determine which nodes to eliminate
        new_G = G[keep, :][:, keep]

        if new_G.shape[0] == 1:
            prob[keep] += 1.0/n 
            continue

        triu_inds = np.triu_indices(new_G.shape[0], 1)
        new_bits = "".join(str(i) for i in new_G[triu_inds].astype("uint8"))
        if not new_bits in ht:
            ht[new_bits] = calculate_prob(new_bits, new_G, new_G.shape[0], ht)
        prob[keep] += 1.0/n * ht[new_bits]

    ht[bits] = prob
    return prob

def get_manipulability(graphs, n, ht, s=3):
	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s)) 
	
	# current graphs for n
	bitgraphs = graphs

	# ways the subset can manipulate 
	subset_bitgraphs = generate_graphs(s)[1:]
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
				new_prob = ht[new_key]
				diff = cur[list(subset)] - new_prob[list(subset)]
				gain = np.sum(diff) # np.max instead??
				if gain > maxGain:
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
	print("Avg Time: %f sec per graph" %time.average_time)
	print("Total Time: %f sec" %time.total_time)
	
	gain = get_manipulability(graphs, n, ht, s=2)
	print("Total gain: ", gain)