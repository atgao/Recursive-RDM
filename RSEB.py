import networkx as nx
import numpy as np
import argparse
import itertools

import collections
import math

from tqdm import tqdm 
from utils.graph_utils import *
from utils.timer import Timer

def calculate_prob(bits, G, n, matches = [], ht={}, dummy=False):
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
    
    if not matches:
        n_prime = 2**math.ceil(math.log(n, 2))
        matches = generate_matches(np.arange(n_prime), {})
        dummy = n_prime-n != 0

    # get probability each node is not eliminated this round
    prob = np.zeros((n), dtype=np.float32)
    inds = np.arange(n)
    # determine eliminated pairs
    for match in matches: 
        elim = []
        # gather the eliminated nodes 
        for k, v in match.items():
            if not dummy:
                elim.append(v) if G[k][v] else elim.append(k)
            else:
                if k < n and v < n:
                    elim.append(v) if G[k][v] else elim.append(k)
        keep = np.delete(inds, elim)
        
        # determine which nodes to eliminate
        new_G = G[keep, :][:, keep]
        if new_G.shape[0] == 1:
            prob[keep] += 1.0/len(matches) 
            continue

        triu_inds = np.triu_indices(new_G.shape[0], 1)
        new_bits = "".join(str(i) for i in new_G[triu_inds].astype("uint8"))

        if not new_bits in ht:
            ht[new_bits] = calculate_prob(new_bits, new_G, new_G.shape[0], ht=ht)
        prob[keep] += 1.0/len(matches) * ht[new_bits]
    
    ht[bits] = prob
    return prob

def generate_matches(nodes, edges):
    if len(nodes) == 0:
        return [edges.copy()]
    
    start = nodes[0]
    acc = []
    for i in range(1, len(nodes)):
        end = nodes[i]
        edges[start] = end 
        
        inds = np.ones(len(nodes), dtype=np.bool_)
        inds[0], inds[i] = False, False
        unused = nodes[inds]

        match = generate_matches(unused, edges)
        
        acc.extend(match)
        edges.pop(start, None)

    return acc

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

    # should store all matches from 1 - 2^floor(log_2 n)
    n_prime = 2**math.ceil(math.log(n, 2))
    matches = generate_matches(np.arange(n_prime), {})
    print("Total: %d matches for %d wtih %d dummy players" % (len(matches), n, n_prime-n))

    time = Timer()
    graphs = generate_graphs(n)
    print("finished generating graphs", len(graphs))
    
    ht = {}
    
    for bitgraph in tqdm(graphs):
        time.tic()
        G = convert_binary_to_graph(bitgraph, n)
        calculate_prob(bitgraph, G, n, matches, ht, n_prime-n != 0)
        time.toc()
    
    for k, v in ht.items():
        print(k, v)

    print(len(ht))
    prev_total = time.total_time
    print("Avg Time: %f sec per graph" %time.average_time)
    print("Total Time: %f sec" %time.total_time)
    
    time.tic()
    gain = get_manipulability(graphs, n, ht)
    time.toc()
    print("Total gain: ", gain)
    print("Total Time: %f sec" % time.total_time - prev_total)