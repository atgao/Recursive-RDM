# import networkx as nx
import numpy as np
import argparse
import itertools

import collections
from tqdm import tqdm 

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

def generate_graphs(n):
	'''
	n: number of nodes in graph
	return: list of all possible directed graphs
	'''
	sums = generate_sums(n)
	bitgraphs = []

	for sum in sums:
		bit = convert_sum_to_binary(sum, n)
		if bit != "":
			bitgraphs.append(bit)
	return bitgraphs

def check_permutation(bg1, bg2, n):
	G1 = convert_binary_to_graph(bg1, n)
	G2 = convert_binary_to_graph(bg2, n)

	perms = list(itertools.permutations(np.arange(n)))[1:]
	P = np.zeros((n, n), dtype=np.uint8)
	inds = np.arange(n)

	for perm in perms:
		P[inds, perm] = 1
		if np.all(P @ (G1 @ P.T) == G2):
			return True
		P[inds, perm] = 0 # reset
	return False

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

def generate_sums(n):
	e = get_num_edges(n)
	
	# first ordering always n-1, n-2, n-3....0
	nums = [i for i in range(1, n)]
	sums = combinationSum(nums, e, n)
	return sums

def combinationSum(nums, target, k):
	res = []
	nums.sort()
	def dfs(left, path, idx):
		if not left and len(path) <= k: res.append(path[::-1])
		else:
			for i, val in enumerate(nums[idx:]):
				if val > left: break
				if len(path) > 0:
					if val == nums[-1] and path[-1] == nums[-1]: 
						break
				dfs(left - val, path + [val], idx + i)
	dfs(target, [], 0)
	return res

def convert_sum_to_binary(sum, n):
	if len(sum) < n: sum.extend(0 for i in range(n-len(sum)))
	sum = collections.deque(sum)
	G = np.identity(n, dtype=np.bool_)

	for i in range(n):
		wins = np.sum(G[i]) - 1
		temp = sum.popleft()
		num = temp - wins
		remaining = n - i - 1

		count = 0
		while num > remaining:
			sum.append(temp)
			temp = sum.popleft()
			num = temp - wins 
			remaining = n - i - 1
			count += 1
			if count > len(sum): 
				# print(i, count, num, remaining)
				return ""

		# set row
		G[i][i+1:i+1+num] = ~G[i][i+1:i+1+num]
		# set col 
		G[i+1:, i] = ~G[i][i+1:]

	triu_inds = np.triu_indices(n, 1)
	bits = "".join(str(i) for i in G[triu_inds].astype("uint8"))

	return bits

def get_all_graphs(n, s=3):
	bitgraphs = generate_graphs(n)
	res = []
	res.extend(bitgraphs)
	manip = []

	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s))

	if s == 2:
		subset_bitgraphs = generate_graphs(s)
	else:
		subset_bitgraphs = generate_graphs(s)[1:]
	count = s-1

	for bits in bitgraphs:
		for subset in subsets:
			for sb in subset_bitgraphs:
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
				manip.append(new_key)
	# print(res)
	# print(2**get_num_edges(n), len(res), len(bitgraphs)) # comparison of how much we r saving
	return res, manip

def get_manipulability(graphs, n, ht, s=3):
	inds = np.arange(n)
	subsets = list(itertools.combinations(inds, s)) 
	
	# current graphs for n
	bitgraphs = graphs

	# ways the subset can manipulate 
	if s == 2:
		subset_bitgraphs = generate_graphs(s)
	else:
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
				diff = new_prob[list(subset)] - cur[list(subset)] 
				gain = np.sum(diff) # np.max instead??
				if gain > maxGain:
					maxGain = gain
	return maxGain

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', type=int, default=4)
	args = parser.parse_args()

	n = args.n

	# sums = generate_sums(n)
	# print(sums, len(sums))
	# bitgraphs = []

	# for sum in sums:
	# 	bitgraphs.append(convert_sum_to_binary(sum, n))
	
	# count = 0
	# for sum, bit in zip(sums, bitgraphs):
	# 	if bit != "": count += 1
	# 	print(sum, bit)
	# print("Total for %d graphs: %d " % (n, count))
	get_all_graphs(n)