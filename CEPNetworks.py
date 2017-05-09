"""
CEPNetworks.py

This module is used to control all of the pipeline flow and do
the reshuffling, etc.  It works with the CEPAlgorithms module which can
easily be adapted to include more algorithms.  The main pipeline is initialized
with all of the information for the rest of the project.

@author: Kevin S. Brown (University of Connecticut), Christopher A. Brown (Palomidez LLC)

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2014, Kevin S. Brown and Christopher A. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.

3. Neither the name of the University of Connecticut nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys, os, unittest, scipy.stats, re, copy, networkx
from numpy import array
from networkx import Graph as nxGraph
from pycsa.CEPLogging import LogPipeline

# decorator function to be used for logging purposes
log_function_call = LogPipeline.log_function_call

class CEPGraph(nxGraph):
	"""This is the main graph class (subclassed from networkx.Graph) for computing statistics on networks
	Note: all of the functions below assume a weighted graph (setup in __init__)"""
	def __init__(self,nwkFile=None):
		"""Initialize with a network file from the pipeline"""
		try:
			# initialize the networkx.Graph class first
			super(CEPGraph,self).__init__()
			self.read_network(nwkFile)
			if self.is_weighted():
				self.weighted = True
			else:
				raise CEPGraphWeightException
		except IOError:
			raise CEPGraphIOException(nwkFile)
		except TypeError:
			super(CEPGraph,self).__init__()

	def is_weighted(self):
		edges = {'weight':False}
		for v1,v2 in self.edges():
			if 'weight' in self[v1][v2]:
				edges['weight'] = True
				break
		return edges['weight']

	def read_network(self,nwkFile):
		"""Read in a network file to the current graph object"""
		nwkFile = open(nwkFile,'r')
		network = nwkFile.readlines()
		nwkFile.close()
		# add edges to self (note: (+)int nodes, (+/-)float edges, (+)float p-values)
		for edge in network:
			link = re.search('(\d+)\t(\d+)\t(-?\d+\.\d+)\t(\d+\.\d+)',edge)
			self.add_edge(int(link.group(1)), int(link.group(2)), weight=float(link.group(3)),pvalue=float(link.group(4)))

	def compute_node_degrees(self):
		"""Computes the node degree (weighted sum if applicable) for a graph"""
		degrees = {}
		for node in self.nodes_iter():
			knode = 0.0
			for neighbor in self.neighbors_iter(node):
				knode += self.get_edge_data(node,neighbor)['weight']
			degrees[node] = knode
		# get half sum of node degrees as well
		halfDegreeSum = 0.5*(array(degrees.values()).sum())
		return degrees, halfDegreeSum

	def prune_graph(self, threshold):
		"""Removes all weighted edges below a certain threshold along with any nodes
		that have been orphaned (no neighbors) by the pruning process"""
		for v1,v2 in self.edges():
			if self[v1][v2]['weight'] < threshold:
				self.remove_edge(v1,v2)
		for n in self.nodes():
			if len(self.neighbors(n)) < 1:
				self.remove_node(n)

	def calculate_pvalue(self,number=None):
		"""Removes edges that aren't significant given their p-values (p > 0.05)
		Note: all MHT corrections, etc. should be taken care of in the method and
		not here (see CEPAlgorithms)"""
		edges = self.edges()
		for v1,v2 in edges:
			if self[v1][v2]['pvalue'] > 0.05:
				self.remove_edge(v1,v2)

	def calculate_mst(self,number=None):
		"""Calculates a maximal spanning tree mapping large weights to small weights"""
		graph = copy.deepcopy(self)
		maxWeight = max([self[v[0]][v[1]]['weight'] for v in self.edges()])
		for v1,v2 in self.edges():
			graph[v1][v2]['weight'] = maxWeight - self[v1][v2]['weight']
		edges = self.edges()
		tree = networkx.minimum_spanning_tree(graph,weight='weight')
		for v1,v2 in edges:
			if (v1,v2) in tree.edges():
				pass
			else:
				self.remove_edge(v1,v2)

	def calculate_top_n(self,number):
		"""Removes edges except for those with the n-largest weights (n = number)"""
		weights = [(self[v[0]][v[1]]['weight'],v[0],v[1]) for v in self.edges()]
		weights.sort()
		weights.reverse()
		# only keep n-largest vertex pairs
		weights = [(v[1],v[2]) for v in weights[:number]]
		edges = self.edges()
		for v1,v2 in edges:
			if (v1,v2) in weights:
				pass
			else:
				self.remove_edge(v1,v2)

	def calculate_bottom_n(self,number):
		"""Removes edges except for those with the n-smallest weights (n = number)"""
		weights = [(self[v[0]][v[1]]['weight'],v[0],v[1]) for v in self.edges()]
		weights.sort()
		# only keep n-smallest vertex pairs
		weights = [(v[1],v[2]) for v in weights[:number]]
		edges = self.edges()
		for v1,v2 in edges:
			if (v1,v2) in weights:
				pass
			else:
				self.remove_edge(v1,v2)

	# this can be removed once the GraphSimilarity module is operational
	def compute_jaccard_index(self,graph):
		"""Computes the Jaccard index for edges between self and another graph.
		note: Jaccard index = edge intersection divided by edge union"""
		union = frozenset(self.edges()).union(graph.edges())
		intersection = frozenset(self.edges()).intersection(graph.edges())
		try:
			jaccard = float(len(intersection))/len(union)
		except ZeroDivisionError:
			jaccard = 0.0
		return jaccard

class CEPGraphIOException(IOError):
	@log_function_call('ERROR : Network File Input')
	def __init__(self,nwkFile):
		print "The network file you have provided, '%s', does not exist.  Please check your file selection."%(nwkFile)

class CEPGraphWeightException(Exception):
	@log_function_call('ERROR : Graph Not Weighted')
	def __init__(self):
		print "The graph you have provided is not a weighted graph.  Most of the methods provided are pointless for binary graphs."

class CEPNetworksTests(unittest.TestCase):
	def setUp(self):
		pass # TODO add unit tests to CEPNetworks

if __name__ == '__main__':
	unittest.main()
