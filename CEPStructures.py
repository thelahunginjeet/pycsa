"""
CEPStructures.py

This module is calculates distances and contacts from pdb files/structures.

This is a pure python package for correlated substitution analysis.  Specifically, it can be
used for the kinds of analyses in the following publication:

"Validation of coevolving residue algorithms via pipeline sensitivity analysis: ELSC and
OMES and ZNMI, oh my!" PLoS One, e10779. doi:10.1371/journal.pone.0010779 (2010)

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

import Bio.PDB as pdb
from numpy import argsort,sort

def calculate_distances(pdbFile,modelNumber=0,chain='A'):
	"""
	From an input PDB file, compute the residue-residue distance and return a dictionary on the edges that
	gives the minimal distance between two residues in Angstroms, calculated as the minimum distance between
	Cb atoms (Ca if residue == Gly).  Ignores waters or any other funky ligands.  Specific chain/model
	combinations can be supplied, but will default to something that exists (with a warning) if either one
	or both keys are invalid.

	Returns the distances as a dictionary keyed on (pos1,pos2).
	"""
	distances = {}
	strucParser = pdb.PDBParser()
	pdbStruc = strucParser.get_structure('pdb',pdbFile)
	if not pdbStruc.child_dict.has_key(modelNumber):
		modelNumber = pdbStruc.child_dict.keys()[0]
		print('WARNING: Model number invalid, using model',modelNumber)
	if not pdbStruc[modelNumber].child_dict.has_key(chain):
		chain = pdbStruc[modelNumber].child_dict.keys()[0]
		print('WARNING: Chain invalid, using chain',chain)
	# fetch the AAs
	residues = [pdbStruc[modelNumber][chain].child_dict]
	distances = {}
	aminoAcids =  ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
					'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
	# using a mapping of AA to atom type (Cb,Ca) avoids lots of annoying case-handling in the distance calculations
	atomType = {}.fromkeys(aminoAcids)
	for k in atomType:
		atomType[k] = 'CB'
	atomType['GLY'] = 'CA'
	for i in range(len(residues)):
		for j in range(i,len(residues)):
			# loop over dicts of residues
			for rk in [x for x in residues[i] if residues[i][x].resname in aminoAcids]:
				for rl in [x for x in residues[j] if x != rk and residues[j][x].resname in aminoAcids]:
					aTk = atomType[residues[i][rk].resname]
					aTl = atomType[residues[j][rl].resname]
					try:
						# the real mapping (Cb for all non-GLY, Ca for GLY)
						atomk = residues[i][rk][aTk]
					except:
						# fallback (try Ca)
						atomk = residues[i][rk]['CA']
					try:
						# same as above
						atoml = residues[j][rl][aTl]
					except:
						atoml = residues[j][rl]['CA']
					distances[(rk[1],rl[1])] = atomk - atoml
	return distances

# This function handles the offset, which is an extremely special case; will need to fix this up
def calculate_distances_offset(pdbFile,offset=0):
	"""
	From an input PDB file, compute the residue-residue distance and return a dictionary on the edges that
	gives the minimal distance between two residues in Angstroms, calculated as the minimum distance between
	Cb atoms (Ca if residue == Gly).  Ignores waters or any other funky ligands.  Optionally, an offset can be set
	that assumes two concatenated proteins.

	Returns the distances as a dictionary keyed on (pos1,pos2).
	"""
	distances = {}
	strucParser = pdb.PDBParser()
	pdbStruc = strucParser.get_structure('pdb',pdbFile)
	modelNumber = pdbStruc.child_dict.keys()[0]
	if offset > 0:
		chain1,chain2 = pdbStruc[modelNumber].child_dict.keys()[:2]
		residues = [pdbStruc[modelNumber][chain1].child_dict,pdbStruc[modelNumber][chain2].child_dict]
	else:
		chain = pdbStruc[modelNumber].child_dict.keys()[0]
		residues = [pdbStruc[modelNumber][chain].child_dict]
	distances = {}
	aminoAcids =  ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
	# using a mapping of AA to atom type (Cb,Ca) avoids lots of annoying case-handling in the distance calculations
	atomType = {}.fromkeys(aminoAcids)
	for k in atomType:
		atomType[k] = 'CB'
	atomType['GLY'] = 'CA'
	for i in range(len(residues)):
		for j in range(i,len(residues)):
			# setup chain shifts
			if i + j == 0:
				di,dj = 0,0
			elif i + j == 1:
				di,dj = 0, offset
			elif i + j == 2:
				di,dj = offset,offset
			# loop over dicts of residues
			for rk in [x for x in residues[i] if residues[i][x].resname in aminoAcids]:
				for rl in [x for x in residues[j] if x != rk and residues[j][x].resname in aminoAcids]:
					aTk = atomType[residues[i][rk].resname]
					aTl = atomType[residues[j][rl].resname]
					try:
						# the real mapping (Cb for all non-GLY, Ca for GLY)
						atomk = residues[i][rk][aTk]
					except:
						# fallback (try Ca)
						atomk = residues[i][rk]['CA']
					try:
						# same as above
						atoml = residues[j][rl][aTl]
					except:
						atoml = residues[j][rl]['CA']
					distances[(rk[1]+di,rl[1]+dj)] = atomk - atoml
	return distances


def calculate_CASP_contacts(distances,angCut=8.0,distCut=0):
	"""
	Takes an input dictionary of residue-residue distances to form a list of pairs of amino acids deemed
	to be "in contact," as defined in the CASP competition.  Here, in contact means:
		1. d(Cb_i,Cb_j) <= angCut (Ca is used in the case of Glycines)
			atom-atom distances are separated by angCut anstroms or fewer
		2. |i - j| >= distCut
			positions must be at least distCut residues apart (ignoring nearest neighbors, for example)
	The contacts are returned in order of increasing distance; thus if you want the N closest contacts in the
	structure, take caspPairs[0:N].

	Contact pairs are returned as a list of tuples (r1,r2), in which r1 < r2.
	"""
	# this takes care of #1
	closePairs = [k for k in distances if distances[k] < angCut]
	# now prune for #2
	caspPairs = [k for k in closePairs if abs(k[0] - k[1]) > distCut]
	# now sort the close pairs
	sortOrder = argsort([distances[x] for x in caspPairs])
	# we take every other key in the sorted list because (n1,n2) and (n2,n1) appear right next
	#   to each other
	return [tuple(sort(caspPairs[x])) for x in sortOrder][0::2]
