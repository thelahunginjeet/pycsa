import Bio.PDB as pdb
from numpy import argsort,sort

def calculate_distances(pdbFile,offset=0):
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
	for i in xrange(len(residues)):
		for j in xrange(i,len(residues)):
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
