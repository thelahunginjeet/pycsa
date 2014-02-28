"""
run_pipeline.py

Created by CA Brown and KS Brown.  For reference see:
Brown, C.A., Brown, K.S. 2010. Validation of coevolving residue algorithms via 
pipeline sensitivity analysis: ELSC and OMES and ZNMI, oh my! PLoS One, e10779. 
doi:10.1371/journal.pone.0010779

This is the main test script for the pipeline.  All of the files that are needed 
are included in the '../tests' directory.  The datasets is small enough (subset of the PDZ 
dataset from our publication) to run a small number of resamplings in a short amount of time.
"""

# import the two CEP modules used below and sys module
import CEPPipeline
import CEPPlotting
import sys

# pipeline parameters
mainDirectory = '/Users/kevinbrown/projects/correlated-substitutions/pdz/'
fileIndicator = 'pdz'
canonSequence = '1IU0'
alignmentFile = '/Users/kevinbrown/projects/correlated-substitutions/pdz/preprocessing/pdz_test.aln'
pdbFile = '/Users/kevinbrown/projects/correlated-substitutions/pdz/pdb/1iu0.pdb'
numPartitions = 100 
resamplingMethod = 'splithalf'
methods = ('chisq','mi','znmi')
pruning = 'topN'
distMethod = 'oneminus'
rescaled = True
repMethod = 'splithalf'
figDirectory = '/Users/kevinbrown/projects/correlated-substitutions/pdz/figures/'
simMethod = 'spearman'
pcLambda = 0.5

def run_pipeline():
	# run pipeline using parameters above
	pipe = CEPPipeline.CEPPipeline(mainDirectory=mainDirectory,fileIndicator=fileIndicator,canonSequence=canonSequence,
								numPartitions=numPartitions,resamplingMethod=resamplingMethod)
    
    # this cleans the network and alignment directories
	pipe.clean_project()
	pipe.resample_alignment(alignmentFile=alignmentFile)
	for meth in methods:
		pipe.calculate_networks(method=meth,pl=pcLambda)
		pipe.calculate_graphs(pruning=pruning)
		pipe.calculate_resampling_statistics(distMethod=distMethod,rescaled=rescaled,repMethod=repMethod,simMethod=simMethod,pdbFile=pdbFile)
		pipe.write_database()


def plot_results():
	# read in the databases
	ceps = {}.fromkeys(methods)
	for meth in methods:
		dbFile = '/Users/kevinbrown/projects/correlated-substitutions/pdz/databases/pdz_800_'+meth+'_'+pruning+'.pydb'
		ceps[meth] = CEPPipeline.CEPPipeline(database=dbFile)
	plot = CEPPlotting.CEPPlotting(figDirectory=figDirectory)
	plot.plot_accuracy_reproducibility(ceps['znmi'],ceps['invcov'],ceps['psicov'],ceps['mfdi'])


"""
def test_pipeline():
	# TODO: ship initialization parameters into a class - will make it easier to collect all the
	#		metaparameters and optimize over them down the road
	# first establish a project by listing:
		#	'mainDirectory'  => base directory for project (note: makes directories here)
		#	'fileIndicator'  => short string which acts as the prefix of the files
		#	'canonSequence'  => case-sensitive string of the fasta header of the canonical sequence for numbering
		#	'numPartitions'	 => number of times to resample from the master alignment
		#   'resamplingMethod' => how to do the resampling
		#
		mi = CEPPipeline.CEPPipeline(mainDirectory='../tests/', fileIndicator='pdz', canonSequence='1IU0', numPartitions=100, resamplingMethod='splithalf')

	# issue this command to clean the network and alignment directories
	#	note: the pipeline won't overwrite network & alignment files so you have to issue 
	#	      clean project or do this manually by deleting the files
	mi.clean_project()

	# first, make subalignments by resampling
	#	'alignmentFile' => path name of the alignment file to be resampled
	#	note: this will create a directory in the 'mainDirectory' named 'alignments'
	mi.resample_alignment(alignmentFile='../tests/preprocessing/pdz_test.aln')

	# second, calculate the networks with an input method/algorithm
	#	'method' => case-insensitive name of an algorithm (e.g. mi, znmi, omes, sca, zres, mip)
	#	note: this will create a directory in the 'mainDirectory' named 'networks'
	mi.calculate_networks(method='mi')

	# calculate the graphs
	#	'pruning' => method by which to prune the dense networks of pair scores (e.g. mst, topn, bottomn)
	mi.calculate_graphs(pruning='mst')

	# calculate the resampling statistics and consensus graph
	#	'pdbFile' => path name of pdb file to be used for accuracy statistics
	#   'distMethod' => transformation of weighted distance
	#	'repMethod' => type of reproducibility (link, bicar, splithalf) to calculate
	mi.calculate_resampling_statistics(distMethod='neglog',rescaled=False,repMethod='link',pdbFile='../tests/pdb/1iu0.pdb)

	#mi.calculate_consensus_graph(pruning='mst', number=None, pdbFile='../tests/pdb/1iu0.pdb', offset=0)

	# return CEPObject for plotting
	return mi
"""

if __name__ == '__main__':
	# create the databases
	run_pipeline()

	# plot results from already created databases
	plot_results()
	
	# load plotting package
	#plot = CEPPlotting.CEPPlotting(figDirectory='../tests/')
	
	# pass mutual information pipeline object to the plotting function
	#	note: this can take a variable number of CEPPipelineObjects (e.g. plot.plot_accuracy_reproducibility(mi, sca, znmi))
	#plot.plot_accuracy_reproducibility(mi)	
	
	
	
	
	
