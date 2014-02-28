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

from pycsa import CEPPipeline,CEPPlotting,CEPLogging
import sys,os

# parameters/meta-parameters
# directory-related
mainDirectory = './pdztest/'            # base project directory
alignmentFile = 'pdz_test.aln'          # master alignment file
pdbFile = '1iu0.pdb'                    # structure for accuracy calculation
figDirectory = 'figures/'     # locations for plots
# file-related
fileIndicator = 'pdz'                   # prefix for network/graph files
canonSequence = '1IU0'                  # FASTA identifier for canonical sequence
# resampling
resamplingMethod = 'splithalf'          # sets resampling ensemble
numPartitions = 10                      # size of resampling ensemble
repMethod = 'splithalf'                 # method for computing reproducibility
simMethod = 'spearman'                  # method for computing pairwise/groupwise graph similarity
# symbol counting
pcType = 'inc'                          # pseudocounting type
pcLambda = 1.0                          # number of scalar pseudocounts (for pcType='fix')
pcMix = 0.5                             # mixture parameter (for pcType = 'inc')
gap = 0.1                               # columns with %(gaps) > gap will not be scored
swtMethod = 'unweighted'                # method for downweighting similar sequences
cutoff = 0.68                           # groups of sequences with fractional_similarity > cutoff share unit weight
# method-related
pruning = 'mst'                         # graph pruning method
number = 90                             # number of edges to keep (ignored for pruning = 'mst')
# accuracy
accMethod = 'distance'                  # metric for accuracy
distMethod = 'oneminus'                 # definition of distance (for accMethod = 'distance')
rescaled = True                         # rescale distances so accuracy is in [0,1]

def run_pipelines(methodList):
    # create logfile
    #   note: if the logfile already exists, new time-stamped logging statements will be appended to this file
    logger = CEPLogging.LogPipeline('./pdzlog.txt')

    # run pipeline using parameters above
    #   'mainDirectory' => base project directory; directories/files are created here
    #   'fileIndicator' => prefix for the network and alignment files
    #   'canonSequence' => case-sensitive string of the FASTA header (or STO sequence identifier) for the canonical sequence used for position numbering
    #   'numPartitions' => number of resamples
    #   'resamplingMethod' => type of resampling (splithalf, bootstrap)
    pipe = CEPPipeline.CEPPipeline(mainDirectory=mainDirectory,fileIndicator=fileIndicator,canonSequence=canonSequence,
                                numPartitions=numPartitions,resamplingMethod=resamplingMethod)
    
    # issue this command to clean the network and alignment directories
    #   note: the pipeline won't overwrite network & alignment files (to avoid unnecessary recalculation) so you have to issue 
    #          clean project or manually by delete the files
    pipe.clean_project()

    # first, make subalignments by resampling
    #   'alignmentFile' => path name of the master alignment file to be resampled
    #   note: this will create a directory in the 'mainDirectory' named 'alignments'
    pipe.resample_alignment(alignmentFile=alignmentFile)

    # calculate the network; pl is the pseudocounting admixture parameter
    #   note: this creates a directory in the 'mainDirectory' named 'networks'
    pipe.calculate_networks(methodList=methodList,pcLambda=pcLambda,gap=gap,pcType=pcType,swtMethod=swtMethod,cutoff=cutoff)
    
    # calculate graphs, resampling stats, and write a results database for each method
    #   'pruning' => method by which to prune the dense networks of pair scores (e.g. mst, topn, bottomn)
    #   'pdbFile' => path name of pdb file to be used for accuracy statistics
    #   'distMethod' => transformation of weighted distance
    #   'repMethod' => type of reproducibility (bicar, splithalf) to calculate
    #   'simMethod' => way to measure graph similarity for reproducibility measures which compare graphs; will be ignored
    #                    if a reproducibility measure that does not use this information is chosen.
    for m in methodList:
        pipe.calculate_graphs(method=m,pruning=pruning,number=number)
        pipe.calculate_resampling_statistics(accMethod=accMethod,repMethod=repMethod,distMethod=distMethod,rescaled=rescaled,
                                            simMethod=simMethod,pdbFile=pdbFile)
        pipe.write_database()

    return


def plot_pipelines(methodList):
    # read in the databases written by run_pipeline
    ceps = {}.fromkeys(methodList)
    for meth in methodList:
        dbFile = mainDirectory+'databases/pdz_800_'+meth+'_'+pruning+'.pydb'
        ceps[meth] = CEPPipeline.CEPPipeline(database=dbFile)
    plot = CEPPlotting.CEPPlotting(figDirectory=mainDirectory+figDirectory)
    #plot.plot_accuracy_reproducibility(ceps['znmi'],ceps['invcov'],ceps['psicov'],ceps['mfdi'])
    plot.plot_accuracy_reproducibility(*ceps.values())


if __name__ == '__main__':
    # list of methods
    methodList = ['mi','mip','fchisq']

    # run
    run_pipelines(methodList)

    # plot results
    plot_pipelines(methodList)
