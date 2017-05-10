"""
CEPPipeline.py

This module is used to control all of the pipeline flow and do
the reshuffling, etc.  It works with the CEPAlgorithms module which can
easily be adapted to include more algorithms.  The main pipeline is initialized
with all of the information for the rest of the project.

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

import sys, os, unittest, random, glob, cPickle, re, itertools
from scipy import mean
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
import numpy as np
import Bio.PDB as pdb
from pycsa.CEPPreprocessing import SequenceUtilities
from pycsa.CEPLogging import LogPipeline
from pycsa import CEPAlgorithms,CEPNetworks,CEPAccuracyCalculator,CEPGraphSimilarity


# decorator function to be used for logging purposes
log_function_call = LogPipeline.log_function_call

def construct_file_name(pieces, subset, extension):
    """Simple function to create a file name from pieces, then add subset plus file extension"""
    return reduce(lambda x,y: str(x)+'_'+str(y),pieces) + subset + extension

def deconstruct_file_name(name):
    """Simple function to break apart a file name into individual pieces without the extension"""
    return os.path.splitext(os.path.split(name)[1])[0].split('_')

def determine_split_subset(name):
    """Simple function to return the split (integer) and the subset ('a' or 'b')"""
    tmp = re.findall('\d+',name)[0]
    return int(tmp),name.replace(tmp,'')

def sample_with_replacement(pop,k):
    """Returns a list of k samples from input population, sampling with replacement"""
    n = len(pop)
    _random, _int = random.random, int # speed in the loop
    return [pop[_int(_random() * n)] for i in itertools.repeat(None,k)]


def map_to_canonical(msa, datadict, canonical):
    """Accept an input MSA object (from CEPAlgorithms.MSA or .MSAAlgorithms) and
    a dictionary keyed on alignment position and returns a dictionary keyed on
    position in the canonical sequence.

    INPUT
    ------
        msa : CEPAlgorithms.MSA object, required

        datadict : dictionary, required
            the datadict can have either single positional keys or a tuple of
            positions (p1,p2,...,pN).  The tuple can be any size (2 is the most
            typical case).  Input positions are assumed to be in aligment space

        canonical : string, required
            this should be a key to one of the sequences in the MSA object

    OUTPUT
    ------
        mapped : dictionary
            the dictionary will have the same key structure (single or tuple) as
            the datadict, but all positions will be with respect to the canonical
            sequence.  Aligment positions with no canonical equivalent are dropped,
            so the sizes of mapped and datadict will be quite different"""
    canon = {}
    mapped = {}
    for c in msa.columns:
        canon[c] = msa.mapping[c][canonical]
    for k, v in datadict.items():
        if type(k) == tuple:
            if sum([canon[x] == None for x in k]) == 0:
                newkey = tuple([canon[x] for x in k])
                mapped[newkey] = v
        else:
            if canon[k] is not None:
                newkey = canon[k]
                mapped[newkey] = v
    return mapped


class CEPParameters(object):
    '''
    Container class to hold the myriad set of parameters for the processing pipeline,
    and to do checking for inconsistent options.
    '''
    def __init__(self,main_directory,alignment_file,canon_sequence):
        '''
        Only three absoultely necessary parameters are:
            main_directory : base project directory
            alignment_file : master alignment file (FASTA or STOCKHOLM)
            canon_sequence : canonical sequence for position numbering

        Default parameters and their values:
            pdb_file          : None (no accuracy will be calculated)
            fig_directory     : 'figures/'
            file_indicator    : 'cep'
            resampling_method : 'splithalf' (split-half resampling)
            num_partitions    : 100 (100 half-splits)
            pc_type           : 'inc' (pseudocount value scales with n. of seqs.)
            pc_lambda         : 1.0 (only used if pc_type = 'fix')
            pc_mix            : mixture parameter for Preal/Ppseudo (only for pc_type = 'inc')
            gap               : 1.0 (score all columns, no matter how many gaps)
            swt_method        : 'henikoff'
            cutoff            : groups of sequences with fractional_similarity > cutoff share unit weight
                    (not used for henikoff weighting or no weighting)
            pruning           : 'mst' (how to prune dense graphs for consensus graph calculation)
            number            : 100 (the n used for 'topn' or 'bottomn' pruning)
            acc_method        : 'avgdist' (use the original scaled average distance definition
                    of Brown and Brown)
            sim_method        : 'spearman' (how to quantify similarity between split graphs)
        '''
        # parameters passed in
        self.main_directory = main_directory
        self.alignment_file = alignment_file
        self.canon_sequence = canon_sequence
        # defaults for others
        self.fig_directory = 'figures/'
        self.pdb_file = None
        self.file_indicator = 'cep'
        self.resampling_method = 'splithalf'
        self.num_partitions = 100
        self.pc_type = 'inc'
        self.pc_lambda = 1.0
        self.pc_mix = 0.1
        self.gap = 1.0
        self.swt_method = 'henikoff'
        self.cutoff = 0.68
        self.pruning = 'mst'
        self.number = 100
        self.acc_method = 'avgdist'
        self.sim_method = 'spearman'


    def set_parameters(self,**kwargs):
        '''
        Parameters can be set one at a time directly, but this wrapper just cleans
        that up by allowing multiple kwargs to be passed in at one time.
        '''
        for k in kwargs:
            setattr(self,k,kwargs[k])



class CEPPipeline(object):
    """This is the main pipeline class for constructing partitions and networks"""
    @log_function_call('Initialized Pipeline')
    def __init__(self,mainDirectory=None, fileIndicator='cep', canonSequence=None, numPartitions=150, database=None, resamplingMethod='splithalf'):
        """ Main constructor for a project:

        fileIndicator -> output string for partitions; default is cep
        canonSequence -> header of the sequence used for numbering of networks; None!
        mainDirectory -> location of ./alignments, ./networks, ./graphs, etc.
        numPartitions -> the number of splits to make (e.g. 200)
        resamplingMethod -> resampling plan to use:
            splithalf : non-overlapping half splits
            bootstrap : basic bootstrap

        Note that not all reproduciblity and accuracy definitions are consistent with all resampling plans.
        """
        if database is not None:
            # start a project by reading in an archived database
            if os.path.exists(database):
                self.read_database(database)
            else:
                raise CEPPipelineDatabaseIOException(database)
        elif database == None and mainDirectory is not None:
            if not os.path.exists(mainDirectory):
                os.mkdir(mainDirectory)
            # establish basic instance variables for a new project
            self.mainDirectory = mainDirectory
            self.databaseDirectory = os.path.join(self.mainDirectory,'databases')
            self.networkDirectory = os.path.join(self.mainDirectory,'networks')
            self.fileIndicator = fileIndicator
            self.canonSequence = canonSequence
            self.numPartitions = numPartitions
            # check for a valid resampling plan
            self.resamplingMethod = '_resample_'+resamplingMethod
            if not hasattr(self,self.resamplingMethod):
                raise CEPPipelineResamplingMethodException(resamplingMethod)
        else:
            raise CEPPipelineDirectoryIOException


    def __str__(self):
        """String representation of the pipeline."""
        strRep = 'Correlated Substitution Analysis Pipeline\n'
        strRep += '   Results stored in '+self.mainDirectory+'\n'
        strRep += '   Canonical sequence : '+self.canonSequence+'\n'
        strRep += '   Resampling Method: '+self.resamplingMethod+'\n'
        return strRep


    @log_function_call('Cleaning Pipeline Project')
    def clean_project(self):
        """Removes all of the files from the partition and network directories along with directory"""
        partition = os.path.join(self.mainDirectory,'alignments')
        network = os.path.join(self.mainDirectory,'networks')
        try:
            alnFiles=os.listdir(partition)
            nwkFiles=os.listdir(network)
        except OSError:
            pass
        else:
            for f in alnFiles:
                os.remove(os.path.join(partition,f))
            for f in nwkFiles:
                os.remove(os.path.join(network,f))


    @log_function_call('Reading Pipeline Database')
    def read_database(self,dbFile):
        """Reads in a database file with a dictionary keyed by:
            dict['graph'] = <networkx graph object>
            dict['statistics'] = statistics
            dict['metadata'] = <instance variables>"""
        # this should set all of the instance variables from the file name
        try:
            dbFile = open(dbFile,'rb')
        except IOError:
            raise CEPPipelineDatabaseIOException(dbFile)
        else:
            dictionary = cPickle.load(dbFile)
            self.consensusGraph = dictionary['graph']
            self.statistics = dictionary['statistics']
            for attribute in dictionary['metadata']:
                self.__dict__[attribute] = dictionary['metadata'][attribute]
            dbFile.close()


    @log_function_call('Writing Pipeline Database')
    def write_database(self):
        """Writes a database file with a dictionary keyed by:
            dict['graph'] = <networkx graph object>
            dict['statistics'] = statistics
            dict['metadata'] = <instance variables>"""
        # write to the default database directory ./database
        if os.path.exists(self.databaseDirectory):
            pass
        else:
            os.mkdir(self.databaseDirectory)
        dbFile = construct_file_name([self.fileIndicator,self.numSequences,self.method], '', '.pydb')
        dbFile = os.path.join(self.databaseDirectory,dbFile)
        dbFile = open(dbFile,'wb')
        dictionary = {'graph':self.consensusGraph,'statistics':self.statistics, 'metadata':{}}
        for attribute in self.__dict__:
            dictionary['metadata'][attribute] = self.__dict__[attribute]
        cPickle.dump(dictionary,dbFile,-1)
        dbFile.close()



    @log_function_call('Resampling Subalignments from Master Alignment')
    def resample_alignment(self,alignmentFile,**kwargs):
        """Makes splits of an input alignment file and write to ./alignments.
        Some plans may require additional arguments, which are simply passed on
        during the dispatching."""
        if os.path.exists(alignmentFile):
            self.alignmentFile = alignmentFile
            self.alignmentExt = os.path.splitext(self.alignmentFile)[1]
            seqDict = SequenceUtilities.read_fasta_sequences(self.alignmentFile)
            # remove bad characters
            # check for the canonical sequence
            try:
                seqDict[self.canonSequence]
            except KeyError:
                raise CEPPipelineCanonException(self.canonSequence)
            else:
                self.partitionDirectory = os.path.join(self.mainDirectory,'alignments')
                # try to make partition directory if not already there
                try:
                    os.mkdir(self.partitionDirectory)
                except OSError:
                    pass
                # add number of sequences
                self.numSequences = len(seqDict)
                # check for a stockholm reference sequence
                try:
                    seqDict['#=GC RF']
                    stockholm = True
                except KeyError:
                    stockholm = False
                # try to resample (method should exist, but check again for safety)
                try:
                    getattr(self,self.resamplingMethod)(stockholm,seqDict,*kwargs)
                except AttributeError:
                    raise CEPPipelineResamplingMethodException(self.resamplingMethod)
        else:
            raise CEPPipelineAlignmentIOException(alignmentFile)


    @log_function_call('Split Half Resampling')
    def _resample_splithalf(self, stockholm, seqDict, **kwargs):
        """Split-half resampling.  Randomly partitions the master alignment into
        non-overlapping pairs of subalignments, nResamples number of times.
        """
        for i in xrange(1,self.numPartitions+1):
            halfOne = {}.fromkeys(random.sample(seqDict.keys(),int(len(seqDict)/2)))
            halfOne[self.canonSequence] = seqDict[self.canonSequence]
            halfTwo = {}
            halfTwo[self.canonSequence] = seqDict[self.canonSequence]
            [halfOne.__setitem__(x,seqDict[x]) for x in halfOne]
            [halfTwo.__setitem__(x,seqDict[x]) for x in seqDict if x not in halfOne]
            if stockholm == True:
                halfOne['#=GC RF'] = seqDict['#=GC RF']
                halfTwo['#=GC RF'] = seqDict['#=GC RF']
            fileNameOne = construct_file_name([self.fileIndicator,self.numSequences,i],'a',self.alignmentExt)
            seqFileOne = os.path.join(self.partitionDirectory,fileNameOne)
            fileNameTwo = construct_file_name([self.fileIndicator,self.numSequences,i],'b',self.alignmentExt)
            seqFileTwo = os.path.join(self.partitionDirectory,fileNameTwo)
            SequenceUtilities.write_fasta_sequences(halfOne,seqFileOne)
            SequenceUtilities.write_fasta_sequences(halfTwo,seqFileTwo)


    @log_function_call('Bootstrap Resampling')
    def _resample_bootstrap(self, stockholm, seqDict, **kwargs):
        """Bootstrap resampling.  Takes the master alignment of N sequences and produces
        resampled alignments that also contain N sequences (plus the canonical for numbering),
        but in which sequences can appear more than once.
        """
        n = len(seqDict)
        for iBoot in xrange(1,self.numPartitions+1):
            # list of sequence names to pick
            sk = sample_with_replacement(seqDict.keys(),n)
            # unique-ized list
            usk = [sk[i-1]+'_'+str(i-1) for i in xrange(1,len(sk)+1)]
            # make the dictionary
            bootSamp = {}
            for iSeq in xrange(0,len(sk)):
                bootSamp[usk[iSeq]] = seqDict[sk[iSeq]]
            # append canonical (need unmodified key!)
            bootSamp[self.canonSequence] = seqDict[self.canonSequence]
            if stockholm == True:
                bootSamp['#=GC RF'] = seqDict['#=GC RF']
            bootFileName = construct_file_name([self.fileIndicator,self.numSequences,iBoot],'boot',self.alignmentExt)
            bootFile = os.path.join(self.partitionDirectory,bootFileName)
            SequenceUtilities.write_fasta_sequences(bootSamp,bootFile)


    @log_function_call('Calculating Networks from Partitions')
    def calculate_networks(self, methodList, subset='*', gap=0.1, pcType='fix', pcMix=0.5, pcLambda=1.0, swtMethod='unweighted',cutoff=0.68):
        """
        Calculates networks using algorithms from CEPAlgorithms.  The methods that are implemented are:

        Mutual-information based:
            'mi'     : basic mutual information
            'nmi'    : mutual information normalized by joint entropy
            'znmi'   : z-scored normalized mutual information
            'zres'   : Chen and Little's zres
            'rpmi'   : combination of zres and znmi
            'mip'    : product-corrected mutual information
            'mic'    : yet another corrected mutual information

        Covariance-matrix based:
            'psicov' : sparse inverse covariance matrix estimation
            'ridge'  : L2-regularized inverse covariance matrix estimation
            'nmf'    : same as nmfdi, but does not compute direct information

        Other:
            'sca'    : Ranganathan's SCA (newer version, ca. 2009)
            'fchisq' : chi-squared computed from frequencies instead of counts

        Inverse Potts Model:
            'ipdi'   : independent pair approximation, with DI to convert to a single coupling
            'nmfdi'  : naive mean field (inverse covariance matrix), again with DI
            'tapdi'  : Thouless-Anderson-Palmer with DI
            'smdi'   : Seesak-Monasson with DI

        All methods use pair frequencies computed from resampled (or a single) alignments.  The methodList variable
        can therefore be iterable, like a list or tuple.

        An optional subset variable can be used to do only 'a' splits or 'b' splits.  The default is to do both
        at the same time ('a' first, then 'b').

        The default maximum gap percent allowed per column is 0.1.

        NOTE: Older algorithms featured in Brown and Brown 2010 ('oldsca', 'omes', 'elsc','mcbasc', and 'random')
        have been deprecated and removed.
        """
        self.networkExt = '.nwk'
        methodList = [m.lower() for m in methodList]
        methods = {'mi':'mutualInformation', 'nmi':'NMI', 'znmi':'ZNMI', 'mip':'MIp', 'zres':'Zres', 'sca':'SCA',
             'rpmi':'RPMI', 'fchisq':'FCHISQ','mic':'MIc', 'psicov':'PSICOV','nmfdi':'nmfDI','nmf':'NMF',
             'ridge':'RIDGE','tapdi':'tapDI','smdi':'smDI','ipdi':'ipDI'}
        # check that subset is OK
        if subset in ('a','b','*'):
            fileNames = construct_file_name([self.fileIndicator,self.numSequences,'*'],subset,self.alignmentExt)
            alignmentFiles = sorted(glob.glob(os.path.join(self.partitionDirectory,fileNames)))
            # make network directory
            self.networkDirectory = os.path.join(self.mainDirectory,'networks')
            try:
                os.mkdir(self.networkDirectory)
            except OSError:
                pass
            # loop over alignments
            for alnFile in alignmentFiles:
                parts = os.path.splitext(os.path.split(alnFile)[1])[0].split('_')
                print "----------building networks for '%s'----------"%os.path.split(alnFile)[1]
                msa = CEPAlgorithms.MSAAlgorithms(alnFile,gapFreqCutoff=gap,pcType=pcType,pcMix=pcMix,pcLambda=pcLambda,swtMethod=swtMethod,cutoff=cutoff)
                for method in methodList:
                    nwkFile = os.path.join(self.networkDirectory,construct_file_name([parts[0],parts[1],method,parts[2]],'',self.networkExt))
                    if os.path.exists(nwkFile):
                        print "network for '%s' already exists, skipping . . ."%method
                    else:
                        # map method variable to unbound methods and then call method
                        methodCall = {'mi':msa.calculate_mutual_information, 'nmi':msa.calculate_NMI, 'znmi':msa.calculate_ZNMI, 'mip':msa.calculate_MIp, \
                            'zres':msa.calculate_Zres, 'sca':msa.calculate_SCA, 'rpmi':msa.calculate_RPMI, 'fchisq':msa.calculate_FCHISQ,\
                            'mic':msa.calculate_MIc, 'psicov':msa.calculate_PSICOV,'nmfdi':msa.calculate_nmfDI, 'nmf':msa.calculate_NMF, \
                            'ridge':msa.calculate_RIDGE,'tapdi':msa.calculate_tapDI,'smdi':msa.calculate_smDI,'ipdi':msa.calculate_ipDI}
                        methodCall[method]()
                        # check for p-values for writing purposes
                        pvalues = hasattr(msa,'pvalues')
                        mapped_data = msa.map_to_canonical(msa.__dict__[methods[method]],self.canonSequence)
                        if pvalues:
                            significance = msa.map_to_canonical(msa.pvalues,self.canonSequence)
                        else:
                            # just put dummy p-values of 1.0 in there
                            significance = {}.fromkeys(mapped_data)
                            for k in significance:
                                significance[k] = 1.0
                        output = open(nwkFile,'w')
                        for ci,cj in mapped_data:
                            output.write("%d\t%d\t%.8f\t%.8f\n"%(ci,cj,mapped_data[(ci,cj)],significance[(ci,cj)]))
        else:
            raise CEPPipelineSubsetException(subset)


    @log_function_call('Initializing Graphs')
    def initialize_graphs(self):
        self.graphs = {}.fromkeys(xrange(1,self.numPartitions+1))
        if self.resamplingMethod == '_resample_splithalf':
            for k in self.graphs:
                self.graphs[k] = {'a':None,'b':None}
        if self.resamplingMethod == '_resample_bootstrap':
            for k in self.graphs:
                self.graphs[k] = {'boot':None}


    @log_function_call('Reading Graphs from Network Files')
    def read_graphs(self, method):
        '''
        Function that reads in network files and creates graphs for a single method.
        Full networks are read in; when graph pruning (to remove poor scores) is
        required, it is done within the accuracy and reproducibility calculations.
        '''
        #pruning = pruning.lower()
        #self.pruning = pruning
        self.method = method
        #pruningMethods = ('mst','topn','bottomn','pvalue')
        if hasattr(self,'networkDirectory'):
            self.initialize_graphs()
            file_names = construct_file_name([self.fileIndicator,self.numSequences,self.method],'*',self.networkExt)
            nwk_files = sorted(glob.glob(os.path.join(self.networkDirectory,file_names)))
            for f in nwk_files:
                # determine split and subset to store the graph
                i,j = determine_split_subset(deconstruct_file_name(f)[-1])
                self.graphs[i][j] = CEPNetworks.CEPGraph(f)
        else:
            raise CEPPipelineNetworkException



    @log_function_call('Calculating Resampling Statistics')
    def calculate_resampling_statistics(self,accMethod='distance',repMethod='splithalf',distMethod='oneminus',simMethod='spearman',rescaled=True,pruning=None,number=None,pdbFile=None, offset=0):
        '''
        Calculates the suite of resampling statistics (accuracy, reproducibility, etc.) from the
        graphs produced from the raw scoring networks.

        For the full details on available accuracy and graph similarity (reproducibility) methods,
        see CEPAccuracyCalculator and CEPGraph Similarity.

        Resampling scheme dictates reproducibility method.
            'splithalf' -> 'splithalf' reproducibility
            'bootstrap' -> 'avgcorr' reproducibility

        Selected accuracy methods (many more are possible; see CEPAccuracyCalculator):
            Direct use of scores:
            -avgdist : scaled, edge-weighted physical distance (used in Brown, Brown 2010 paper)
            -contact : inferior to performance string methods, but included for historical reasons

            Performance strings:
            -hamming  : hamming distance between ideal and performance strings
            -whamming : weighted hamming distance
            -rand     : rand index
            -bacc     : balanced accuracy
            -jaccard  : jaccard index

        Selected reproducibility methods:
        '''
        # THIS DOES NOT NEED TO BE DONE - REP. METHOD IS DETERMINED BY RESAMPLING METHOD
        # check for consistent options
        if self.resamplingMethod == '_resample_splithalf':
            if repMethod == 'bicar':
                raise CEPPipelineOptionsConflict(self.resamplingMethod,repMethod)
        elif repMethod == 'splithalf':
            raise CEPPipelineOptionsConflict(self.resamplingMethod,repMethod)
        # check that the graphs exist
        if not hasattr(self,'graphs'):
            raise CEPPipelineStatisticsException
        else:
            # graphs exist; deal with the case in which no PDB file is available
            self.acc_calc = None
            self.sim_calc = CEPGraphSimilarity.CEPGraphSimilarity()
            if pdbFile is not None:
                # try to find/read the supplied file
                try:
                    self.acc_calc = CEPAccuracyCalculator.CEPAccuracyCalculator(pdbFile)
                except:
                    raise CEPPipelineStructureIOException(pdbFile)
            # carry on with the statistics
            self.statistics = {'reproducibility':{}, 'accuracy':{}}
            self.consensusGraph = CEPNetworks.CEPGraph()
            # accuracy calculation
            self.calculate_accuracy(accMethod,pruning,number)
            # reproducibility calculation
            self.calculate_reproducibility(simMethod,pruning,number)
            # function distpatch and reproducibility calc.
            #repFunc = '_calculate_rep_'+repMethod
            #getattr(self,repFunc)(simMethod)


    def _prune_graph(self,partition,p,pruning,number):
        '''
        Prunes a graph according to the input pruning method and number.  Pruning
        modifies the graph in place, so it only needs to be called once.
        '''
        gmethods = {'mst':self.graphs[partition][p].calculate_mst, 'topn':self.graphs[partition][p].calculate_top_n,
            'bottomn':self.graphs[partition][p].calculate_bottom_n,'pvalue':self.graphs[partition][p].calculate_pvalue}
        gmethods[pruning](number)


    @log_function_call('Calculating Accuracy')
    def calculate_accuracy(self,acc_method,pruning,number):
        '''
        Loops over all the graphs (all graphs, all splits) and computes accuracy
        with the help of the accuracy calculator.  If no accuracy calculator
        exists (because of no struture existing), zero accuracy is assigned
        to every graph.
        '''
        nR = len(self.graphs)
        nS = len(self.graphs.values()[0])
        for partition in self.graphs:
            for p in self.graphs[partition]:
                acc_avg = 0.0
                if self.acc_calc is None:
                    self.statistics['accuracy'][partition] = 0.0
                else:
                    # arrange edges/weights into a dictionary
                    scores = {}
                    # need to prune the graph if we want to use 'avgdist' or 'contact'
                    if acc_method in ('avgdist','contact'):
                            self._prune_graph(partition,p,pruning,number)
                    # assemble the scores
                    for e in self.graphs[partition][p].edges():
                        scores[e] = self.graphs[partition][p].get_edge_data(e[0],e[1])['weight']
                    # pass the scores to the accuracy calculator
                    acc_avg += self.acc_calc.calculate(scores,acc_method)
                # divide by number of graphs in the partition
                self.statistics['accuracy'][partition] = acc_avg/len(self.graphs[partition])



    @log_function_call('Calculating Reproducibility')
    def calculate_reproducibility(self,sim_method,pruning,number):
        # XXX print(self.resamplingMethod)
        if self.resamplingMethod == '_resample_splithalf':
            self._calculate_rep_splithalf(sim_method,pruning,number)
        elif self.resamplingMethod == '_resample_bootstrap':
            self._calculate_rep_bootstrap(sim_method,pruning,number)
        else:
            print('ERROR! Cannot calculate reproducibility.')



    # REPRODUCIBILITY CALCS START HERE
    @log_function_call('Calculating Splithalf Reproducibility')
    def _calculate_rep_splithalf(self,simMethod,pruning,number):
        '''
        Reproduciblity is the split-to-split similarity of the two graphs; each
        split has a reproducibility value.
        '''
        #simFunc = '_graph_similarity_'+simMethod
        norm = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
        for partition in self.graphs:
            self._prune_graph(partition,'a',pruning,number)
            self._prune_graph(partition,'b',pruning,number)
            # sub-split similarity
            self.statistics['reproducibility'][partition] = self.sim_calc.calculate(self.graphs[partition]['a'],self.graphs[partition]['b'],'weight',simMethod)
            #self.statistics['reproducibility'][partition] = getattr(self,simFunc)(self.graphs[partition]['a'],self.graphs[partition]['b'])
            # consensus graph
            for i,j in self.graphs[partition]['a'].edges()+self.graphs[partition]['b'].edges():
                if self.consensusGraph.has_edge(i,j):
                    self.consensusGraph[i][j]['weight'] += 1/norm
                else:
                    self.consensusGraph.add_edge(i,j,weight=1/norm)


    @log_function_call('Calculating BICAR Reproducibility')
    def _calculate_rep_bootstrap(self,simMethod,pruning,number):
        '''
        Reproducibility is the average similarity among all unique pairs of
        resampled graphs; there is only a single reproducibility value.
        '''
        simFunc = '_graph_similarity_'+simMethod
        nG = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
        norm = (nG*(nG-1.0))/2.0
        avgSim = 0.0
        for p1 in self.graphs:
            for p2 in self.graphs:
                if int(p2) > int(p1):
                    self._prune_graph(p1,'boot',pruning,number)
                    self._prune_graph(p2,'boot',pruning,number)
                    #avgSim += getattr(self,simFunc)(self.graphs[p1]['boot'],g2 = self.graphs[p2]['boot'])
                    avgSim += self.sim_calc.calculate(self.graphs[partition]['a'],self.graphs[partition]['b'],'weight',simMethod)
            # consensus graph
            for i,j in self.graphs[p1]['boot'].edges():
                if self.consensusGraph.has_edge(i,j):
                    self.consensusGraph[i][j]['weight'] += 1/nG
                else:
                    self.consensusGraph.add_edge(i,j,weight=1/nG)
        # normalize
        self.statistics['reproducibility'][0] = avgSim/norm


    # implemented in CEPGraphSimilarity
    def _graph_similarity_jaccard(self,gOne,gTwo):
        """Calculates the Jaccard similarity (Jaccard index) for two graphs.  Ignores the
        weights in the graphs; simply uses presence and absence of edges."""
        return gOne.compute_jaccard_index(gTwo)

    # implemented in CEPGraphSimilarity
    def _graph_similarity_spearman(self,gOne,gTwo):
        """Calculates the correlational similarity between two graphs, using spearman correlation.
        Similarity is defined as for two half-splits in Brown and Brown, 2010."""
        # edge overlap calculations
        edgeUnion = frozenset(gOne.edges()).union(frozenset(gTwo.edges()))
        edgeInt = frozenset(gOne.edges()).intersection(frozenset(gTwo.edges()))
        edgeUniOne = edgeUnion.difference(frozenset(gTwo.edges()))
        edgeUniTwo = edgeUnion.difference(frozenset(gOne.edges()))
        # common edges
        eListOne = [gOne.get_edge_data(x[0],x[1])['weight'] for x in edgeInt]
        eListTwo = [gTwo.get_edge_data(x[0],x[1])['weight'] for x in edgeInt]
        # edges in A but not in B
        eListOne += [gOne.get_edge_data(x[0],x[1])['weight'] for x in edgeUniOne]
        eListTwo += [0.0 for x in edgeUniOne]
        # edges in B but not in A
        eListOne += [0.0 for x in edgeUniTwo]
        eListTwo += [gTwo.get_edge_data(x[0],x[1])['weight'] for x in edgeUniTwo]
        # return spearman correlation
        return spearmanr(eListOne,eListTwo)[0]

    # implemented in CEPGraphSimilarity
    def _graph_similarity_pearson(self,gOne,gTwo):
        """Calculates the correlations similarity between two graphs, using pearson correlation.
        Similarity is defined as for two half-split in Brown and Brown, 2010."""
        # edge overlap calculations
        edgeUnion = frozenset(gOne.edges()).union(frozenset(gTwo.edges()))
        edgeInt = frozenset(gOne.edges()).intersection(frozenset(gTwo.edges()))
        edgeUniOne = edgeUnion.difference(frozenset(gTwo.edges()))
        edgeUniTwo = edgeUnion.difference(frozenset(gOne.edges()))
        # common edges
        eListOne = [gOne.get_edge_data(x[0],x[1])['weight'] for x in edgeInt]
        eListTwo = [gTwo.get_edge_data(x[0],x[1])['weight'] for x in edgeInt]
        # edges in A but not in B
        eListOne += [gOne.get_edge_data(x[0],x[1])['weight'] for x in edgeUniOne]
        eListTwo += [0.0 for x in edgeUniOne]
        # edges in B but not in A
        eListOne += [0.0 for x in edgeUniTwo]
        eListTwo += [gTwo.get_edge_data(x[0],x[1])['weight'] for x in edgeUniTwo]
        # return pearson correlation
        return pearsonr(eListOne,eListTwo)[0]

    # implemented in CEPGraphSimilarity
    def _graph_similarity_frobenius(self,gOne,gTwo):
        """Calculates the similarity between two weighted graphs as the Frobenius norm
        (the sum of the squares of the singular values) of the difference in the (weighted)
        adjacency matrices."""
        deltaA = self._adj_matrix_diff(gOne,gTwo)
        normA = np.sqrt((deltaA**2).sum())
        #return np.exp(-1.0*normA)
        return 1.0/(1.0 + normA)

    # implemented in CEPGraphSimilarity
    def _graph_similarity_spectral(self,gOne,gTwo):
        """Calculates the similarity between two weighted graphs as the spectral norm (largest
        singular value) of the difference in the weighted adjacency matrices."""
        deltaA = self._adj_matrix_diff(gOne,gTwo)
        s = svd(deltaA,compute_uv=False)
        normA = s[0]
        #return np.exp(-1.0*normA)
        return 1.0/(1.0 + normA)

    # implemented in CEPGraphSimilarity
    def _graph_similarity_nuclear(self,gOne,gTwo):
        """Calculates the similarity between the two weighted graphs as the nuclear norm (sum of
        the singular values) of the differenc in the weighted adjacency matrices."""
        deltaA = self._adj_matrix_diff(gOne,gTwo)
        s = svd(deltaA,compute_uv=False)
        normA = s.sum()
        #return np.exp(-1.0*s.sum())
        return 1.0/(1.0 + normA)

    # implemented in CEPGraphSimilarity
    def _adj_matrix_diff(self,gOne,gTwo):
        """Calculates the difference in weighted adjacency matrices from two input graphs; useful
        for any similarity measure based norm.  All weighted adjacency matrices are have their
        column sums fixed at unity."""
        # edges in one graph and not in another get a zero in the adjacency matrix
        nodeUnion = frozenset(gOne.nodes()).union(frozenset(gTwo.nodes()))
        wOne = np.zeros((len(nodeUnion),len(nodeUnion)))
        wTwo = np.zeros((len(nodeUnion),len(nodeUnion)))
        deltaA = np.zeros((len(nodeUnion),len(nodeUnion)))
        matrixlocs = {}.fromkeys(nodeUnion)
        # map position keys to matrix elements
        cnt = 0
        for k in matrixlocs:
            matrixlocs[k] = cnt
            cnt = cnt + 1
        # make weight matrices by looking at edge union
        edgeUnion = frozenset(gOne.edges()).union(frozenset(gTwo.edges()))
        for e1,e2 in edgeUnion:
            n1,n2 = matrixlocs[e1],matrixlocs[e2]
            if gOne.has_edge(e1,e2):
                wOne[n1,n2] = gOne.get_edge_data(e1,e2)['weight']
            if gTwo.has_edge(e1,e2):
                wTwo[n1,n2] = gTwo.get_edge_data(e1,e2)['weight']
        # correct for scaling; have to take care of zero-sum columns carefully
        nfacOne = np.asarray([max(x,1.0e-08) for x in wOne.sum(axis=0)])
        nfacTwo = np.asarray([max(x,1.0e-08) for x in wTwo.sum(axis=0)])
        return wOne/nfacOne - wTwo/nfacTwo



class CEPPipelineDirectoryIOException(IOError):
    @log_function_call('ERROR : Input Directory')
    def __init__(self):
        print "There is a problem with your input directory.  Check the path name."

class CEPPipelineAlignmentIOException(IOError):
    @log_function_call('ERROR : Alignment File')
    def __init__(self,fileName):
        print "There is a problem with your input alignment file: %s.  Check the path and file name."%(fileName)

class CEPPipelineStructureIOException(IOError):
    @log_function_call('ERROR : Structure')
    def __init__(self,pdb):
        print "There is a problem loading your pdb structure: %s.  Check the path and file name."%(pdb)

class CEPPipelineDistanceException(Exception):
    @log_function_call('ERROR : Distances')
    def __init__(self):
        print "Residue-residue distances do not exist; compute them first before computing contacts."

class CEPPipelineCanonException(KeyError):
    @log_function_call('ERROR : Canonical Sequence')
    def __init__(self,canon):
        print "Your canonical sequence: '%s' cannot be found in the input file.  Please check the sequence name and file."%(canon)

class CEPPipelineDatabaseIOException(IOError):
    @log_function_call('ERROR : Database File')
    def __init__(self,name):
        print "There is a problem loading your database file: %s.  Check the path and file name."%(name)

class CEPPipelineNetworkException(Exception):
    @log_function_call('ERROR : Network Files')
    def __init__(self):
        print "You have attempted to calculate graphs without network files.  Please make your networks first, then calculate the graphs."

class CEPPipelinePruningException(Exception):
    @log_function_call('ERROR : Pruning Method')
    def __init__(self,pruning):
        print "Your choice of pruning method, '%s', is not supported.  Please check the available methods and change your selection."%(pruning)

class CEPPipelineSubsetException(Exception):
    @log_function_call('ERROR : Subset Selection')
    def __init__(self,subset):
        print "You must choose to process subset 'a', subset 'b', or both '*' (default).  You chose: '%s'.  Please check your subset selection."%(subset)

class CEPPipelineMethodException(Exception):
    @log_function_call('ERROR : Method Selection')
    def __init__(self,method):
        print "You must pick a supported method; '%s' is not supported.  Please check the method you have selected for making networks."%(method)

class CEPPipelineResamplingMethodException(AttributeError):
    @log_function_call('ERROR : Resampling Plan Selection')
    def __init__(self,method):
        print "Method %s for resampling not currently supported.  Please choose an allowed resampling plan."%(method)

class CEPPipelineOptionsConflict(Exception):
    @log_function_call('ERROR : Incompatible Options')
    def __init__(self,method1,method2):
        print "Option %s is not compatible with option %s. Please check your input options."%(method1,method2)

class CEPPipelineDistanceMethodException(Exception):
    @log_function_call('ERROR : Accuracy Method Error')
    def __init__(self,method):
        print "Method %s for accuracy transformation not recognized.  Please choose from an allowed transformation."%(method)

class CEPPipelineStatisticsException(Exception):
    @log_function_call('ERROR: Graphs Not Present')
    def __init__(self):
        print "Graphs need to be loaded before statistics can be calculated.  Run read_graphs() then try again."

class CEPPipelineTests(unittest.TestCase):
    def setUp(self):
        pass

if __name__ == '__main__':
    unittest.main()
