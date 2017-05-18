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

import sys, os, unittest, random, glob, cPickle, re, itertools, copy
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
    def __init__(self,options,database=None):
        """ Main constructor for a processing pipeline.

        All pipeline options/parameters are contained in a CEPParameters object.
        """
        self.options = options
        if database is not None:
            # start a project by reading in an archived database, along with the
            #   options archived there
            if os.path.exists(database):
                self.read_database(database)
            else:
                raise CEPPipelineDatabaseIOException(database)
        elif database == None and self.options.main_directory is not None:
            if not os.path.exists(self.options.main_directory):
                os.mkdir(self.options.main_directory)
            # establish basic instance variables for a new project
            self.database_directory = os.path.join(self.options.main_directory,'databases')
            self.network_directory = os.path.join(self.options.main_directory,'networks')
            # check for a valid resampling plan
            if self.options.resampling_method is not None:
                self.resampling_method = '_resample_'+self.options.resampling_method
            else:
                self.resampling_method = '_resample_null'
            if not hasattr(self,self.resampling_method):
                raise CEPPipelineResamplingMethodException(self.options.resampling_method)
            # initialize the statistics, even if not all are calculated
            self.statistics = {'reproducibility':{}, 'accuracy':{}}
        else:
            raise CEPPipelineDirectoryIOException


    def __str__(self):
        """String representation of the pipeline."""
        strRep = 'Correlated Substitution Analysis Pipeline\n'
        strRep += '   Results stored in '+self.options.main_directory+'\n'
        strRep += '   Canonical sequence : '+self.options.canon_sequence+'\n'
        strRep += '   Resampling Method: '+self.options.resampling_method+'\n'
        return strRep


    @log_function_call('Cleaning Pipeline Project')
    def clean_project(self):
        """Removes all of the files from the partition and network directories along with directory"""
        partition = os.path.join(self.options.main_directory,'alignments')
        network = os.path.join(self.options.main_directory,'networks')
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
    def read_database(self,db_file):
        '''
        Reads in a database file.  The database is guaranteed to contain:
         with a dictionary keyed by:
            dict['graph'] = <networkx graph object>
            dict['statistics'] = statistics
            dict['options'] = set of options that produced the database file
            dict['metadata'] = <instance variables>
        If a consensus graph was calculated, the database dictionary will also
        contain:
            dict['graph'] = <networkx graph object>
        '''
        # this should set all of the instance variables from the file name
        try:
            db_ptr = open(db_file,'rb')
        except IOError:
            raise CEPPipelineDatabaseIOException(db_file)
        else:
            dictionary = cPickle.load(db_ptr)
            self.statistics = dictionary['statistics']
            self.options = dictionary['options']
            if dictionary.has_key('graph'):
                self.consensus_graph = dictionary['graph']
            for attribute in dictionary['metadata']:
                self.__dict__[attribute] = dictionary['metadata'][attribute]
            db_ptr.close()


    @log_function_call('Writing Pipeline Database')
    def write_database(self):
        '''
        Writes a database file with a dictionary keyed by:
            dict['statistics'] = statistics
            dict['options'] = set of options that produced the database file
            dict['metadata'] = <instance variables>
        If the consensus graph has been calculated, the database dictionary will
        also have:
            dict['graph'] = <networkx graph object>
        '''
        # write to the default database directory ./database
        if os.path.exists(self.database_directory):
            pass
        else:
            os.mkdir(self.database_directory)
        db_file = construct_file_name([self.options.file_indicator,self.num_sequences,self.method], '', '.pydb')
        db_file = os.path.join(self.database_directory,db_file)
        db_ptr = open(db_file,'wb')
        dictionary = {'statistics':self.statistics, 'options':self.options, 'metadata':{}}
        if hasattr(self,'consensus_graph'):
            dictionary['graph'] = self.consensus_graph
        for attribute in self.__dict__:
            dictionary['metadata'][attribute] = self.__dict__[attribute]
        cPickle.dump(dictionary,db_ptr,-1)
        db_ptr.close()



    @log_function_call('Resampling Subalignments from Master Alignment')
    def resample_alignment(self,**kwargs):
        """Makes splits of an input alignment file and write to ./alignments.
        Some plans may require additional arguments, which are simply passed on
        during the dispatching."""
        if os.path.exists(self.options.alignment_file):
            #self.alignmentFile = alignmentFile
            self.alignment_ext = os.path.splitext(self.options.alignment_file)[1]
            seq_dict = SequenceUtilities.read_fasta_sequences(self.options.alignment_file)
            # remove bad characters
            # check for the canonical sequence
            try:
                seq_dict[self.options.canon_sequence]
            except KeyError:
                raise CEPPipelineCanonException(self.options.canon_sequence)
            else:
                self.partition_directory = os.path.join(self.options.main_directory,'alignments')
                # try to make partition directory if not already there
                try:
                    os.mkdir(self.partition_directory)
                except OSError:
                    pass
                # add number of sequences
                self.num_sequences = len(seq_dict)
                # check for a stockholm reference sequence
                try:
                    seq_dict['#=GC RF']
                    stockholm = True
                except KeyError:
                    stockholm = False
                # try to resample (method should exist, but check again for safety)
                try:
                    getattr(self,self.resampling_method)(stockholm,seq_dict,*kwargs)
                except AttributeError:
                    raise CEPPipelineResamplingMethodException(self.resampling_method)
        else:
            raise CEPPipelineAlignmentIOException(self.options.alignment_file)


    @log_function_call('No Resampling')
    def _resample_null(self, stockholm, seq_dict, **kwargs):
        '''
        No resampling; simply writes the full alignment to the alignments directory.
        Used to allow single-shot (full alignment) calculations.  If this is the
        plan chosen, you CANNOT calculate either reproducibility or a consensus
        graph.
        '''
        fname = construct_file_name([self.options.file_indicator,self.num_sequences,1],'full',self.alignment_ext)
        seqfile = os.path.join(self.partition_directory,fname)
        full_seq = copy.deepcopy(seq_dict)
        if stockholm == True:
            full_seq['#=GC RF'] = seq_dict['#=GC RF']
        SequenceUtilities.write_fasta_sequences(full_seq,seqfile)


    @log_function_call('Split Half Resampling')
    def _resample_splithalf(self, stockholm, seq_dict, **kwargs):
        """Split-half resampling.  Randomly partitions the master alignment into
        non-overlapping pairs of subalignments, nResamples number of times.
        """
        for i in xrange(1,self.options.num_partitions+1):
            half_one = {}.fromkeys(random.sample(seq_dict.keys(),int(len(seq_dict)/2)))
            half_one[self.options.canon_sequence] = seq_dict[self.options.canon_sequence]
            half_two = {}
            half_two[self.options.canon_sequence] = seq_dict[self.options.canon_sequence]
            [half_one.__setitem__(x,seq_dict[x]) for x in half_one]
            [half_two.__setitem__(x,seq_dict[x]) for x in seq_dict if x not in half_one]
            if stockholm == True:
                half_one['#=GC RF'] = seq_dict['#=GC RF']
                half_two['#=GC RF'] = seq_dict['#=GC RF']
            fname_one = construct_file_name([self.options.file_indicator,self.num_sequences,i],'a',self.alignment_ext)
            seqfile_one = os.path.join(self.partition_directory,fname_one)
            fname_two = construct_file_name([self.options.file_indicator,self.num_sequences,i],'b',self.alignment_ext)
            seqfile_two = os.path.join(self.partition_directory,fname_two)
            SequenceUtilities.write_fasta_sequences(half_one,seqfile_one)
            SequenceUtilities.write_fasta_sequences(half_two,seqfile_two)


    @log_function_call('Bootstrap Resampling')
    def _resample_bootstrap(self, stockholm, seq_dict, **kwargs):
        """Bootstrap resampling.  Takes the master alignment of N sequences and produces
        resampled alignments that also contain N sequences (plus the canonical for numbering),
        but in which sequences can appear more than once.
        """
        n = len(seqDict)
        for iboot in xrange(1,self.options.num_partitions+1):
            # list of sequence names to pick
            sk = sample_with_replacement(seq_dict.keys(),n)
            # unique-ized list
            usk = [sk[i-1]+'_'+str(i-1) for i in xrange(1,len(sk)+1)]
            # make the dictionary
            boot_samp = {}
            for iseq in xrange(0,len(sk)):
                boot_samp[usk[iseq]] = seq_dict[sk[iseq]]
            # append canonical (need unmodified key!)
            boot_samp[self.options.canon_sequence] = seq_dict[self.options.canon_sequence]
            if stockholm == True:
                boot_samp['#=GC RF'] = seq_dict['#=GC RF']
            boot_filename = construct_file_name([self.options.file_indicator,self.num_sequences,iboot],'boot',self.alignment_ext)
            boot_file = os.path.join(self.partition_directory,boot_filename)
            SequenceUtilities.write_fasta_sequences(boot_samp,boot_file)


    @log_function_call('Calculating Networks from Partitions')
    def calculate_networks(self,method_list,subset='*'):
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
        self.network_ext = '.nwk'
        method_list = [m.lower() for m in method_list]
        methods = {'mi':'mutualInformation', 'nmi':'NMI', 'znmi':'ZNMI', 'mip':'MIp', 'zres':'Zres', 'sca':'SCA',
             'rpmi':'RPMI', 'fchisq':'FCHISQ','mic':'MIc', 'psicov':'PSICOV','nmfdi':'nmfDI','nmf':'NMF',
             'ridge':'RIDGE','tapdi':'tapDI','smdi':'smDI','ipdi':'ipDI'}
        # check that subset is OK
        if subset in ('a','b','*'):
            file_names = construct_file_name([self.options.file_indicator,self.num_sequences,'*'],subset,self.alignment_ext)
            alignment_files = sorted(glob.glob(os.path.join(self.partition_directory,file_names)))
            # make network directory
            self.network_directory = os.path.join(self.options.main_directory,'networks')
            try:
                os.mkdir(self.network_directory)
            except OSError:
                pass
            # loop over alignments
            for aln_file in alignment_files:
                parts = os.path.splitext(os.path.split(aln_file)[1])[0].split('_')
                print "----------building networks for '%s'----------"%os.path.split(aln_file)[1]
                msa = CEPAlgorithms.MSAAlgorithms(aln_file,gapFreqCutoff=self.options.gap,pcType=self.options.pc_type,pcMix=self.options.pc_mix,
                    pcLambda=self.options.pc_lambda,swtMethod=self.options.swt_method,cutoff=self.options.cutoff)
                for method in method_list:
                    nwk_file = os.path.join(self.network_directory,construct_file_name([parts[0],parts[1],method,parts[2]],'',self.network_ext))
                    if os.path.exists(nwk_file):
                        print "network for '%s' already exists, skipping . . ."%method
                    else:
                        # map method variable to unbound methods and then call method
                        method_call = {'mi':msa.calculate_mutual_information, 'nmi':msa.calculate_NMI, 'znmi':msa.calculate_ZNMI, 'mip':msa.calculate_MIp, \
                            'zres':msa.calculate_Zres, 'sca':msa.calculate_SCA, 'rpmi':msa.calculate_RPMI, 'fchisq':msa.calculate_FCHISQ,\
                            'mic':msa.calculate_MIc, 'psicov':msa.calculate_PSICOV,'nmfdi':msa.calculate_nmfDI, 'nmf':msa.calculate_NMF, \
                            'ridge':msa.calculate_RIDGE,'tapdi':msa.calculate_tapDI,'smdi':msa.calculate_smDI,'ipdi':msa.calculate_ipDI}
                        method_call[method]()
                        # check for p-values for writing purposes
                        pvalues = hasattr(msa,'pvalues')
                        mapped_data = msa.map_to_canonical(msa.__dict__[methods[method]],self.options.canon_sequence)
                        if pvalues:
                            significance = msa.map_to_canonical(msa.pvalues,self.options.canon_sequence)
                        else:
                            # just put dummy p-values of 1.0 in there
                            significance = {}.fromkeys(mapped_data)
                            for k in significance:
                                significance[k] = 1.0
                        output = open(nwk_file,'w')
                        for ci,cj in mapped_data:
                            output.write("%d\t%d\t%.8f\t%.8f\n"%(ci,cj,mapped_data[(ci,cj)],significance[(ci,cj)]))
        else:
            raise CEPPipelineSubsetException(subset)


    @log_function_call('Initializing Graphs')
    def initialize_graphs(self):
        self.graphs = {}.fromkeys(xrange(1,self.options.num_partitions+1))
        if self.resampling_method == '_resample_splithalf':
            for k in self.graphs:
                self.graphs[k] = {'a':None,'b':None}
        if self.resampling_method == '_resample_bootstrap':
            for k in self.graphs:
                self.graphs[k] = {'boot':None}
        if self.resampling_method == '_resample_null':
            self.graphs = {1:{'full':None}}


    @log_function_call('Reading Graphs from Network Files')
    def read_graphs(self, method):
        '''
        Function that reads in network files and creates graphs for a single method.
        Full networks are read in; when graph pruning (to remove poor scores) is
        required, it is done within the accuracy and reproducibility calculations.
        '''
        self.method = method
        if hasattr(self,'network_directory'):
            self.initialize_graphs()
            file_names = construct_file_name([self.options.file_indicator,self.num_sequences,self.method],'*',self.network_ext)
            nwk_files = sorted(glob.glob(os.path.join(self.network_directory,file_names)))
            for f in nwk_files:
                # determine split and subset to store the graph
                i,j = determine_split_subset(deconstruct_file_name(f)[-1])
                self.graphs[i][j] = CEPNetworks.CEPGraph(f)
        else:
            raise CEPPipelineNetworkException


    # DEPRECATED: CAN REMOVE
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
            self.calculate_accuracy(accMethod,pruning,number,pdbFile)
            # reproducibility calculation
            self.calculate_reproducibility(simMethod,pruning,number)
            # function distpatch and reproducibility calc.
            #repFunc = '_calculate_rep_'+repMethod
            #getattr(self,repFunc)(simMethod)


    def _prune_graph(self,partition,p):
        '''
        Prunes a graph according to the input pruning method and number.  Pruning
        modifies the graph in place, so it only needs to be called once.
        '''
        gmethods = {'mst':self.graphs[partition][p].calculate_mst, 'topn':self.graphs[partition][p].calculate_top_n,
            'bottomn':self.graphs[partition][p].calculate_bottom_n,'pvalue':self.graphs[partition][p].calculate_pvalue}
        gmethods[self.options.pruning](self.options.number)


    @log_function_call('Calculating Accuracy')
    #def calculate_accuracy(self,acc_method,pruning,number,pdb_file):
    def calculate_accuracy(self):
        '''
        Loops over all the graphs (all graphs, all splits) and computes accuracy
        with the help of the accuracy calculator.  If no accuracy calculator
        exists (because of no struture existing), zero accuracy is assigned
        to every graph.
        '''
        try:
            self.acc_calc = CEPAccuracyCalculator.CEPAccuracyCalculator(self.options.pdb_file)
        except:
            raise CEPPipelineStructureIOException(self.options.pdb_file)
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
                    if self.options.acc_method in ('avgdist','contact'):
                            self._prune_graph(partition,p)
                    # assemble the scores
                    for e in self.graphs[partition][p].edges():
                        scores[e] = self.graphs[partition][p].get_edge_data(e[0],e[1])['weight']
                    # pass the scores to the accuracy calculator
                    acc_avg += self.acc_calc.calculate(scores,self.options.acc_method)
                # divide by number of graphs in the partition
                self.statistics['accuracy'][partition] = acc_avg/len(self.graphs[partition])



    @log_function_call('Calculating Reproducibility')
    def calculate_reproducibility(self):
        self.sim_calc = CEPGraphSimilarity.CEPGraphSimilarity()
        if self.resampling_method == '_resample_splithalf':
            self._calculate_rep_splithalf()
        elif self.resampling_method == '_resample_bootstrap':
            self._calculate_rep_bootstrap()
        else:
            print('ERROR! Cannot calculate reproducibility.')


    @log_function_call('Calculating Consensus Graph')
    def calculate_consensus_graph(self):
        '''
        The consensus graph's edges are the number of resamples in which that edge
        occured in the pruned set of edges.
        '''
        self.consensus_graph = CEPNetworks.CEPGraph()
        if self.resampling_method == '_resample_splithalf':
            norm = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
            for partition in self.graphs:
                self._prune_graph(partition,'a')
                self._prune_graph(partition,'b')
                for i,j in self.graphs[partition]['a'].edges()+self.graphs[partition]['b'].edges():
                    if self.consensus_graph.has_edge(i,j):
                        self.consensus_graph[i][j]['weight'] += 1/norm
                    else:
                        self.consensus_graph.add_edge(i,j,weight=1/norm)
        elif self.resampling_method == '_resample_bootstrap':
            nG = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
            norm = (nG*(nG-1.0))/2.0
            for partition in self.graphs:
                self._prune_graph(partition,'boot')
                for i,j in self.graphs[partition]['boot'].edges():
                    if self.consensus_graph.has_edge(i,j):
                        self.consensus_graph[i][j]['weight'] += 1/nG
                    else:
                        self.consensus_graph.add_edge(i,j,weight=1/nG)
        else:
            print('ERROR! Cannot calculate consensus graph.')



    @log_function_call('Calculating Splithalf Reproducibility')
    def _calculate_rep_splithalf(self):
        '''
        Reproduciblity is the split-to-split similarity of the two graphs; each
        split has a reproducibility value.
        '''
        for partition in self.graphs:
            self._prune_graph(partition,'a')
            self._prune_graph(partition,'b')
            # sub-split similarity
            self.statistics['reproducibility'][partition] = self.sim_calc.calculate(self.graphs[partition]['a'],self.graphs[partition]['b'],'weight',self.options.sim_method)


    @log_function_call('Calculating BICAR Reproducibility')
    def _calculate_rep_bootstrap(self):
        '''
        Reproducibility is the average similarity among all unique pairs of
        resampled graphs; there is only a single reproducibility value.
        '''
        sim_func = '_graph_similarity_'+self.options.sim_method
        nG = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
        norm = (nG*(nG-1.0))/2.0
        avg_sim = 0.0
        for p1 in self.graphs:
            for p2 in self.graphs:
                if int(p2) > int(p1):
                    self._prune_graph(p1,'boot')
                    self._prune_graph(p2,'boot')
                    avg_sim += self.sim_calc.calculate(self.graphs[p1]['boot'],self.graphs[p2]['boot'],'weight',sim_method)
        # normalize
        self.statistics['reproducibility'][0] = avg_sim/norm



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
