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

import sys, os, unittest, random, glob, pickle, re, itertools, copy, inspect, glob
from scipy import mean
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
import numpy as np
import Bio.PDB as pdb
from networkx import Graph
from pycsa.CEPPreprocessing import SequenceUtilities
from pycsa.CEPLogging import LogPipeline
from pycsa import CEPAlgorithms,CEPNetworks,CEPAccuracyCalculator,CEPGraphSimilarity
from pyrankagg import rankagg


# decorator function to be used for logging purposes
log_function_call = LogPipeline.log_function_call

def get_fileset_nums(nwk_dir):
    '''
    Gets all the fileset numbers for the network files in nwk_dir; used when
    trying to produce voted results for ensembles of resamples.
    '''
    unique_ind = {}
    files = glob.glob(nwk_dir+'/*.nwk')
    for f in files:
        num_part = f.split('/')[-1].split('.')[0].split('_')[-2]
        num_part = int(num_part)
        if not unique_ind.has_key(num_part):
            unique_ind[num_part] = None
    return unique_ind.keys()

def construct_file_name(pieces, extension):
    """Simple function to create a file name from pieces, then add subset plus file extension"""
    return reduce(lambda x,y: str(x)+'_'+str(y),pieces) + extension

def deconstruct_file_name(name):
    """Simple function to break apart a file name into individual pieces without the extension"""
    return os.path.splitext(os.path.split(name)[1])[0].split('_')

def determine_split_subset(name):
    '''
    Simple function that breaks up a pipeline file and returns the split (integet) and the
    subset ('a','b','boot','full',etc.)
    '''
    pieces = deconstruct_file_name(name)
    return int(pieces[-2]),pieces[-1]


def parse_network_file(nwk_file):
    '''
    Reads a network file with lines of the form:
        ri(<int>) rj(<int>) score(<float>) pvalue(<float>)
    and returns a dictionary of (ri,rj):score.
    The bootstrap .nwk file names are of the form:
        protein_Nres_method_nboot.nwk
    from which we extract the method name.
    '''
    scores = {}
    with open(nwk_file,'r') as f:
        lines = f.readlines()
    f.close()
    for l in lines:
        clean_line = ' '.join(l.strip().split())
        atoms = clean_line.split()
        scores[(int(atoms[0]),int(atoms[1]))] = float(atoms[2])
    method = nwk_file.split('_')[2]
    return method,scores


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


    def __str__(self):
        '''
        Pretty-print string representation of the options
        '''
        str_rep = 'Correlated Substitutions Pipeline Options:\n'
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    str_rep += '\n\t'+i[0]+' : '+str(i[1])
        return str_rep


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
            dictionary = pickle.load(db_ptr)
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
        db_file = construct_file_name([self.options.file_indicator,self.num_sequences,self.method],'.pydb')
        db_file = os.path.join(self.database_directory,db_file)
        db_ptr = open(db_file,'wb')
        dictionary = {'statistics':self.statistics, 'options':self.options, 'metadata':{}}
        if hasattr(self,'consensus_graph'):
            dictionary['graph'] = self.consensus_graph
        for attribute in self.__dict__:
            dictionary['metadata'][attribute] = self.__dict__[attribute]
        pickle.dump(dictionary,db_ptr,-1)
        db_ptr.close()



    @log_function_call('Resampling Subalignments from Master Alignment')
    def resample_alignment(self,**kwargs):
        """Makes splits of an input alignment file and write to ./alignments.
        Some plans may require additional arguments, which are simply passed on
        during the dispatching."""
        if os.path.exists(self.options.alignment_file):
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
        fname = construct_file_name([self.options.file_indicator,self.num_sequences,1,'full'],self.alignment_ext)
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
            fname_one = construct_file_name([self.options.file_indicator,self.num_sequences,i,'a'],self.alignment_ext)
            seqfile_one = os.path.join(self.partition_directory,fname_one)
            fname_two = construct_file_name([self.options.file_indicator,self.num_sequences,i,'b'],self.alignment_ext)
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
            boot_filename = construct_file_name([self.options.file_indicator,self.num_sequences,iboot,'boot'],self.alignment_ext)
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
            file_names = construct_file_name([self.options.file_indicator,self.num_sequences,'*',subset],self.alignment_ext)
            alignment_files = sorted(glob.glob(os.path.join(self.partition_directory,file_names)))
            # make network directory
            self.network_directory = os.path.join(self.options.main_directory,'networks')
            try:
                os.mkdir(self.network_directory)
            except OSError:
                pass
            # loop over alignments
            for aln_file in alignment_files:
                #parts = os.path.splitext(os.path.split(aln_file)[1])[0].split('_')
                parts = deconstruct_file_name(aln_file)
                print('----------building networks for \'%s\'----------' % os.path.split(aln_file)[1])
                msa = CEPAlgorithms.MSAAlgorithms(aln_file,gapFreqCutoff=self.options.gap,pcType=self.options.pc_type,pcMix=self.options.pc_mix,
                    pcLambda=self.options.pc_lambda,swtMethod=self.options.swt_method,cutoff=self.options.cutoff)
                for method in method_list:
                    nwk_file = os.path.join(self.network_directory,construct_file_name([parts[0],parts[1],method,parts[2],parts[3]],self.network_ext))
                    if os.path.exists(nwk_file):
                        print('network for \'%s\' already exists, skipping . . .' %method)
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
            file_names = construct_file_name([self.options.file_indicator,self.num_sequences,self.method,'*'],self.network_ext)
            nwk_files = sorted(glob.glob(os.path.join(self.network_directory,file_names)))
            for f in nwk_files:
                # determine split and subset to store the graph
                i,j = determine_split_subset(f)
                self.graphs[i][j] = CEPNetworks.CEPGraph(f)
        else:
            raise CEPPipelineNetworkException


    def _prune_graph(self,partition,p):
        '''
        Prunes a graph according to the input pruning method and number.  Pruning
        modifies the graph in place, so it only needs to be called once.
        '''
        gmethods = {'mst':self.graphs[partition][p].calculate_mst, 'topn':self.graphs[partition][p].calculate_top_n,
            'bottomn':self.graphs[partition][p].calculate_bottom_n,'pvalue':self.graphs[partition][p].calculate_pvalue}
        gmethods[self.options.pruning](self.options.number)


    @log_function_call('Voting')
    def calculate_voted_network(self):
        '''
        Rank aggregates the results from different scoring methods.  If no PDB file is
        supplied, a consensus_graph can be calculated but no accuracies for either the
        individual methods or the voted model can be computed.  Results are written
        to a special database in the databases directory.

        The consensus graph(s) for voting are constructed differently from the non-voted
        case.  In this case, for each set of methods calculated on a particular alignment,
        the self.options.number of top ranked edges are added to the graph, for each method.
        The weight of the edge = number of methods which placed that pair in its set of
        top-ranking edges.
        '''
        self.statistics = {'accuracy':{'voted':[]}}
        # check to see if we can calculate accuracies
        no_acc = True
        try:
            self.acc_calc = CEPAccuracyCalculator.CEPAccuracyCalculator(self.options.pdb_file)
            no_acc = False
        except:
            no_acc = True
        # there may be many consensus graphs
        self.consensus_graph = []
        # create the rank aggregator
        flra = rankagg.FullListRankAggregator()
        # get all the existing file indices in the networks directory
        file_nums = get_fileset_nums(self.network_directory)
        # loop over network files
        for nwk_indx in file_nums:
            score_list = []
            file_wc = construct_file_name([self.options.file_indicator,self.num_sequences,'*',nwk_indx,'*'],self.network_ext)
            nwk_file_list = sorted(glob.glob(os.path.join(self.network_directory,file_wc)))
            for f in nwk_file_list:
                # read the scores into a dictionary and figure out the method name
                method,scores = parse_network_file(f)
                # add key to the acc dict if necessary
                if not self.statistics['accuracy'].has_key(method):
                    self.statistics['accuracy'][method] = []
                # save the scores for rank aggregation
                score_list.append(scores)
                # compute method accuracy, if the accuracy calculator exists
                #    (otherwise just put None there)
                if no_acc:
                    self.statistics['accuracy'][method].append(None)
                else:
                    self.statistics['accuracy'][method].append(self.acc_calc.calculate(scores,self.options.acc_method))
            # now do the voting
            agg_ranks = flra.aggregate_ranks(score_list,areScores=True,method='borda')
            # in order to make the performance vector for the voted ranks, we need the highest
            #   rank item to be the LARGEST score, so create a fake set of scores = 1/rank; we
            #   also use these scores for edge weights in the voted network
            voted_scores = {k:1.0/v for k,v in agg_ranks.iteritems()}
            if no_acc:
                self.statistics['accuracy']['voted'].append(None)
            else:
                self.statistics['accuracy']['voted'].append(self.acc_calc.calculate(voted_scores,self.options.acc_method))
            # form the consensus graph for this set of alignments
            con_graph = CEPNetworks.CEPGraph()
            for (i,j),s in voted_scores.iteritems():
                con_graph.add_edge(i,j,weight=s)
            self.consensus_graph.append(con_graph)
        # now dump the results
        self.method = 'voted'
        self.write_database()

    """
    # weight the votes by accuracy?
    weight = True
    # these will store the accuracy results
    acc_hamming = {'voted':[]}
    acc_topN = {'voted':[]}
    # value of N for topN
    N = 90
    # this will be the 'ideal' performance vector
    I = None
    # get list of contacts (only need to do 1X)
    pdbFile = 'pdz/pdb/1iu0.pdb'
    distances = cepstruc.calculate_distances(pdbFile)
    C = cepstruc.calculate_CASP_contacts(distances)
    # rank aggregator
    FLRA = rankagg.FullListRankAggregator()
    K = len(C)
    # find all the exiting file indices (this is in case of partial runs)
    fileNums = get_fileset_nums('pdz/networks')
    # BIG LOOP OVER NETWORK FILES
    for nwkIndx in fileNums:
        print 'Working on nwk files in sample',nwkIndx
        Slist = []
        accList = []
        nwkList = glob.glob('pdz/networks/pdz_800_*_'+str(nwkIndx)+'boot.nwk')
        for f in nwkList:
            # read scores into a dictionary and figure out method name
            method,S = parse_network_file(f)
            # set up ideal predictor
            if I is None:
                I = ''.join(['1' for k in xrange(0,K)]+['0' for k in xrange(0,len(S) - K)])
            # add keys to data dicts if necessary
            if not acc_hamming.has_key(method):
                acc_hamming[method] = []
            if not acc_topN.has_key(method):
                acc_topN[method] = []
            # save the scores for voting
            Slist.append(S)
            # produce the performance string, and save for later
            P = make_perfstring(S,C)
            # compute accuracies for the individual methods
            # Hamming
            acc_hamming[method].append(1 - 1.0*hamming_dist_binary(P,I)/(2*K))
            # topN
            acc_topN[method].append(1 - 1.0*hamming_dist_binary(P[:N],I[:N])/N)
            # THIS IS WHERE WE SAVE THE ACCURACIES FOR WEIGHTING
            accList.append(1 - 1.0*hamming_dist_binary(P,I)/(2*K))
        # voting
        if weight:
            aggranks = FLRA.aggregate_ranks(Slist,areScores=True,method='borda',weights=[exp(x) for x in accList])
        else:
            aggranks = FLRA.aggregate_ranks(Slist,areScores=True,method='borda')
        # KO is probably prohibitive for this problem
        # lkoranks = FLRA.locally_kemenize(aggranks,[FLRA.convert_to_ranks(s) for s in Slist])
        # in order to make the performance vector, we need the highest rank item to be
        #   the LARGEST score.  (So we send in 1/rank)
        PV = make_perfstring({k:1.0/v for k,v in aggranks.iteritems()},C)
        acc_hamming['voted'].append(1 - 1.0*hamming_dist_binary(PV,I)/(2*K))
        acc_topN['voted'].append(1 - 1.0*hamming_dist_binary(PV[:N],I[:N])/N)
        # reset I
        I = None
    # save scoring dictionaries
    cPickle.dump(acc_hamming,open('voted_PDZ_hamming_expwt.pydb','wb'),protocol=-1)
    cPickle.dump(acc_topN,open('voted_PDZ_topN_expwt.pydb','wb'),protocol=-1)
    #
    """

    @log_function_call('Calculating Accuracy')
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
        print('There is a problem with your input directory.  Check the path name.')

class CEPPipelineAlignmentIOException(IOError):
    @log_function_call('ERROR : Alignment File')
    def __init__(self,fileName):
        print('There is a problem with your input alignment file: %s.  Check the path and file name.' %(fileName))

class CEPPipelineStructureIOException(IOError):
    @log_function_call('ERROR : Structure')
    def __init__(self,pdb):
        print('There is a problem loading your pdb structure: %s.  Check the path and file name.' %(pdb))

class CEPPipelineDistanceException(Exception):
    @log_function_call('ERROR : Distances')
    def __init__(self):
        print('Residue-residue distances do not exist; compute them first before computing contacts.')

class CEPPipelineCanonException(KeyError):
    @log_function_call('ERROR : Canonical Sequence')
    def __init__(self,canon):
        print('Your canonical sequence: \'%s\' cannot be found in the input file.  Please check the sequence name and file.' %(canon))

class CEPPipelineDatabaseIOException(IOError):
    @log_function_call('ERROR : Database File')
    def __init__(self,name):
        print('There is a problem loading your database file: %s.  Check the path and file name.' %(name))

class CEPPipelineNetworkException(Exception):
    @log_function_call('ERROR : Network Files')
    def __init__(self):
        print('You have attempted to calculate graphs without network files.  Please make your networks first, then calculate the graphs.')

class CEPPipelinePruningException(Exception):
    @log_function_call('ERROR : Pruning Method')
    def __init__(self,pruning):
        print('Your choice of pruning method, \'%s\', is not supported.  Please check the available methods and change your selection.' %(pruning))

class CEPPipelineSubsetException(Exception):
    @log_function_call('ERROR : Subset Selection')
    def __init__(self,subset):
        print('You must choose to process subset \'a\', subset \'b\', or both \'*\' (default).  You chose: \'%s\'.  Please check your subset selection.' %(subset))

class CEPPipelineMethodException(Exception):
    @log_function_call('ERROR : Method Selection')
    def __init__(self,method):
        print('You must pick a supported method; \'%s\' is not supported.  Please check the method you have selected for making networks.' %(method))

class CEPPipelineResamplingMethodException(AttributeError):
    @log_function_call('ERROR : Resampling Plan Selection')
    def __init__(self,method):
        print('Method %s for resampling not currently supported.  Please choose an allowed resampling plan.' %(method))

class CEPPipelineOptionsConflict(Exception):
    @log_function_call('ERROR : Incompatible Options')
    def __init__(self,method1,method2):
        print('Option %s is not compatible with option %s. Please check your input options.' %(method1,method2))

class CEPPipelineDistanceMethodException(Exception):
    @log_function_call('ERROR : Accuracy Method Error')
    def __init__(self,method):
        print('Method %s for accuracy transformation not recognized.  Please choose from an allowed transformation.' %(method))

class CEPPipelineStatisticsException(Exception):
    @log_function_call('ERROR: Graphs Not Present')
    def __init__(self):
        print('Graphs need to be loaded before statistics can be calculated.  Run read_graphs() then try again.')

class CEPPipelineTests(unittest.TestCase):
    def setUp(self):
        pass

if __name__ == '__main__':
    unittest.main()
