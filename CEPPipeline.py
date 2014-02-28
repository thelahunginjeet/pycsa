"""
CEPPipeline.py

Created by CA Brown and KS Brown.  For reference see:
Brown, C.A., Brown, K.S. 2010. Validation of coevolving residue algorithms via 
pipeline sensitivity analysis: ELSC and OMES and ZNMI, oh my! PLoS One, e10779. 
doi:10.1371/journal.pone.0010779

This module is used to control all of the pipeline flow and do 
the reshuffling, etc.  It works with the CEPAlgorithms module which can 
easily be adapted to include more algorithms.  The main pipeline is initialized 
with all of the information for the rest of the project. 
"""

# CHANGED pdb parsing and calculations have been moved into a CEPStructures module

# Big TODOs for next version:
# TODO: wrap parameters into another object for easier specification, perturbation, and passing; 
#        arguments have gotten complicated and multi-level
# TODO: extract mapping to canonical sequence positions from compute_networks: make it a separate
#        step - easier to compare to multiple canonicals this way
# TODO: make it easier to compute new statistics - different acc/rep defs, etc. after having already
#        done the network scoring
# TODO: sequence weighting, rather than drops
# TODO: make restarts much easier

# Misc.
# TODO: __str__ for the pipeline, giving a summary of the options
# TODO: fix thing with resampling Method
# TODO: fix method naming conventions so we don't have to check for case and lower case-ize things
# TODO: log the acc/rep method parameters to the pipeline object (database) so plotting is nicer; 
#        it would be better to use the accuracy limits if we know what they are (for contact and 
#        rescaled, for example)


import sys, os, unittest, random, glob, cPickle, re, itertools
from scipy import mean
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
import numpy as np
import Bio.PDB as pdb
from CEPPreprocessing import SequenceUtilities
from CEPLogging import LogPipeline
import CEPAlgorithms
import CEPNetworks
import CEPStructures


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

    # NOT TESTED THOROUGHLY
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
        dbFile = construct_file_name([self.fileIndicator,self.numSequences,self.method,self.pruning], '', '.pydb')
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
    

    # TODO: remove double counting of canonical sequence
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
        have been deprecated and removed
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
                        significance = {}
                        # map sequence positions to canonical sequence positions
                        canon = {}
                        for column in msa.columns:
                            canon[column] = msa.mapping[column][self.canonSequence]
                        mappedMSA = {}
                        for column1,column2 in msa.__dict__[methods[method]]:
                            mappedMSA[(canon[column1],canon[column2])] = msa.__dict__[methods[method]][(column1,column2)]
                            if pvalues:
                                # set the p-value for writing to a network file
                                significance[(canon[column1],canon[column2])] = msa.pvalues[(column1,column2)]
                            else:
                                # fix the p-value at 1.0
                                significance[(canon[column1],canon[column2])] = 1.0
                        # remove any column pairs where the canonical is gapped
                        keys = mappedMSA.keys()
                        [mappedMSA.pop(p) for p in keys if None in p]
                        output = open(nwkFile,'w')
                        for column1,column2 in mappedMSA:
                            output.write("%d\t%d\t%.8f\t%.8f\n"%(column1,column2,mappedMSA[(column1,column2)],significance[(column1,column2)]))
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
            

    
    @log_function_call('Calculating Graphs from Networks')
    def calculate_graphs(self, method, pruning, number=None):
        """Function reads in networks files and creates graphs from an input pruning tag, for a single method.
        Sets self.method = method for subsequent calls to calculate resampling stats and write databases."""
        pruning = pruning.lower()
        self.pruning = pruning
        self.method = method
        pruningMethods = ('mst','topn','bottomn','pvalue')
        if hasattr(self,'networkDirectory') and pruning in pruningMethods:
            self.initialize_graphs()
            fileNames = construct_file_name([self.fileIndicator,self.numSequences,self.method],'*',self.networkExt)
            nwkFiles = sorted(glob.glob(os.path.join(self.networkDirectory,fileNames)))
            for nwkFile in nwkFiles:
                # initialize graph and methods
                nwkGraph = CEPNetworks.CEPGraph(nwkFile)
                gmethods = {'mst':nwkGraph.calculate_mst, 'topn':nwkGraph.calculate_top_n, 'bottomn':nwkGraph.calculate_bottom_n, \
                'pvalue':nwkGraph.calculate_pvalue}
                # determine split and subset and construct graph
                i,j = determine_split_subset(deconstruct_file_name(nwkFile)[-1])
                gmethods[pruning](number)
                self.graphs[i][j] = nwkGraph
        elif not hasattr(self,'networkDirectory'):
            raise CEPPipelineNetworkException
        elif pruning not in pruningMethods:
            raise CEPPipelinePruningException(pruning)
    

    def _calculate_weighted_distance(self,graph,rescaled=True):
        """Calculates the edge-weighted physical distance for a scoring network.  The weighted distance will almost 
        certainly fail if the weights are not positive.  Returns both the weight sum and the sum of the weighted 
        distances.  Distance will either be bare physical distance (in Angstroms) or rescaled [0,1] distance, using
        the shortest distance in the protein and the protein diameter."""
        if rescaled == True:
            proteinDiameter = mean(self.distances.values()) - min(self.distances.values())
            proteinMinimum = min(self.distances.values())
        else:
            proteinDiameter = 1.0
            proteinMinimum = 0.0
        weightSum,distSum = 0.0,0.0
        for i,j in [x for x in graph.edges() if x in self.distances]:
            weightSum += graph[i][j]['weight']
            distSum += graph[i][j]['weight']*((self.distances[(i,j)]-proteinMinimum)/proteinDiameter)
        return weightSum,distSum
    
    
    # TODO: need to check that graphs exist
    @log_function_call('Calculating Resampling Statistics')
    def calculate_resampling_statistics(self,accMethod='distance',repMethod='splithalf',distMethod='oneminus',simMethod='spearman',rescaled=True,number=None, pdbFile=None, offset=0):
        """Calculates the suite of resampling statistics (accuracy, reproducibility, etc.) from the pruned graphs
        produced from the raw scoring networks.
        
        If you choose contact accuracy, distMethod and simMethod are ignored.  The Jaccard index is used for similarity, and
        accuracy is defined as TP/(TP+FP) when comparing the pruned graphs to the closest contact pairs."""
        # check for consistent options
        if self.resamplingMethod == '_resample_splithalf':
            if repMethod == 'bicar':
                raise CEPPipelineOptionsConflict(self.resamplingMethod,repMethod)
        elif repMethod == 'splithalf':
            raise CEPPipelineOptionsConflict(self.resamplingMethod,repMethod)
        # check that the graphs exist
        if not hasattr(self,'graphs'):
            raise CEPPipelineStatisticsException
        try:
            self.distances = CEPStructures.calculate_distances(pdbFile,offset)
        except IOError:
            raise CEPPipelineStructureIOException(pdbFile)
        else:
            self.statistics = {'reproducibility':{}, 'accuracy':{}}
            self.consensusGraph = CEPNetworks.CEPGraph()
            # for dispatch to reproducibility and accuracy functions
            repFunc = '_calculate_rep_'+repMethod
            accFunc = '_calculate_acc_'+accMethod
            # accuracy can be made one function
            # self._calculate_accuracy(distMethod,rescaled)
            getattr(self,accFunc)(distMethod,rescaled)
            getattr(self,repFunc)(simMethod)

            
    
    @log_function_call('Calculating Distance-Based Accuracy')
    def _calculate_acc_distance(self,distMethod,rescaled):
        """Calculates distance-based accuracy for resamples; defined as some function of the
        score-weighted average distance of pairs in the structure."""
        nR = len(self.graphs)
        nS = len(self.graphs.values()[0])
        for partition in self.graphs:
            wSum, dSum = 0.0,0.0
            for p in self.graphs[partition]:
                # weighted distance sum 
                wS,dS = self._calculate_weighted_distance(self.graphs[partition][p],rescaled)
                wSum += wS/nS
                dSum += dS/nS
            self.statistics['accuracy'][partition] = dSum/wSum
        if distMethod == 'neglog':
            for k in self.statistics['accuracy']:
                self.statistics['accuracy'][k] = -1.0*np.log(self.statistics['accuracy'][k])
        elif distMethod == 'inverse':
            for k in self.statistics['accuracy']:
                self.statistics['accuracy'][k] = 1.0/self.statistics['accuracy'][k]
        elif distMethod == 'oneminus':
            for k in self.statistics['accuracy']:
                self.statistics['accuracy'][k] = 1.0 - self.statistics['accuracy'][k]
        else:
            raise CEPPipelineAccuracyMethodException(distMethod)

    
    # TODO: way to get angCut, distCut in here
    @log_function_call('Calculating Contact-Based Accuracy')
    def _calculate_acc_contacts(self,distMethod,rescaled):
        """Calculates contact-based accuracy for resamples; after pruning, edge weights are
        ignored and the accuracy is simply defined as TP/(TP + FP), where
            TP = number of true positives (top scores which are contacts)
            FP = number of false positives (#edges - TP)
        Hence, acc = TP/#edges in each network."""
        nR = len(self.graphs)
        nS = len(self.graphs.values()[0])
        contacts = CEPStructures.calculate_CASP_contacts(self.distances,distCut=0)
        for partition in self.graphs:
            for p in self.graphs[partition]:
                TP = 0.0
                nE = 0.0
                # TP/#edges
                for e in self.graphs[partition][p].edges():
                    # contacts are stored sorted
                    sEdge = tuple(np.sort(e))
                    TP = TP + contacts.count(sEdge)
                nE = nE + len(self.graphs[partition][p].edges())
            self.statistics['accuracy'][partition] = TP/nE

    
    @log_function_call('Calculating Splithalf Reproducibility')
    def _calculate_rep_splithalf(self,simMethod):
        simFunc = '_graph_similarity_'+simMethod
        norm = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
        for partition in self.graphs:
            gA = self.graphs[partition]['a']
            gB = self.graphs[partition]['b']
            # sub-split similarity
            self.statistics['reproducibility'][partition] = getattr(self,simFunc)(gA,gB)
            # consensus graph
            for i,j in gA.edges()+gB.edges():
                if self.consensusGraph.has_edge(i,j):
                    self.consensusGraph[i][j]['weight'] += 1/norm
                else:
                    self.consensusGraph.add_edge(i,j,weight=1/norm)

    
    # TODO : VERY FRAGILE!  Fix if we want to use for split-half resampling.
    @log_function_call('Calculating BICAR Reproducibility')
    def _calculate_rep_bicar(self,simMethod):
        """Defines reproducibility as the average similarity among all unique pairs of 
        resampled graphs."""
        simFunc = '_graph_similarity_'+simMethod
        nG = float(len(self.graphs.keys())*len(self.graphs.values()[0]))
        norm = (nG*(nG-1.0))/2.0
        avgSim = 0.0
        for p1 in self.graphs:
            for p2 in self.graphs:
                if int(p2) > int(p1):
                    g1 = self.graphs[p1]['boot']
                    g2 = self.graphs[p2]['boot']
                    avgSim += getattr(self,simFunc)(g1,g2)
            # consensus graph
            for i,j in self.graphs[p1]['boot'].edges():
                if self.consensusGraph.has_edge(i,j):
                    self.consensusGraph[i][j]['weight'] += 1/nG
                else:
                    self.consensusGraph.add_edge(i,j,weight=1/nG)
        # normalize
        self.statistics['reproducibility'][0] = avgSim/norm
        
    
    def _graph_similarity_jaccard(self,gOne,gTwo):
        """Calculates the Jaccard similarity (Jaccard index) for two graphs.  Ignores the
        weights in the graphs; simply uses presence and absence of edges."""
        return gOne.compute_jaccard_index(gTwo)
        
    
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

    def _graph_similarity_frobenius(self,gOne,gTwo):
        """Calculates the similarity between two weighted graphs as the Frobenius norm
        (the sum of the squares of the singular values) of the difference in the (weighted) 
        adjacency matrices."""
        deltaA = self._adj_matrix_diff(gOne,gTwo)
        normA = np.sqrt((deltaA**2).sum())
        #return np.exp(-1.0*normA)
        return 1.0/(1.0 + normA)

    def _graph_similarity_spectral(self,gOne,gTwo):
        """Calculates the similarity between two weighted graphs as the spectral norm (largest
        singular value) of the difference in the weighted adjacency matrices."""
        deltaA = self._adj_matrix_diff(gOne,gTwo)
        s = svd(deltaA,compute_uv=False)
        normA = s[0]
        #return np.exp(-1.0*normA)
        return 1.0/(1.0 + normA)
    
    def _graph_similarity_nuclear(self,gOne,gTwo):
        """Calculates the similarity between the two weighted graphs as the nuclear norm (sum of
        the singular values) of the differenc in the weighted adjacency matrices."""
        deltaA = self._adj_matrix_diff(gOne,gTwo)
        s = svd(deltaA,compute_uv=False)
        normA = s.sum()
        #return np.exp(-1.0*s.sum())
        return 1.0/(1.0 + normA)


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
        print "Graphs need to be calculated before statistics can be calculated.  Run calculate_graphs() then try again."

class CEPPipelineTests(unittest.TestCase):
    def setUp(self):
        pass # TODO add unit tests to CEPPipeline

if __name__ == '__main__':
    unittest.main()
