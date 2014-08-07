"""
CEPAlgorithms.py

This module is used to package all of the two-point correlation 
algorithms (e.g. ZNMI, SCA, OMES, . . .) for the CEPPipeline.  Many
of the algorithms have been removed (e.g. old SCA, ELSC, and others)
that relied Anthony Fodor's code.  These can easily be recoded and 
wrapped back in.

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

3. Neither the name of the University of Connecticut  nor the names of its contributors 
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

import sys, os, unittest, copy, operator
from numpy import log as log
from numpy import log2 as log2
from numpy import exp as exp
from numpy import power as power
from numpy import diag, array, mean, std, var, sqrt, nan_to_num, zeros, ones, sort, triu, tril, real_if_close
from numpy import asmatrix, asarray, matrix, array, multiply, eye, max, min, delete, dot, argmax, sort, hstack
from numpy.random import randn, permutation
from numpy.linalg import cholesky, svd, pinv, inv, eig
from scipy.stats import ks_2samp, linregress
from scipy import randn
from itertools import izip,imap
from pycsa.CEPPreprocessing import SequenceUtilities
from pycsa.CEPLogging import LogPipeline

# rpy2 stuff is for the graphical lasso - current python implementations don't work with
#    the covariance matrix directly, and are therefore utterly useless
# rpy2 juju follows
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

# decorator function to be used for logging purposes
log_function_call = LogPipeline.log_function_call

# helper functions
def clean_zeros(dictionary):
    """
    Simple helper function to remove keys with values less than or equal to zero.
    """
    keys = dictionary.keys()
    [dictionary.pop(x) for x in keys if dictionary[x] <= 0.0]

def prior_covariance(sigmaSample,targetType='dense'):
    """
    Returns a prior covariance estimate sigmaPrior, using the sampleCovariance 
    matrix.  Allowed types:
        'diagonal' : prior is diagonal, with the diagonal elements = mean(diag(sigmaSample))
        'dense' : diagonal elements same as 'diagonal', with off diagonal elements equal to 
                  the average off-diagonal element in sigmaSample
    """
    meanS = mean(diag(sigmaSample))
    sigmaPrior = meanS*eye(sigmaSample.shape[0])
    if targetType is 'dense':
        meanCVS = 2*triu(sigmaSample).sum()/(sigmaSample.shape[0]*(sigmaSample.shape[0]-1))
        sigmaPrior = sigmaPrior + meanCVS*(ones(sigmaSample.shape) - eye(sigmaSample.shape[0]))
    return sigmaPrior

def empirical_shrinkage(sigmaPrior,sigmaSample):
    """
    Returns a shrunken covariance matrix estimate sigmaHat, formed using:
            sigmaHat = rho*sigmaPrior + (1-rho)*sigmaSample
    The rho parameter is empirically determined by choosing the minimum rho which makes
    sigmaHat positive definite.
    """
    rho = 1.0e-02
    mu = 1.1
    while rho < 1.0:
        try:
            cholesky(rho*sigmaPrior + (1.0-rho)*sigmaSample)
            break
        except:
            rho = mu*rho
    # last value of rho gives the minimum shrinkage necessary
    return rho*sigmaPrior + (1.0-rho)*sigmaSample

def fractional_similarity(s1,s2):
    """
    Fraction of similar positions in sequences s1 and s2, computed using the hamming distance.
    Defined as 1 - hamming(s1,s2)/len(s1) (since s1 and s2 must be of identical length).
    """
    assert len(s1) == len(s2)
    # number of dissimilar positions
    h = 1.0*sum(imap(operator.ne,s1,s2))
    return 1.0 - h/len(s1)

    
class MSA(object):
    """
    This is a simple multiple sequence alignment class to be used with MSAAlgorithms.
    """
    def __init__(self,alnFile):
        """
        Simple constructor that loads the sequences, columns, and dimensions and translates the positions
        """
        self.load_msa(alnFile)
        # maps alignment (gapped) positions to sequence (gapless) positions
        self.translate_positions()
        # column-wise gap frequencies
        self.calculate_gap_frequency()
        # check to see if a reference HMMER sequence is present
        if self.sequences.has_key('#=GC RF'):
            self.hmmer = True
        else:
            self.hmmer = False
        # initialize sequence weights (default is equal weights)
        self.seqwts = {}.fromkeys(self.sequences)
        for k in self.seqwts:
            self.seqwts[k] = 1.0

    
    def calculate_sequence_weights(self,method='unweighted',cutoff=0.68):
        """
        Computes weights for similar sequences, discounting contributions to frequent counts
        from similar sequences.
        
        Supported methods:
            
            'unweighted' : weight each sequence equally, regardless of pairwise similarity
                            (sum(wts) = N_sequences)
            
            'similarity' : weight sequences so that those more similar than cutoff receive
                           a total weight of 1.0 (sum(wts) = N_eff)
            
            'henikoff'   : position-based weighting (Henikoff and Henikoff, JMB 1994) (this version
                           normalizes the weights so that sum(wts) = N_sequences)
            
            'symmpurge'  : a symmetric purging method; the number of times each sequence would
                           be purged for being too similar to a target sequence is counted, for
                           the identity of the target being cycled through the sequence set.  
                           The weights are then proportional to 1 - fraction of times purged
                           (normalized so that sum(wts) = N_sequences)
        """
        # clear the weights
        for k in self.seqwts:
            self.seqwts[k] = 1.0
        print 'calculating sequence weights . . .'
        # switch on the method
        if method is 'unweighted':
            return
        elif method is 'similarity' or method is 'symmpurge':
            done = []
            for k1 in self.sequences:
                k2List = [k for k in self.sequences if k not in done and k is not k1]
                for k2 in k2List:
                    flag = (fractional_similarity(self.sequences[k1],self.sequences[k2]) > cutoff)
                    self.seqwts[k1] += 1.0*flag
                    self.seqwts[k2] += 1.0*flag
                done.append(k1)
            if method is 'similarity':
                for k in self.seqwts:
                    self.seqwts[k] = 1.0/self.seqwts[k]
            elif method is 'symmpurge':
                for k in self.seqwts:
                    self.seqwts[k] = 1.0 - ((self.seqwts[k] - 1.0)/len(self.sequences))
                wtsum = sum(self.seqwts.values())
                for k in self.seqwts:
                    self.seqwts[k] = len(self.sequences)*(self.seqwts[k]/wtsum)
        elif method is 'henikoff':
            for k in self.seqwts:
                self.seqwts[k] = 0.0
            for c in self.columns:
                sunique = {}.fromkeys(self.columns[c])
                u = len(sunique)
                for sym in sunique:
                    sunique[sym] = 1.0/(u*self.columns[c].count(sym))
                for s in self.sequences:
                    self.seqwts[s] += sunique[self.sequences[s][c]]
            # normalize
            wtsum = sum(self.seqwts.values())
            for k in self.seqwts:
                self.seqwts[k] = len(self.sequences)*(self.seqwts[k]/wtsum)

    
    def calculate_symbol_counts(self):
        """
        Calculates pair and single amino acid counts, including gaps, for each column in an alignment.
        This function takes account of sequence weighting.
        """
        aminoAcids = tuple('ACDEFGHIKLMNPQRSTVWY-')
        # amino acid to row/col identity map
        aaMap = {}
        for x in xrange(len(aminoAcids)):
            aaMap[aminoAcids[x]] = x
        self.doublets = {}
        self.singlets = {}
        print 'calculating weighted frequencies . . .'
        # singlets
        for c in self.columns:
            self.singlets[c] = matrix(zeros([len(aminoAcids),1], dtype='float64'))
            for s in self.sequences:
                aa = self.sequences[s][c]
                self.singlets[c][aaMap[aa]] += self.seqwts[s]
            # same-site frequences
            self.doublets[(c,c)] = asmatrix(diag(asarray(self.singlets[c]).flat))
        # doublets    
        for column1 in self.columns:
            for column2 in [x for x in self.columns if x > column1]:
                self.doublets[(column1,column2)] = matrix(zeros([len(aminoAcids),len(aminoAcids)],dtype='float64'))
                for s in self.sequences:
                    aa1 = self.sequences[s][column1]
                    aa2 = self.sequences[s][column2]
                    self.doublets[(column1,column2)][aaMap[aa1],aaMap[aa2]] += self.seqwts[s]


    def calculate_consensus_sequence(self):
        """
        Computes a consensus sequence for the MSA, by doing the following:
            1. Henikoff weight the alignment
            2. Compute symbol counts (singlets are used here)
            3. Compute the column entropy, IGNORING GAPS
            4. Compute uncertainty reduction for each column, using 
                    R_c = log2(20) - H_c
            5. Reduce R_c by the gap fraction (R_c' = (1-gaps[c])*R_c
            6. Calculate a score for each position and each non-gap character via 
                    S_c(A) = p_c(A)*R_c'
            7. Find max(S_c(A)) over A, for each c.  That's the consensus character.

        If n > 1, the top n scoring characters are returned.

        For a rough score interpretation, a nongapped column that is partitioned equally among
        the 20 amino acids will have a score of 0.0.

        This function returns an alignment-position indexed dictionary of consensus AAs (this will
        be a tuple of length n) and score (score is only reported for the highest-scoring character), 
        along with a 20 x Npos matrix of character scores, and a corresponding key to identity of the rows.

        No pseudocounting is used in determining character frequencies.
        """
        consensus = {}
        aminoAcids = tuple('ACDEFGHIKLMNPQRSTVWY-')
        self.calculate_sequence_weights(method='Henikoff')
        self.calculate_symbol_counts()
        # copy the frequency info to avoid corruption
        freqs = copy.copy(self.singlets)
        # convert to frequencies, ignoring gaps
        Rscore = {}
        for c in freqs:
            freqs[c] = freqs[c]/freqs[c][0:-1].sum()
            Rscore[c] = (1-self.gaps[c])*(log2(20.0) + 1.0*freqs[c][0:-1].T*log2(freqs[c][0:-1] + 1.0e-12))[0,0]
            # can overwrite the frequency info now (making character scores) and drop the gap
            freqs[c] = Rscore[c]*freqs[c][0:-1]
            # find the consensus character
            maxscore = max(freqs[c])
            maxchar = aminoAcids[argmax(freqs[c])]
            consensus[c] = (maxchar,maxscore)
        # last thing is to dump out the whole matrix of scores
        orderedCols = sort(freqs.keys())
        firstDone = False
        for c in orderedCols:
            if not firstDone:
                scorematrix = hstack([freqs[c]])
                firstDone = True
            else:
                scorematrix = hstack([scorematrix,freqs[c]])
        return consensus,list(aminoAcids[0:-1]),asarray(scorematrix)





    def calculate_symbol_counts_fast(self):
        """
        Calculates pair and single amino acid counts, including gaps, for each column in an alignment.
        This version does not include sequence weighting; we can default to it if the weighting method
        is 'unweighted'.  It is much faster.
        """
        aminoAcids = tuple('ACDEFGHIKLMNPQRSTVWY-')
        # amino acid to row/col identity map
        aaMap = {}
        for x in xrange(len(aminoAcids)):
            aaMap[aminoAcids[x]] = x
        self.doublets = {}
        self.singlets = {}
        print 'calculating frequencies . . .'
        for column1 in self.columns:
            # singlets
            self.singlets[column1] = matrix(zeros([len(aminoAcids),1], dtype='float64'))
            syms = [p for p in self.columns[column1]]
            sunique = {}.fromkeys(syms)
            for s in sunique:
                scount = syms.count(s)
                self.singlets[column1][aaMap[s]] += scount
            # same-site frequencies
            self.doublets[(column1,column1)] = asmatrix(diag(asarray(self.singlets[column1]).flat))
            # doublets
            for column2 in [x for x in self.columns if x > column1]:
                self.doublets[(column1,column2)] = matrix(zeros([len(aminoAcids),len(aminoAcids)], dtype='float64'))
                # get all pairs
                pairs = [p for p in izip(self.columns[column1],self.columns[column2])]
                punique = {}.fromkeys(pairs)
                for p in punique:
                    pcount = pairs.count(p)
                    self.doublets[(column1,column2)][aaMap[p[0]],aaMap[p[1]]] += pcount

    
    def load_msa(self,msaFile):
        """Function to load a multiple sequence alignment for further processing"""
        self.sequences = SequenceUtilities.read_fasta_sequences(msaFile)
        # make sure all of the sequences are uppercase for algorithms
        for s in [x for x in self.sequences if x != '#=GC RF']:
            self.sequences[s] = self.sequences[s].upper()
        # check to make sure that every sequence is the same length (aka an alignment)
        if len({}.fromkeys([len(x) for x in self.sequences.values()])) > 1:
            raise MSADimensionException(msaFile)
        else:
            self.dimensions = (len(self.sequences),len(self.sequences.values()[0]))
            # make a fixed list of sequences to iterate through excluding HMMER reference
            sequenceList = [self.sequences[x] for x in self.sequences if x != '#=GC RF']
            self.columns = {}
            for i in xrange(self.dimensions[1]):
                self.columns[i] = [x[i] for x in sequenceList]
    
    
    def translate_positions(self):
        """Function to make a mapping (dict) between the MSA position and individual sequence positions"""
        self.mapping = dict()
        for column in self.columns:
            position = dict() # Uses physical numbering of positions and NOT Python numbering
            for sequence in self.sequences:
                if self.sequences[sequence][column] is '-':
                    position[sequence] = None
                else:
                    position[sequence] = len(self.sequences[sequence][:column+1])-self.sequences[sequence][:column+1].count('-')
            self.mapping[column] = position
            
    
    def calculate_gap_frequency(self):
        """Function to calculate the gap frequency as a function of column position"""
        self.gaps = dict()
        for column in self.columns:
            self.gaps[column] = self.columns[column].count('-')/float(len(self.columns[column]))



class MSAAlgorithms(MSA):
    """
    Multiple Sequence Alignment Algorithms class for computing alignment pair scores.  Pseudocounting is implemented in a
    variety of ways, controlled by the pcType, pcMix,and pcLambda parameters.  All pseudocouting is covered via mixtures, 
    meaning:
        P = (1-pcMix)*P_true + pcMix*P_psuedo
    where P_true is a density constructed from true counts and P_pseudo is the appropriate pseudocount distribution.  
    
    WARNING: Be careful when comparing the results from 'fixed' to scalar count addition; MI methods which ignore gaps 
    have the effect of increasing the effective pseudocount parameter.  For example, using pcType='fixed' and calculating
    basic MI is equivalent to adding scalar psuedocounts equal to (21.0/20.0)*pcLambda.  
    
    This is not a problem, but just reflects the impossibility of making things work the same way for methods which include
    all symbols (+gaps) and those that do not.

        'none' : do not use pseudocounts at all (P_pseudo = P_true)
        
        'fix'  : adds a fixed number of counts (pcLambda), by fixing rho as 21*lambda/(21*lambda + N_seq); with this option the
                    number of fake counts does not scale with the size of the alignment
        
        'inc'  : adds a fixed weight (pcMix) of P_psuedo, which has the effect of adding more fake counts for larger
                    alignments
        
        'bg'   : adds counts according to background amino acid frequencies/abundances
        
        'emp'  : like background counting, but uses observed frequencies of symbols in the aa
    
    Recalculating P_true is not required when changing psueudocount method; call set_pseudocounting() with the set 
    of desired arguments, and then appropriate scoring method will use the new pseudocount distribution.
    """
    def __init__(self, alnFile, gapFreqCutoff=0.1, percentChange=0.05, pcType='fix', pcMix=0.5, pcLambda=1.0, swtMethod='unweighted', cutoff=0.38):
        """Call the MSA __init__() to load the sequences, columns, dimensions, mapping and set a maximal allowed gap frequency"""
        super(MSAAlgorithms,self).__init__(alnFile)
        self.gapFreqCutoff = gapFreqCutoff
        # minimum percent change needed in a column for the algorithms (e.g. 5% of sites must differ)
        #    note: this effectively sets a bound on the minimum entropy (assumes two letters)
        self.minEntropy = -percentChange*nan_to_num(log(percentChange)) - (1-percentChange)*nan_to_num(log(1-percentChange))
        # calculate the entropy for this alignment; also gives us the reduced columns based on self.minEntropy
        self.calculate_entropy()
        # sequence weighting
        self.calculate_sequence_weights(method=swtMethod,cutoff=cutoff)
        # calculate the pair counts
        if swtMethod == 'unweighted':
            self.calculate_symbol_counts_fast()
        else:
            self.calculate_symbol_counts()
        # set pseudocounting parameters
        self.set_pseudocounting(pcType,pcMix,pcLambda)
        # this is the dictionary of pseudocount types; a tuple of weight, n_i and n_ij are returned
        #   normalization is performed at mixture time
        self.pseudo = {}
        self.pseudo['none'] = (0.0, matrix(zeros([21,1], dtype='float64')), matrix(zeros([21,21],dtype='float64')))
        self.pseudo['fix'] = (21.0*self.pcLambda/(21.0*self.pcLambda + sum(self.seqwts.values())), matrix(ones([21,1],dtype='float64')), matrix(ones([21,21],dtype='float64')))
        self.pseudo['inc'] = (self.pcMix, matrix(ones([21,1],dtype='float64')), matrix(ones([21,21],dtype='float64')))
        freqs = 21.0*matrix(array([0.073,0.025,0.05,0.061,0.042,0.072,0.023,0.053,0.064,0.089,0.023,0.043,0.052,0.04,0.052,0.073,0.056,0.063,0.013,0.033,0.0],dtype='float64')).T
        self.pseudo['bg'] = (self.pcMix, freqs, freqs*freqs.T)
        emfreqs = matrix(zeros([21,1],dtype='float64'))
        # counts exist at this point, so the empirical frequency counts can be used
        for c in self.singlets:
            emfreqs += self.singlets[c]
        emfreqs = 21.0*(emfreqs/emfreqs.sum())
        self.pseudo['emp'] = (self.pcMix, emfreqs, emfreqs*emfreqs.T)


    def set_pseudocounting(self,pcType,pcMix,pcLambda):
        self.pcType = pcType
        self.pcMix = pcMix
        self.pcLambda = pcLambda
        # force recalculation of mixtures
        self.Pij = {}
        self.Pi = {}
        # this will force MI recalculation
        self.mutualInformation = {}


    # might want to include the gaps in column entropy?        
    def calculate_entropy(self):
        """Calculates the column entropy (in nats).  Columns are removed from further calculation if 
        they are not entropic enough (see MSAAlgorithms.__init__() for the definition of the entropy
        cutoff.
        NOTE: This entropy is not used in MI calculation - the marginals are recomputed for each
            pair of columns, since the gaps cause inconsistent values."""
        self.entropy = {}
        # make a list of columns that are less gapped than self.gapFreqCutoff
        columnList = {}.fromkeys([c for c in self.columns if self.gaps[c] < self.gapFreqCutoff])
        print "calculating entropy . . ."        
        for column in columnList:
            letters = {}.fromkeys([x for x in self.columns[column] if x is not '-'])
            numLetters = float(self.dimensions[0]-self.columns[column].count('-'))
            # loop through each observed residue and calculate entropy
            ent = 0.0
            for l in letters:
                # make sure entire column isn't gapped
                try:
                    letters[l] = self.columns[column].count(l)/numLetters
                except ZeroDivisionError:
                    letters[l] = 0.0
                finally:
                    ent -= letters[l]*log(letters[l])
            self.entropy[column] = ent
        # make an even more reduced set of columns; only use those with entropy > minEntropy
        self.reducedColumns = {}.fromkeys([c for c in columnList if self.entropy[c] > self.minEntropy])
        if self.hmmer == True:
            keys = self.reducedColumns.keys()
            [self.reducedColumns.pop(x) for x in keys if self.sequences['#=GC RF'][x] == '-']
        
    
    def calculate_mutual_information(self):
        """Calculates the mutual information for reduced columns, including gaps."""
        self.jointEntropy = {}
        self.mutualInformation = {}
        print 'calculating mutual information including gaps . . .'
        if len(self.Pi) == 0:
            self.mix_distributions()
        for column1 in self.reducedColumns:
            column2List = {}.fromkeys([c for c in self.reducedColumns if c > column1])
            negent1 = -1.0*multiply(self.Pi[column1],nan_to_num(log(self.Pi[column1]))).sum()
            for column2 in column2List:
                # column entropy
                negent2 = -1.0*multiply(self.Pi[column2],nan_to_num(log(self.Pi[column2]))).sum()
                # joint entropy
                self.jointEntropy[(column1,column2)] = -1.0*multiply(self.Pij[(column1,column2)],nan_to_num(log(self.Pij[column1,column2]))).sum()
                self.mutualInformation[(column1,column2)] = negent1 + negent2 - self.jointEntropy[(column1,column2)]
    

    def mix_distributions(self):
        """Uses the true counts and pseudocounting type to compute effective pair and singlet frequencies.
        """
        # construct Pi,Pij from true and pseudocounts
        for c1 in self.reducedColumns:
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            rho,nipc,nijpc = self.pseudo[self.pcType]
            nitrue = self.singlets[c1]
            self.Pi[c1] = (1.0 - rho)*(nitrue/nitrue.sum()) + rho*(nipc/nipc.sum())
            for c2 in c2List:
                nijtrue = self.doublets[(c1,c2)]
                self.Pij[(c1,c2)] = (1.0 - rho)*(nijtrue/nijtrue.sum()) + rho*(nijpc/nijpc.sum())
            # diagonal correction
            nijtrue = self.doublets[(c1,c1)]
            self.Pij[(c1,c1)] = (1.0 - rho)*(nijtrue/nijtrue.sum()) + rho*(asmatrix(diag(asarray(nipc.flat)))/nipc.sum())

    
    def calculate_covariance_matrix(self,Pij,Pi,colToInt):
        """Several algorithms work on the empirical covariance matrix, formed from the pair and single site
        frequencies.  This function computes the full (gaps included) covariance matrix.  The gap portions
        can be removed as desired within the individual algorithms.
        """
        Npositions = len(colToInt)
        covMat = matrix(zeros([Npositions*21,Npositions*21]), dtype='float64')
        # fill up the covariance matrix
        for c1 in self.reducedColumns:
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            indx1 = colToInt[c1]
            for c2 in c2List:
                submatrix = Pij[(c1,c2)] - Pi[c1]*Pi[c2].T                
                # map to the matrix
                indx2 = colToInt[c2]
                covMat[indx1*21:(indx1+1)*21,indx2*21:(indx2+1)*21] = submatrix
                # fill in the transpose
                covMat[indx2*21:(indx2+1)*21,indx1*21:(indx1+1)*21] = submatrix.T
            # diagonal bits
            freqs = Pij[(c1,c1)] - Pi[c1]*Pi[c1].T
            covMat[indx1*21:(indx1+1)*21,indx1*21:(indx1+1)*21] = freqs
        return covMat

        
    def apc_correction(self,S):
        """
        Accepts a dictionary of scores S and performs an average product correction (as initially used in 
        MIp) on those scores.  The input score dictionary is modified during this call.
        """
        columnS = {}.fromkeys(self.reducedColumns)
        meanS = mean(S.values())
        for column in columnS:
            columnS[column] = []
        for column1,column2 in S:
            columnS[column1].append(S[(column1,column2)])
            columnS[column2].append(S[(column1,column2)])
        # calculate S(i,j) = S(i,j) - apc
        for column1,column2 in S:
            mean1 = mean(columnS[column1])
            mean2 = mean(columnS[column2])
            apc = (mean1*mean2)/meanS
            S[(column1,column2)] = S[(column1,column2)] - apc
    
    
    def map_potts_DI(self,eij,Pi,ci,cj):
        """
        Methods which involve covariance matrix calculation/inversion need to have some way to convert the
        (q-1)x(q-1) or q x q (depending on method) matrix of couplings for each pair of positions to a single
        scalar.  One method is the "Direct Information," coming from an assumed two-site model (see
        T. Hwa paper).  This function accepts a matrix of such couplings and returns the (scalar) direct
        information.  This mapping has the benefit of being gauge invariant.
        """
        eps = 1.0e-04
        W = ones([21,21])
        W[0:-1,0:-1] = asmatrix(exp(-1.0*eij),dtype='float64')
        # calculate self-consistent fields
        delta = 1.0
        mui = (1.0/21)*asmatrix(ones([1,21]),dtype='float64')
        muj = (1.0/21)*asmatrix(ones([1,21]),dtype='float64')
        # may not always terminate . . .
        while delta > eps:
            # multiply
            sci = muj*W.T
            scj = mui*W
            # compare to known frequencies
            newi = Pi[ci].T/sci
            newj = Pi[cj].T/scj
            # normalize
            newi = newi/newi.sum()
            newj = newj/newj.sum()
            # difference
            delta = max(abs(newi-mui).max(),abs(newj-muj).max())
            # update and continue
            mui = newi
            muj = newj
        # combine couplings and scfs to get DI
        PDI = multiply(W,mui.T*muj)
        PDI = PDI/PDI.sum()
        freqprod = Pi[ci]*Pi[cj].T
        nmfDI = multiply(PDI,nan_to_num(log(PDI))).sum() - multiply(PDI,nan_to_num(log(freqprod))).sum()
        return nmfDI
            

 
    def calculate_NMI(self):
        """Calculate the mutual information (natural log base) normalized by joint entropy for reducedColumns"""
        self.NMI = {}
        if len(self.mutualInformation) == 0:
            self.calculate_mutual_information()
        print "calculating normalized mutual information (NMI) . . ."
        for column1,column2 in self.mutualInformation:
            # normalization could be zero if minEntropy is zero
            try:
                self.NMI[(column1,column2)] = self.mutualInformation[(column1,column2)]/self.jointEntropy[(column1,column2)]
            except ZeroDivisionError:
                self.NMI[(column1,column2)] = 0.0

    
    def calculate_ZNMI(self):
        """Calculate the z-scored product normalized mutual information (natural log base) for reducedColumns"""
        self.ZNMI = {}
        self.calculate_NMI()
        print "calculating z-scored product normalized mutual information (ZNMI) . . ."
        columnNMI = {}.fromkeys(self.reducedColumns)
        for column in columnNMI:
            columnNMI[column] = []
        for column1,column2 in self.NMI:
            columnNMI[column1].append(self.NMI[(column1,column2)])
            columnNMI[column2].append(self.NMI[(column1,column2)])
        # check columnwise distribution against a gaussian approximation
        self.pvalue = {}
        for column in columnNMI:
            gaussian = std(columnNMI[column])*array(randn(len(columnNMI[column])))+mean(columnNMI[column])
            self.pvalue[column] = ks_2samp(gaussian,columnNMI[column])[1]
        # calculate ZNMI
        for column1,column2 in self.NMI:
            mean1,mean2 = mean(columnNMI[column1]),mean(columnNMI[column2])
            var1,var2 = var(columnNMI[column1]),var(columnNMI[column2])
            meanProduct = (mean1*var2 + mean2*var1)/float(var1+var2)
            stdProduct = sqrt((var1*var2)/(var1+var2))
            self.ZNMI[(column1,column2)] = (self.NMI[(column1,column2)] - meanProduct)/stdProduct
    
    
    def calculate_MIc(self):
        """Calculate another variant of mutual information [Lee and Kim (2009) Bioinformatics] (natural log base) for reducedColumns"""
        self.MIc = {}
        if len(self.mutualInformation) == 0:
            self.calculate_mutual_information()
        print "calculating mutual information corrected (MIc) . . ."
        # CAREFUL IN WHAT COMES BELOW - ncols*(ncols-1)/2 IS NOT THE NUMBER OF KEYS IN THE MI DICT DUE TO 
        #    DROPPING ZEROS
        CPS = {}.fromkeys(self.mutualInformation)
        meanCPS = 0.0
        nmikeys = len(self.mutualInformation)
        # actual, corrected number of columns
        ncols = (1 + sqrt(1+8*nmikeys))/2
        # initialize CPS
        for column1,column2 in CPS:
            CPS[(column1,column2)] = 0.0
        # double loop to calculate column CPS
        for column1,column2 in self.mutualInformation:
            # have to do a check - dictionary may not have the key due to zero dropping
            for columnk in self.reducedColumns:
                if columnk is not column1 and columnk is not column2:
                    tup1 = tuple(sort((column1,columnk)))
                    tup2 = tuple(sort((column2,columnk)))
                    # have to do a check - dictionary may not have the reduced column key due to zero dropping
                    try:
                        cpsPiece = (self.mutualInformation[tup1]*self.mutualInformation[tup2]/(ncols-2.0))
                        CPS[(column1,column2)] += cpsPiece
                    except:
                        cpsPiece = 0.0
                    meanCPS += (1.0/nmikeys)*cpsPiece
        # now do the corrected MI calculation
        meanCPS = sqrt(meanCPS)
        for column1,column2 in self.mutualInformation:
            self.MIc[(column1,column2)] = self.mutualInformation[(column1,column2)] - CPS[(column1,column2)]/meanCPS
    
            
    def calculate_MIp(self):
        """Calculate the mutual information product [Dunn et al. (2008) Bioinformatics] (natural log base) for reducedColumns"""
        self.MIp = {}
        if len(self.mutualInformation) == 0:
            self.calculate_mutual_information()
        print "calculating mutual information product (MIp) . . ."
        self.MIp = copy.copy(self.mutualInformation)
        self.apc_correction(self.MIp)
        
    
    def calculate_Zres(self):
        """Calculate the z-scored residual mutual information [Little & Chen (2009) PLoS One] (natural log base) for reducedColumns"""
        self.Zres = {}
        if len(self.mutualInformation) == 0:
            self.calculate_mutual_information()
        print "calculating z-scored residual mutual information (Zres) . . ."
        columnMI = {}.fromkeys(self.reducedColumns)
        for column in columnMI:
            columnMI[column] = []
        for column1,column2 in self.mutualInformation:
            columnMI[column1].append(self.mutualInformation[(column1,column2)])
            columnMI[column2].append(self.mutualInformation[(column1,column2)])
        # make lists of MI and the column MI mean products 
        listMI = []
        listProduct = []
        for column1,column2 in self.mutualInformation:
            listMI.append(self.mutualInformation[(column1,column2)])
            listProduct.append(mean(columnMI[column1])*mean(columnMI[column2]))
        # linearly regress out the correlation
        slope, intercept, corr, pvalue, sterr = linregress(listProduct,listMI)
        residuals = {}
        columnRes = {}.fromkeys(self.reducedColumns)
        for column in columnRes:
            columnRes[column] = []
        for column1,column2 in self.mutualInformation:
            approx = slope*(mean(columnMI[column1])*mean(columnMI[column2])) + intercept
            residuals[(column1,column2)] = self.mutualInformation[(column1,column2)] - approx
            columnRes[column1].append(self.mutualInformation[(column1,column2)] - approx)
            columnRes[column2].append(self.mutualInformation[(column1,column2)] - approx)
        # calculate zres
        for column1,column2 in residuals:
            mean1 = mean(columnRes[column1])
            mean2 = mean(columnRes[column2])
            std1 = std(columnRes[column1])
            std2 = std(columnRes[column2])
            zres1 = (residuals[(column1,column2)] - mean1)/std1
            zres2 = (residuals[(column1,column2)] - mean2)/std2
            if zres1 > 0 and zres2 > 0:
                self.Zres[(column1,column2)] = zres1 * zres2
            else:
                self.Zres[(column1,column2)] = -abs(zres1 * zres2)


    def calculate_RPMI(self):
        """Calculate the residual product mutual information (natural log base) for reducedColumns
        Note: this is basically a combination of Zres with our product scoring theme"""
        self.RPMI = {}
        self.pvalues = {}
        largePrime = 11173
        if len(self.mutualInformation) == 0:
            self.calculate_mutual_information()
        print "calculating residual product mutual information (RPMI) . . ."
        columnMI = {}.fromkeys(self.reducedColumns)
        for column in columnMI:
            columnMI[column] = []
        for column1,column2 in self.mutualInformation:
            columnMI[column1].append(self.mutualInformation[(column1,column2)])
            columnMI[column2].append(self.mutualInformation[(column1,column2)])
        # make lists of MI and the column MI mean products
        # use only a fraction for speed (5000 random) 
        listMI = []
        listProduct = []
        sample = permutation(self.mutualInformation.keys())
        sample = sample.tolist()[:5000]
        for column1,column2 in sample: 
            listMI.append(self.mutualInformation[(column1,column2)])
            listProduct.append(mean(columnMI[column1])*mean(columnMI[column2]))
        # linearly regress out the correlation
        slope, intercept, corr, pvalue, sterr = linregress(listProduct,listMI)
        residuals = {}
        columnRes = {}.fromkeys(self.reducedColumns)
        for column in columnRes:
            columnRes[column] = []
        for column1,column2 in self.mutualInformation:
            approx = slope*(mean(columnMI[column1])*mean(columnMI[column2])) + intercept
            residuals[(column1,column2)] = self.mutualInformation[(column1,column2)] - approx
            columnRes[column1].append(self.mutualInformation[(column1,column2)] - approx)
            columnRes[column2].append(self.mutualInformation[(column1,column2)] - approx)
        # calculate rpmi
        for column1,column2 in residuals:
            residual = residuals[(column1,column2)]
            mean1,mean2 = mean(columnRes[column1]),mean(columnRes[column2])
            var1,var2 = var(columnRes[column1]),var(columnRes[column2])
            meanProduct = (mean1*var2 + mean2*var1)/float(var1+var2)
            stdProduct = sqrt((var1*var2)/float(var1+var2))            
            # residual must be positive (means can be positive or negative)
            if residual > mean1 and residual > mean2:
                self.RPMI[(column1,column2)] = (residual - meanProduct)/stdProduct
                significance = stdProduct*randn(largePrime) + meanProduct
                pvalue = len([x for x in significance if x > residual])/float(largePrime)
                self.pvalues[(column1,column2)] = pvalue
            else:
                self.RPMI[(column1,column2)] = 0.0
        # use a Bonferroni correction for MHT
        bonferroni = len(self.pvalues)
        for pair in self.pvalues:
            self.pvalues[pair] = min(1.0, bonferroni*self.pvalues[pair])


    def calculate_FCHISQ(self):
        """
        Calculates a statistic proportional to chi-squared (technically chi-squared divided by
        the number of observations).  This allows us to compute the statistic in terms of
        frequencies, rather than counts, integrating it with the rest of the methods.  This
        method therefore includes gaps.
        """
        self.FCHISQ = {}
        print 'calculating frequency-based chi-squared (FCHISQ) . . .'
        if len(self.Pi) == 0:
            self.mix_distributions()
        for ci in self.reducedColumns:
            cjList = {}.fromkeys([c for c in self.reducedColumns if c > ci])
            for cj in cjList:
                Oij = self.Pij[(ci,cj)]
                Eij = self.Pi[ci]*self.Pi[cj].T
                chisq = power(Oij-Eij,2)/Eij
                # only include non-empty Oij cells
                self.FCHISQ[(ci,cj)] = chisq[Oij > 0.0].sum()

    
    def calculate_SCA(self):
        """Calculate the SCA 3.0 [Halabi (2009) Cell] (natural log base) for reducedColumns"""
        print "calculating statistical coupling analysis (SCA) . . ."
        self.SCA = {}
        Nsequences = self.dimensions[0]
        Npositions = len(self.reducedColumns)
        # map column numbers to consecutive integers
        integers = {}
        for i in xrange(Npositions):
            integers[i] = self.reducedColumns.keys()[i]
        # the 20 amino acids including gap and frequencies (excluding gaps)
        codeAA = 'ACDEFGHIKLMNPQRSTVWY-'
        frequencyBG = matrix(array([0.073,0.025,0.05,0.061,0.042,0.072,0.023,0.053,0.064,0.089,0.023,0.043,0.052,0.04,0.052,0.073,0.056,0.063,0.013,0.033],dtype='float64'))
        # determine amino acid frequency at each position
        frequency =  matrix(zeros([21,Npositions],dtype='float64'))
        for i in xrange(len(codeAA)):
            for j in xrange(Npositions):
                frequency[i,j] = [x.upper() for x in self.columns[integers[j]]].count(codeAA[i])/float(Nsequences)
        # determine the most prevalent amino acid at each position
        frequencyBin = matrix(zeros([1,Npositions],dtype='float64'))
        prevalentAA = matrix(zeros([1,Npositions],dtype='int'))
        for i in xrange(Npositions):
            frequencyBin[0,i] = frequency[:len(codeAA)-1,i].max()
            prevalentAA[0,i] = frequency[:len(codeAA)-1,i].argmax()
        # make a simplified alignment in binary approximation (note: gaps ignored)
        msaBin = matrix(zeros([Nsequences,Npositions],dtype='float64'))
        frequencyBGBin = matrix(zeros([1,Npositions],dtype='float64'))
        for i in xrange(Nsequences):
            for j in xrange(Npositions):
                if self.columns[integers[j]][i].upper() == codeAA[prevalentAA[0,j]]:
                    msaBin[i,j] = 1.0
                    frequencyBGBin[0,j] = frequencyBG[0,prevalentAA[0,j]]
        # compute the relative entropy ('D_bin' in Ranganathan's code)
        relativeEntropy = matrix(array(frequencyBin)*array(log(frequencyBin/frequencyBGBin))+array((1-frequencyBin))*array(log((1-frequencyBin)/(1-frequencyBGBin))))
        # compute the frequency pairs, binary correlation, weights, and sca matrix
        frequencyPairsBin = (msaBin.T*msaBin)/Nsequences
        correlationBin = frequencyPairsBin-(frequencyBin.T*frequencyBin)
        weights = matrix(log(array(frequencyBin)*array(1-frequencyBGBin)/(array(frequencyBGBin)*array(1-frequencyBin))))
        scaMatrix = array(weights.T*weights)*array(abs(correlationBin))
        scaMatrix = nan_to_num(scaMatrix)
        for i in xrange(Npositions):
            for j in xrange(i+1,Npositions):
                self.SCA[(integers[i],integers[j])] = scaMatrix[i,j]

    
    def calculate_NMF(self):
        """Calculates an inverse of the sample covariance matrix, in which gaps are not included.  
        The submatrix norm for each positional pair then gives the score.  This is an alternative to 
        the "two-site" model of Direct Information; it (supposedly) carries the same 
        information but does not require self-consistent field calculation.  This algorithm may 
        behave quite poorly without large numbers of pseudocounts.
        """
        print "calculating NMF . . ."
        self.NMF = {}
        # calculate effective pair and singlet probabilities if they do not exist
        if len(self.Pi) == 0:
            self.mix_distributions()
        # ignore the heavily gapped columns
        Npositions = len(self.reducedColumns)
        Nsequences = len(self.sequences)
        # map column positions to consecutive integers, starting at 0
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
            # covariance matrix calculation
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # drop the 21st state
        L = covMat.shape[0]
        covMat = delete(delete(covMat,xrange(20,L,21),axis=0),xrange(20,L,21),axis=1)
        # REGULARIZATION?
        # invert
        eij = inv(covMat)
        #compute submatrix norms
        for c1 in self.reducedColumns:
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                # norm of the appropriate submatrix
                W = -1.0*eij[indx1*20:(indx1+1)*20,indx2*20:(indx2+1)*20]
                self.W = W
                # change the gauge
                sumAlpha = W.sum(axis=0)
                sumBeta = W.sum(axis=1)
                W = W - sumAlpha.T*sumAlpha - sumBeta*sumBeta.T + W.sum()
                #self.INVCOV[(c1,c2)] = (multiply(W,W).sum()/(20*20))
                # this gives pretty big scores . . .
                self.NMF[(c1,c2)] = sqrt(multiply(W,W).sum())


    def calculate_nmfDI(self):
        """Calculates the Direct Information (based on an inverse Ising model), using a naive
        mean-field approximation.  No regularization - beyond pseudocounting if used - is 
        applied to the covariance matrix before inversion.  Thus, this algorithm might behave
        extremely poorly without large numbers of psuedocounts.
        """
        print 'calculating nmfDI . . .'
        self.nmfDI = {}
        # ignore the heavily gapped columns
        Npositions = len(self.reducedColumns)
        Nsequences = len(self.sequences)
        # map column positions to consecutive integers, starting at 0
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
        # mix pseudocounts with real counts to obtain effective pair, singlet freqs.
        if len(self.Pi) == 0:
            self.mix_distributions()
        # covariance matrix calculation
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # drop the 21st state
        L = covMat.shape[0]
        covMat = delete(delete(covMat,xrange(20,L,21),axis=0),xrange(20,L,21),axis=1)
        # REGULARIZATION?
        eij = inv(covMat)
        for c1 in self.reducedColumns:
            # starting index for 21 x 21 submatrix
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                self.nmfDI[(c1,c2)] = self.map_potts_DI(eij[indx1*20:(indx1+1)*20,indx2*20:(indx2+1)*20],self.Pi,c1,c2)
    

    def calculate_ipDI(self):
        """
        Independent pair approximation, using DI to map to a salar.
        """
        print 'calculating ipDI . . .'
        self.ipDI = {}
        Npositions = len(self.reducedColumns)
        Nsequences = len(self.sequences)
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
        # mix
        if len(self.Pi) == 0:
            self.mix_distributions()
        # calculate covariance matrix
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # drop the last state
        L = covMat.shape[0]
        covMat = delete(delete(covMat,xrange(20,L,21),axis=0),xrange(20,L,21),axis=1)
        # now calculate
        for c1 in self.reducedColumns:
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                Cij = covMat[indx1*20:(indx1+1)*20,indx2*20:(indx2+1)*20]
                pi = self.Pi[c1][0:-1]
                pj = self.Pi[c2][0:-1].T
                num1 = 1.0 + Cij/multiply(pi,pj)
                num2 = 1.0 + Cij/multiply(1.0-pi,1.0-pj)
                den1 = 1.0 - Cij/multiply(1.0-pi,pj)
                den2 = 1.0 - Cij/multiply(pi,1.0-pj)
                Jij = log(nan_to_num((multiply(num1,num2)/multiply(den1,den2))))
                # use DI to get a scalar
                self.ipDI[(c1,c2)] = self.map_potts_DI(Jij,self.Pi,c1,c2)
    
    
    
    def calculate_smDI(self):
        """
        Seesak-Monasson, using DI to map to a scalar.
        """
        print 'calculating smDI . . .'
        self.smDI = {}
        Npositions = len(self.reducedColumns)
        Nsequences = len(self.sequences)
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
        # mix
        if len(self.Pi) == 0:
            self.mix_distributions()
        # calculate covariance matrix
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # drop the last state
        L = covMat.shape[0]
        covMat = delete(delete(covMat,xrange(20,L,21),axis=0),xrange(20,L,21),axis=1)
        # regularization?
        eij = inv(covMat)
        for c1 in self.reducedColumns:
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                # use these later
                Cij = covMat[indx1*20:(indx1+1)*20,indx2*20:(indx2+1)*20]
                pi = self.Pi[c1][0:-1]
                pj = self.Pi[c2][0:-1].T
                # loop term
                Jloop = -1.0*eij[indx1*20:(indx1+1)*20,indx2*20:(indx2+1)*20]
                # 2-spin term
                num1 = 1.0 + Cij/multiply(pi,pj)
                num2 = 1.0 + Cij/multiply(1.0-pi,1.0-pj)
                den1 = 1.0 - Cij/multiply(1.0-pi,pj)
                den2 = 1.0 - Cij/multiply(pi,1.0-pj)
                J2spin = log(nan_to_num((multiply(num1,num2)/multiply(den1,den2))))
                # correction for overcounting
                Jcorr = Cij/(multiply(pi,1.0-pi)*multiply(pj,1.0-pj) - multiply(Cij,Cij))
                # use DI to get a scalar
                self.smDI[(c1,c2)] = self.map_potts_DI(Jloop+J2spin-Jcorr,self.Pi,c1,c2)


    
    # TAP equations are correct, but it tends to behave poorly (discriminant is often negative)
    def calculate_tapDI(self):
        """
        Direct Information for scoring, based on Thouless-Anderson-Palmer (TAP) mean-field equations.
        """
        print 'calculating tapDI . . .'
        self.tapDI = {}
        # ignore the heavily gapped columns
        Npositions = len(self.reducedColumns)
        Nsequences = len(self.sequences)
        # column to integer mapping
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
        # mixing with pseudocounts
        if len(self.Pi) == 0:
            self.mix_distributions()
        # covariance matrix calculation
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # drop the last state
        L = covMat.shape[0]
        covMat = delete(delete(covMat,xrange(20,L,21),axis=0),xrange(20,L,21),axis=1)
        # REGULARIZATION BEFORE INVERSION?
        cijinv = inv(covMat)
        # now compute scores/direct information
        for c1 in self.reducedColumns:
            # starting index for submatrix
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                # correlation submatrix (20 x 20 - other couplings are zero)
                Cinv = asmatrix(cijinv[indx1*20:(indx1+1)*20,indx2*20:(indx2+1)*20])
                m1 = 1.0 - 2*self.Pi[c1][0:-1]
                m2 = 1.0 - 2*self.Pi[c2][0:-1]
                eij = (sqrt(1.0 - 2.0*multiply(m1*m2.T,Cinv)) - 1)/(m1*m2.T)
                self.tapDI[(c1,c2)] = self.map_potts_DI(eij,self.Pi,c1,c2)
    

    def calculate_RIDGE(self):
        """Uses L2-regularized inversion of the full covariance matrix to calculate pair scores.  In most respects, 
        this is the L2 equivalent of PSICOV.
        """
        print "calculating RIDGE . . . "
        self.RIDGE = {}
        rglasso = importr('glasso')
        # ignore heavily gapped columns
        Npositions = len(self.reducedColumns)
        # map column positions to consecutive integers, starting at 0
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
        # mix pseudocounts with real counts to obtain effective pair, singlet freqs.
        if len(self.Pi) == 0:
            self.mix_distributions()
        # calculate the covariance matrix
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # first regularize covariance matrix by shrinkage
        sigmaPrior = prior_covariance(covMat,'dense')
        covMat = empirical_shrinkage(sigmaPrior,covMat)
        # L2 problem is just an eigenvalue problem
        s,vt = eig(covMat)
        rho = 0.005
        theta = (1.0/(4.0*rho))*(sqrt(s**2 + 8*rho) - s)
        # this will be the precision matrix
        prec = dot(dot(vt.T,diag(theta)),vt)
        # compute the RIDGE scores
        for c1 in self.reducedColumns:
            # starting index for 21 x 21 submatrix
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                # fetch the appropriate submatrix (ignoring the gaps)
                W = prec[indx1*21:(indx1*21 + 20),indx2*21:(indx2*21 + 20)]
                self.RIDGE[(c1,c2)] = (W*W).sum()
                # in the case of really bad conditioning, scores might have
                #    imaginary components; this allows us to soldier on, though the
                #    scores are probably junk
                self.RIDGE[(c1,c2)] = self.RIDGE[(c1,c2)].real
        # APC
        self.apc_correction(self.RIDGE)


    def calculate_PSICOV(self):
        """
        Uses sparse covariance matrix estimation (Jones et al. 2012) to compute pair scores.
        """
        print "calculating PSICOV . . . "
        self.PSICOV = {}
        rglasso = importr('glasso')
        # ignore heavily gapped columns
        Npositions = len(self.reducedColumns)
        # map column positions to consecutive integers, starting at 0
        colToInt = {}
        for i in xrange(Npositions):
            colToInt[self.reducedColumns.keys()[i]] = i
        # mix pseudocounts with real counts to obtain effective pair, singlet freqs.
        if len(self.Pi) == 0:
            self.mix_distributions()
        # calculate the covariance matrix
        covMat = self.calculate_covariance_matrix(self.Pij,self.Pi,colToInt)
        # regularize covariance matrix by shrinkage before L1 inversion
        sigmaPrior = prior_covariance(covMat,'dense')
        # covMat = empirical_shrinkage(sigmaPrior,covMat)
        covMat = 0.5*covMat + 0.5*sigmaPrior
        # solve sparse problem:  things depend rather critically on the sparseness parameter        
        outDict = rglasso.glasso(asarray(covMat),rho=0.1)
        # overwrite old covariance matrix with the new precision matrix
        covMat = asmatrix(outDict.rx2('wi'))
        # compute the PSICOV scores
        for c1 in self.reducedColumns:
            # starting index for 21 x 21 submatrix
            indx1 = colToInt[c1]
            c2List = {}.fromkeys([c for c in self.reducedColumns if c > c1])
            for c2 in c2List:
                indx2 = colToInt[c2]
                # fetch the appropriate submatrix (ignores the gaps)
                self.PSICOV[(c1,c2)] = abs(covMat[indx1*21:(indx1*21 + 20),indx2*21:(indx2*21 + 20)]).sum()
        # product correction
        self.apc_correction(self.PSICOV)


class MSADimensionException(ValueError):
    def __init__(self,alnFile):
        print "You alignment file, '%s' doesn't appear to have the proper dimensions.  Every sequence should be the same length (aka an alignment).  Please check your input file."%(alnFile)

class MSAAlgorithmTests(unittest.TestCase):
    def setUp(self):
        pass # TODO add unit tests to CEPAlgorithms

if __name__ == '__main__':
    unittest.main()
