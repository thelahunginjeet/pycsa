from numpy import zeros,sqrt,log,pi,mean
from operator import xor,itemgetter
import pycsa.CEPStructures as cepstruc

def sort_by_value(D,reverse=False):
    """
    There are many ways to sort a dictionary by value and return lists/tuples/etc.
    This is one recommended in PEP265.
    """
    return sorted(D.iteritems(),key=itemgetter(1),reverse=reverse)

class ConfusionMatrix(object):
    '''
    The ConfusionMatrix class is designed for working with binary confusion
    matrices. Class methods compute slews of confusion matrix statistics for
    assessing classifier performance.
    '''
    def __init__(self,predicted=None,actual=None):
        self.confmatrix = zeros((2,2))
        self.compute_matrix(predicted,actual)

    def compute_matrix(self,predicted,actual):
        '''
        Fills in the 2x2 binary confusion matrix for input class labels
        predicted and known.  These labels must be 1's and 0's, and can be
        passed in as either lists of integers or strings, e.g.:

            predicted = '10010' and [1,0,0,1,0] are both allowed

        Predicted and actual class labels need to be the same type and
        same length.

        INPUT:
        ------
        predicted : string or array of 1's and 0's, giving classifier
            predictions for binary class labels

        actual    : string or array of true (known) class labels
        '''
        if predicted is None or actual is None:
            return
        assert type(predicted) == type(actual)
        assert len(predicted) == len(actual)
        # zero matrix
        self.confmatrix = zeros((2,2))
        # mapping from class labels to matrix locations
        pn = {1:0, 0:1}
        if type(predicted) == str:
            predicted = [int(c) for c in predicted]
            actual = [int(c) for c in actual]
        for i in xrange(0,len(predicted)):
            row = pn[predicted[i]]
            col = pn[actual[i]]
            self.confmatrix[row,col] += 1
        return

    def classification_accuracy(self):
        return self.confmatrix.trace()/self.confmatrix.sum()

    def error_rate(self):
        return (self.confmatrix[0,1] + self.confmatrix[1,0])/self.confmatrix.sum()

    def precision(self):
        return self.confmatrix[0,0]/(self.confmatrix[0,0] + self.confmatrix[0,1])

    def sensitivity(self):
        return self.confmatrix[0,0]/(self.confmatrix[0,0] + self.confmatrix[1,0])

    def specificity(self):
        return self.confmatrix[1,1]/(self.confmatrix[1,1] + self.confmatrix[0,1])

    def positive_likelihood(self):
        return self.sensitivity()/(1.0 - self.specificity())

    def negative_likelihood(self):
        return self.specificity()/(1.0 - self.sensitivity())

    def balanced_classification_rate(self):
        return 0.5*(self.sensitivity() + self.specificity())

    def balanced_error_rate(self):
        return 1.0 - self.balanced_classification_rate()

    def f_measure(self):
        return 2*self.precision()*self.sensitivity()/(self.precision() + self.sensitivity())

    def g_means(self):
        return sqrt(self.sensitivity()*self.specificity())

    def youdens_J(self):
        return self.sensitivity() - (1.0 - self.specificity())

    def matthews_correlation(self):
        N = self.confmatrix.sum()
        S = (self.confmatrix[0,0]+self.confmatrix[1,0])/N
        P = (self.confmatrix[0,0]+self.confmatrix[0,0])/N
        return (self.confmatrix[0,0]/N - S*P)/sqrt(P*S*(1-P)*(1-S))

    def discriminant_power(self):
        return (sqrt(3.0)/pi)*(log(self.positive_likelihood()) + log(self.negative_likelihood()))




class CEPAccuracyCalculator(object):
    '''
    WRITE DOC FOR OBJECT.
    '''

    def __init__(self,pdbFile,modelNumber=0,chain='A'):
        '''
        Upon initialization, distances/contacts are read from the supplied
        pdb file.
        '''
        self.distances = cepstruc.calculate_distances(pdbFile,modelNumber=modelNumber,chain=chain)
        self.contacts = cepstruc.calculate_CASP_contacts(self.distances)
        self.K = len(self.contacts)


    def calculate(self,scores):
        '''
        Accepts a set of input scores and computes a set of accuracies.  The
        input scores MUST be mapped to real (canonical sequence) positions,
        corresponding to the PDB structure supplied in the class constructor.
        Accuracies computed (see object documentation for the list) are stored
        in the accuracies dictionary.

        INPUT:
        ------
        scores: dictionary, required
            This is a dictionary of pair scores, keyed on (ri,rj) (residue
            numbers in the structure, not in the alignment) with entries equal
            to the values.
        '''
        # hold the results
        self.accuracies = {}
        # make performance strings for the classifier-based methods
        Ps,PI = self.make_performance_strings(scores,self.contacts)
        # compute the confusion matrix for TP/FP/etc. methods
        self.confmatrix = ConfusionMatrix(predicted=Ps,actual=PI)
        # now start filling in accuracy methods
        self.accuracies['hamming'] = self.hamming(Ps,PI)
        self.accuracies['weighted_hamming'] = self.weighted_hamming(Ps,PI)
        self.accuracies['bcr'] = self.confmatrix.balanced_classification_rate()
        self.accuracies['avgdist'] = self.average_distance(scores)
        return


    def average_distance(self,scores):
        '''
        Calculates the scaled edge-weighted physical distance for a set of pair scores.
        The weighted distance will almost certainly fail if the weights are not
        positive.
        '''
        proteinMin = min(self.distances.values())
        proteinD = mean(self.distances.values()) - proteinMin
        weightSum,distSum = 0.0,0.0
        for k in scores:
            if k in self.distances:
                weightSum += scores[k]
                distSum += scores[k]*((self.distances[k] - proteinMin)/proteinD)
        return 1.0 - distSum/weightSum


    def weighted_hamming(self,sx,sy):
        '''
        Computes a positionally weighted hamming distance between the input
        strings and sy.  The distance is given as:

            H_G = (1/Z) sum_{i=1}^N log(i+1)*(1 - delta(sx_i,sy_i))

        and the normalizing factor is


            Z = log(N-K+1) + . . . + log(N-1) + log(N)

        Other sublinear functions (like sqrt(i+1)) give similar rank orderings.
        '''
        assert len(sx) == len(sy)
        sxlist = [int(c) for c in sx]
        sylist = [int(c) for c in sy]
        wthamm = 0.0
        # compute distance
        for i in xrange(0,len(sxlist)):
            wthamm += log(i+2)*xor(sxlist[i],sylist[i])
        # normalizer
        Z = sum([log(i+2) for i in xrange(0,self.K)]) + sum([log(i+2) for i in xrange(len(sx)-self.K,len(sx))])
        return 1.0 - wthamm/Z


    def hamming(self,sx,sy):
        '''
        Computes an accuracy based on the Hamming distance between two strings
        of 0's and 1's.  The distance measure is:

            H = 1.0 - (1/2*K)*H(sx,sy)

        where H(sx,sy) is the Hamming distance between the two strings.
        '''
        assert len(sx) == len(sy)
        count,z = 0,int(sx,2)^int(sy,2)
        while z:
            count += 1
            z &= z-1
        return 1.0 - (1.0/(2*self.K))*count


    def make_performance_strings(self,scores,contacts):
        '''
        Accepts a set of scores and a list of contacts.  Sorts the scores in
        descending order and then returns a binary "performance string" which
        labels scores in the sorted list which are contacts with a '1' and those
        that are not with a '0'.  An ideal performance string is also constructed,
        which has #(contacts) 1's followed by #(scores) - #(contacts) 0's.
        '''
        # sort the scores in descending order
        pairs,vals = zip(*sort_by_value(scores,reverse=True))
        perflist = ['1' if contacts.count(p) > 0 else '0' for p in pairs]
        scorestring  = ''.join(perflist)
        # make the ideal string
        K = len(contacts)
        S = len(scores)
        idealstring = ''.join(['1' for k in xrange(0,K)]+['0' for k in xrange(0,S - K)])
        return scorestring,idealstring
