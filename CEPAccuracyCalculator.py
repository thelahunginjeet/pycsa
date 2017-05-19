from numpy import zeros,sqrt,log,pi,mean,sort
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

    def tp(self):
        '''
        True positives.
        '''
        return self.confmatrix[0,0]

    def tn(self):
        '''
        True negatives.
        '''
        return self.confmatrix[1,1]

    def fp(self):
        '''
        False positives.
        '''
        return self.confmatrix[1,0]

    def fn(self):
        '''
        False negatives.
        '''
        return self.confmatrix[0,1]

    def P(self):
        '''
        Total number of positives.
        '''
        return self.tp() + self.fp()

    def N(self):
        '''
        Total number of negatives.
        '''
        return self.tn() + self.fn()

    def tpr(self):
        '''
        True positive rate; also called sensitivity, hit rate, or recall.
        '''
        return self.tp()/(self.tp() + self.fn())

    def tnr(self):
        '''
        True negative rate; also called specificity.
        '''
        return self.tn()/(self.tn() + self.fp())

    def fpr(self):
        '''
        False positive rate; also called fallout.
        '''
        return self.fp()/(self.fp() + self.tn())

    def fnr(self):
        '''
        False negative rate; also called miss rate.
        '''
        return self.fn()/(self.fn()+self.tp())

    def ppv(self):
        '''
        Positive predictive value; also called precision.
        '''
        return self.tp()/(self.tp() + self.fp())

    def npv(self):
        '''
        Negative predictive value.
        '''
        return self.tn()/(self.tn() + self.fn())

    def fdr(self):
        '''
        False discovery rate.
        '''
        return 1 - self.ppv()

    def rand(self):
        '''
        This is sometimes just called the accuracy.
        '''
        return (self.tp() + self.tn())/(self.P() + self.N())

    def bacc(self):
        '''
        Balanced (for unequal numbers of P and N) accuracy.
        '''
        return (self.tp()/self.P() + self.tn()/self.N())/2

    def f1_score(self):
        '''
        Harmonic mean of precision and sensitivity.
        '''
        return 2.0*self.tp()/(2.0*self.tp() + self.fp() + self.fn())

    def g_means(self):
        return sqrt(self.tpr()*self.tnr())

    def mcc(self):
        '''
        Matthews correlation coefficient; also known as the phi coefficient.
        '''
        num = self.tp()*self.tn() - self.fp()*self.fn()
        A = self.tp() + self.fp()
        B = self.tp() + self.fn()
        C = self.tn() + self.fp()
        D = self.tn() + self.fn()
        den = sqrt(A*B*C*D)
        return num/den

    def informedness(self):
        '''
        Also called DeltaP' and Youden's J.
        '''
        return self.tpr() + self.tnr() - 1

    def markedness(self):
        '''
        Also called DeltaP in the psychology literature.
        '''
        return self.ppv() + self.npv() - 1

    def dor(self):
        '''
        Diagnostic odds rate.
        '''
        return (self.tp()*self.tn())/(self.fp()*self.fn())



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


    def make_performance_strings(self,scores):
        '''
        Accepts a dictionary of scores.  Sorts the scores in descending order and then
        returns a binary "performance string" which labels scores in the sorted list which
        are contacts with a '1' and those that are not with a '0'.  An ideal performance string
        is also constructed, which has #(contacts) 1's followed by #(scores) - #(contacts) 0's.
        '''
        # sort the scores in descending order
        pairs,vals = zip(*sort_by_value(scores,reverse=True))
        perflist = ['1' if self.contacts.count(p) > 0 else '0' for p in pairs]
        scorestring  = ''.join(perflist)
        # make the ideal string
        K = len(self.contacts)
        S = len(scores)
        idealstring = ''.join(['1' for k in xrange(0,K)]+['0' for k in xrange(0,S - K)])
        return scorestring,idealstring


    def calculate(self,scores,method):
        '''
        Accepts a set of input scores and computes an accuracy.  The input scores
        MUST be mapped to real (canonical sequence) positions, corresponding to the
        PDB structure supplied in the class constructor.

        INPUT:
        ------
        scores: dictionary, required
            This is a dictionary of pair scores, keyed on (ri,rj) (residue
            numbers in the structure, not in the alignment) with entries equal
            to the values.

        method : string, required
            which method to use to compute the accuracy
        '''
        acc = 0.0
        if method in ('hamming','whamming'):
            # these rely on the performance and ideal strings
            Ps,PI = self.make_performance_strings(scores)
            acc = getattr(self,'acc_'+method)(Ps,PI)
        elif method in ('rand','bacc','f1_score','informedness','mcc','jaccard'):
            # these need the confusion matrix between Ps and PI as well, and the
            #   functions are members of ConfusionMatrix
            Ps,PI = self.make_performance_strings(scores)
            self.confusion_matrix = ConfusionMatrix(predicted=Ps,actual=PI)
            acc = getattr(self.confusion_matrix,method)()
        elif method in ('avgdist','contact'):
            # these need neither; they are kept for historical reasons
            acc = getattr(self,'acc_'+method)(scores)
        return acc


    def acc_contact(self,scores):
        '''
        Basically a (historically-preserved) version of accuracy that works
        similarly to the perfomance string versions but on a truncated set of
        scores.  Basically the Jaccard (TP/(TP + FP)) on the truncated score
        set, where:
            TP = top scores which are contacts
            FP = len(scores) - TP
        Hence, this definition = TP/len(scores).
        '''
        TP = 0
        for s in scores:
            # contacts are sorted so that ri < rj, ensure this in the score list
            s_edge = tuple(sort(s))
            TP = TP + self.contacts.count(s_edge)
        return 1.0*TP/len(scores)


    def acc_avgdist(self,scores):
        '''
        Calculates the scaled edge-weighted physical distance for a set of pair scores.
        The weighted distance will almost certainly fail if the weights are not
        positive.
        '''
        protein_min = min(self.distances.values())
        protein_dia = mean(self.distances.values()) - protein_min
        weight_sum,dist_sum = 0.0,0.0
        for i,j in scores:
            weight_sum += scores[(i,j)]
            dist_sum += scores[(i,j)]*((self.distances[(i,j)] - protein_min)/protein_dia)
        return 1.0 - dist_sum/weight_sum


    def acc_whamming(self,sx,sy):
        '''
        Computes a positionally weighted hamming distance between the input
        strings sx and sy.  The distance is given as:

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


    def acc_hamming(self,sx,sy):
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
