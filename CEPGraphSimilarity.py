from numpy import zeros,sqrt,log,pi,mean,asarray
from operator import xor,itemgetter
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd


class CEPGraphSimilarity(object):
    '''
    Class that computes the similarity of two input graphs (networkx.Graph
    objects) via various means.  Some methods use edge weights and others
    (like Jaccard) do not.
    '''
    def __init__(self):
        return


    def adj_matrix_diff(self,ga,gb,edata_key='weight'):
        '''
        Calculates the difference in weighted adjacency matrices from two input graphs; useful
        for any similarity measure based norm.  All weighted adjacency matrices are have their
        column sums fixed at unity.
        '''
        # edges in one graph and not in another get a zero in the adjacency matrix
        node_union = frozenset(ga.nodes()).union(gb.nodes())
        wa = zeros((len(nodeUnion),len(nodeUnion)))
        wb = zeros((len(nodeUnion),len(nodeUnion)))
        delta_A = zeros((len(nodeUnion),len(nodeUnion)))
        matrix_locs = {}.fromkeys(nodeUnion)
        # map position keys to matrix elements
        cnt = 0
        for k in matrix_locs:
            matrix_locs[k] = cnt
            cnt = cnt + 1
        # make weight matrices by looking at edge union
        edge_union = frozenset(ga.edges()).union(gb.edges())
        for e1,e2 in edge_union:
            n1,n2 = matrixlocs[e1],matrixlocs[e2]
            if ga.has_edge(e1,e2):
                wa[n1,n2] = ga.get_edge_data(e1,e2)[edata_key]
            if gb.has_edge(e1,e2):
                wb[n1,n2] = gb.get_edge_data(e1,e2)[edata_key]
        # correct for scaling; have to take care of zero-sum columns carefully
        nfa = asarray([max(x,1.0e-08) for x in wa.sum(axis=0)])
        nfb = asarray([max(x,1.0e-08) for x in wb.sum(axis=0)])
        return wa/nfa - wb/nfb


    def calculate(self,ga,gb,edata_key,method):
        '''
        Accepts two input graphs (networkx Graphs() or CEPNetwork() objects) and
        computes their similarity based on the method argument.

        INPUT:
        ------
        ga,gb: Graph objects, required
            each should be a valid networkx object

        edata_key : string, required
            key for (continuous) edge data

        method : string, required
            which method to use to compute the similarity
        '''
        sim = getattr(self,'sim_'+method)(ga,gb,edata_key)
        return sim


    def sim_jaccard(self,ga,gb,edata_key='weight'):
        '''
        Jaccard similarity.
        '''
        union = frozenset(ga.edges()).union(gb.edges())
        intersect = frozenset(ga.edges()).intersection(gb.edges())
        try:
            jaccard = 1.0*len(intersect)/len(union)
        except ZeroDivisionError:
            jaccard = 0.0
        return jaccard


    def sim_pearson(self,ga,gb,edata_key='weight'):
        '''
        Pearson correlational similarity between two graphs.
        '''
        # edgeset overlap calculations
        e_union = frozenset(ga.edges()).union(gb.edges())
        e_int = frozenset(ga.edges()).intersection(gb.edges())
        e_uni_a = e_union.difference(gb.edges()))
        e_uni_b = e_union.difference(ga.edges()))
        # weights of edges common to both
        e_list_a = [ga.get_edge_data(x[0],x[1])[edata_key] for x in e_int]
        e_list_b = [gb.get_edge_data(x[0],x[1])[edata_key] for x in e_int]
        # edges in a but not in b
        e_list_one += [ga.get_edge_data(x[0],x[1])[edata_key] for x in e_uni_a]
        e_list_two += [0.0 for x in e_uni_a]
        # edges in b but not in a
        e_list_one += [0.0 for x in e_uni_b]
        e_list_two += [gb.get_edge_data(x[0],x[1])[edata_key] for x in e_uni_b]
        # compute correlation
        return pearsonr(e_list_one,e_list_two)[0]


    def sim_spearman(self,ga,gb,edata_key='weight'):
        '''
        Pearson correlational similarity between two graphs.
        '''
        # edgeset overlap calculations
        e_union = frozenset(ga.edges()).union(gb.edges())
        e_int = frozenset(ga.edges()).intersection(gb.edges())
        e_uni_a = e_union.difference(gb.edges()))
        e_uni_b = e_union.difference(ga.edges()))
        # weights of edges common to both
        e_list_a = [ga.get_edge_data(x[0],x[1])[edata_key] for x in e_int]
        e_list_b = [gb.get_edge_data(x[0],x[1])[edata_key] for x in e_int]
        # edges in a but not in b
        e_list_one += [ga.get_edge_data(x[0],x[1])[edata_key] for x in e_uni_a]
        e_list_two += [0.0 for x in e_uni_a]
        # edges in b but not in a
        e_list_one += [0.0 for x in e_uni_b]
        e_list_two += [gb.get_edge_data(x[0],x[1])[edata_key] for x in e_uni_b]
        # compute correlation
        return spearmanr(e_list_one,e_list_two)[0]


    def sim_frobenius(self,ga,gb,edata_key='weight'):
        '''
        Calculates the similarity between two weighted graphs as the Frobenius
        norm (sum of the squares of the singular values) of the difference in
        the (weighted) adjacency matrices.
        '''
        deltaA = self.adj_matrix_diff(ga,gb,edata_key):
        normA = sqrt((deltaA**2).sum())
        return 1.0/(1.0 + normA)


    def sim_spectral(self,ga,gb,edata_key='weight'):
        '''
        Calculates the similarity between two weighted graphs as the spectral norm
        (largest singular value) of the difference in the weighted adjacency matrices.
        '''
        deltaA = self.adj_matrix_diff(ga,gb,edata_key)
        s = svd(deltaA,compute_uv=False)
        return 1.0/(1.0 + s[0])


    def sim_nuclear(self,ga,gb,edata_key='weight'):
        '''
        Calculates the similarity between the two weighted graphs as the nuclear norm
        (sum of the singular values) of the difference in the weighted adjacency matrices.
        '''
        deltaA = self.adj_matrix_diff(ga,gb,edata_key)
        s = svd(deltaA,compute_uv=False)
        return 1.0/(1.0 + s.sum())
