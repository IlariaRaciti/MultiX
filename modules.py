import numpy as np
import math
import scipy
import random
import matplotlib
matplotlib.use("Agg")
import itertools
from itertools import groupby
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from pyclustering.utils import euclidean_distance
from pyclustering.utils import list_math_addition, list_math_multiplication, list_math_division_number, list_math_subtraction


############ FUNCTIONS ###########

# genes = .name|.chrom|.pos|.expr|[.cluster]
class GeneExpression:
    def __init__(self, line):
        s = line.strip().split("\t")
        self.name = s[0]
        self.chrom = s[1]
        #self.pos = int (s[2])
        self.expr = [float(x[1:-1]) for x in s[2:]]
    def clustera (self, c):
        self.cluster = c

# Distinguish expressed and not expressed genes
def all_same(items):
    return all(x == items[0] for x in items)

# Filter genes that has at least 10% of patients expressed more than one milionesim
def filter_express(item, mean):
    return len(list(filter(lambda x : x> mean, item))) > 0.1 * len(item)

# clusters_parser opens the clusters given by Xmeans by Pelleg
def clusters_parser(line):
    s = line.strip()
    clus = [int(x) for x in s]
    return clus

def expressions_parser(line):
    s = line.strip().split("\t")
    gl = [float(x) for x in s]
    return gl

def maskgenes_parser(line):
    s = line.strip().split("\t")
    mg = [int(x) for x in s]
    return mg


def _randomNormalizedExpression(gg_pp_SS):
    # gg is the percentage of genes to consider
    # S is the complete matrix

    gg, pp, SS = gg_pp_SS

    NN = len(SS)
    PP = len(SS[0])

    # p is the random number of patients
    vu = np.random.uniform(0, 1)
    p = int(pp*PP+vu*(0.1*PP))

    # g is the random number of genes
    mu = np.random.uniform(0, 1)
    g = int(gg*NN+mu*(0.1*NN))

    mask_genes = sorted(random.sample(range(0, NN), g))
    mask_patients = sorted(random.sample(range(0, PP), p))

    ixg = np.ix_(mask_genes, mask_patients)

    returnmatrix = np.matrix(SS)

    return returnmatrix[ixg], mask_genes


def create_genepairs(iteration_maskgenes_clusters):
    # iteration is the number of run
    # maskgenes is the list of actual indeces og genes considered in SS
    # clusters is the list of cluster labels (return of Xmeans)

    iteration, maskgenes, clusters = iteration_maskgenes_clusters

    #gg = len(maskgenes[0])
    gg = len(maskgenes)
    # SetIndeces contains the pairs of genes that are in clusterized as in the same cluster
    SetIndeces = []
    for i in range(0, gg - 1):
        g1 = maskgenes[i]
        for j in range(i + 1, gg):
            g2 = maskgenes[j]
            if (clusters[i] == clusters[j]):
                SetIndeces.append((g1, g2))

    return SetIndeces


# clusterize: finds the clusters from matrix W
def clusterize(W, l, M, N, verbose):
    if verbose == 1:
        print("M:",M,",l:",l)
    G = np.zeros((N,N))
    for i in range(0,N-1):
        for j in range(i+1, N):
            if W[i][j] > (float(l)/float(M)):
                G[i][j] = W[i][j]

    return list(scipy.sparse.csgraph.connected_components(G, directed=False, return_labels=True)[1])

def longest_interval(lst, verbose):
    # lst is is a list of points
    A = list([(k, list(g)) for k, g in groupby(lst, lambda x: x[1])])
    B = [(x[0], x[1], len(x[1])) for x in A]
    maxlength = max(B, key = lambda x: x[2])[2]
    Amax = [x for x in B if x[2] == maxlength]
    if maxlength == 1:
        if verbose == 1:
            print('There is not a longest interval!')
        return [x[1][0] for x in B], 0
    else:
        return list([x[1] for x in Amax]), 1


def long_interval(lst, verbose):
    # lst is is a list of points
    A = list([(k, list(g)) for k, g in groupby(lst, lambda x: x[1])])
    B = [(x[0], x[1], len(x[1])) for x in A]
    Amax = [x for x in B if x[2] > 2]
    if Amax == []:
        if verbose == 1:
            print('There is not a long interval!')
        return [x[1] for x in B], 0
    else:
        return [x[1] for x in Amax], 1


def max_couples(lst):
    # lst is is a list of points
    A = list([(k, list(g)) for k, g in groupby(lst, lambda x: x[1])])
    maxheight =  max(A, key = lambda x: x[0])
    return maxheight[1]


def create_clustering(subset, best_cutoff, M, W, N, verbose):
    # best_cutoff is the optimal cutoff

    final_subgraph = clusterize(W, best_cutoff * M, M, N, verbose)
    #not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) != 1]
    not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) > 9]
    for i in range(len(subset)):
        if final_subgraph[i] in not_singletons:
            subset[i].clustera(final_subgraph[i])
        else:
            subset[i].clustera('N')
    return subset


def inner_correlation(X_g1):
    # X is an expression matrix
    # g1 is the gene with respect to which we compute correlations

    X, g1 = X_g1
    inner_correlations = []
    express1 = X[g1]
    for g2 in range(g1 + 1, len(X)):
        express2 = X[g2]
        inner_correlations.append(np.corrcoef(express1, express2)[0][1])

    return inner_correlations


def compute_correlations(X):
    # X is an expression matrix

    p = Pool(processes=16)
    result = p.map_async(inner_correlation, [(X, i) for i in range(0, len(X) - 1)])
    correlations = result.get()
    p.close()
    p.join()
    p.terminate()

    return list(itertools.chain(*correlations))


def compute_distancematrix(X, correlations):
    # geneset is a GeneExpression class type
    # correlations is an ordered list of correlation values

    distancematrix = np.zeros((len(X), len(X)))
    np.fill_diagonal(distancematrix, 1)
    vals = np.array([x for x in correlations])
    inds = np.triu_indices_from(distancematrix, 1)
    distancematrix[inds] = vals
    distancematrix[(inds[1], inds[0])] = vals

    # distancematrix is the correlation matrix
    return distancematrix

#### MNDL ####
def __get_centers(data, clusters):
    """
@brief Centers of clusters in line with contained objects.
    @param[in] clusters (list): Clusters that contain indexes of objects from data.
    @return (list) Centers.

    """

    centers = [[] for i in range(len(clusters))]
    dimension = len(data[0])

    for index in range(len(clusters)):
        point_sum = [0.0] * dimension

        for index_point in clusters[index]:
            point_sum = list_math_addition(point_sum, data[index_point])

        centers[index] = list_math_division_number(point_sum, len(clusters[index]))

    return centers


def __get_clusters(resultclusterize):
    labels = list(set([x for x in resultclusterize]))

    dictionary = {}
    for k in labels:
        dictionary[k] = [i for i in range(len(resultclusterize)) if resultclusterize[i] == k]

    clusters = []
    for f in dictionary.keys():
        clusters.append(dictionary[f])

    return clusters


def __validate_parallel(cf_M_W_subperform_measure_N_verbose):
    cf,M,W,subperform,measure,N,verbose = cf_M_W_subperform_measure_N_verbose
    par_list_clusters = clusterize(W, cf*M, M, N, verbose)
    clusters = __get_clusters(par_list_clusters)
    #real_clusters = [c for c in clusters if len(c) != 1]

    par_centers = __get_centers(subperform, clusters)

    if measure == 'BIC':
        return __bayesian_information_criterion(subperform, clusters, par_centers)
    if measure == 'MNDL':
        return __minimum_noiseless_description_length(subperform, clusters, par_centers)

def __minimum_noiseless_description_length(data, clusters, centers):
    """
    @brief Calculates splitting criterion for input clusters using minimum noiseless description length criterion.

    @param[in] clusters (list): Clusters for which splitting criterion should be calculated.
    @param[in] centers (list): Centers of the clusters.

    @return (double) Returns splitting criterion in line with bayesian information criterion. 
            Low value of splitting cretion means that current structure is much better.

    """

    scores = [0.0] * len(clusters)

    W = 0.0
    K = len(clusters)
    N = 0.0

    sigma_sqrt = 0.0

    alpha = 0.9
    betta = 0.9

    for index_cluster in range(0, len(clusters), 1):
        for index_object in clusters[index_cluster]:
            delta_vector = list_math_subtraction(data[index_object], centers[index_cluster])
            delta_sqrt = sum(list_math_multiplication(delta_vector, delta_vector))

            W += delta_sqrt
            sigma_sqrt += delta_sqrt

        N += len(clusters[index_cluster])

    if (N - K != 0):
        W /= N

        sigma_sqrt /= (N - K)
        sigma = sigma_sqrt ** 0.5

        for index_cluster in range(0, len(clusters), 1):
            Kw = (1.0 - K / N) * sigma_sqrt
            Ks = (2.0 * alpha * sigma / (N ** 0.5)) + ((alpha ** 2.0) * sigma_sqrt / N + W - Kw / 2.0) ** 0.5
            U = W - Kw + 2.0 * (alpha ** 2.0) * sigma_sqrt / N + Ks

            Z = K * sigma_sqrt / N + U + betta * ((2.0 * K) ** 0.5) * sigma_sqrt / N

            if (Z == 0.0):
                scores[index_cluster] = float("inf")
            else:
                scores[index_cluster] = Z

    else:
        scores = [float("inf")] * len(clusters)

    return sum(scores)

def __bayesian_information_criterion(data, clusters, centers):
    """
    @brief Calculates splitting criterion for input clusters using bayesian information criterion.

    @param[in] clusters (list): Clusters for which splitting criterion should be calculated.
    @param[in] centers (list): Centers of the clusters.

    @return (double) Splitting criterion in line with bayesian information criterion.
            High value of splitting cretion means that current structure is much better.

    """

    scores = [0.0] * len(clusters)  # splitting criterion
    dimension = len(data[0])

    # estimation of the noise variance in the data set
    sigma = 0.0
    K = len(clusters)
    N = 0.0

    for index_cluster in range(0, len(clusters)):
        for index_object in clusters[index_cluster]:
            sigma += (euclidean_distance(data[index_object], centers[index_cluster]))  # It works

        N += len(clusters[index_cluster])

    if (N - K != 0):
        sigma /= (N - K)

        # splitting criterion
        for index_cluster in range(0, len(clusters)):
            n = len(clusters[index_cluster])

            if (sigma > 0.0):
                scores[index_cluster] = n * math.log(n) - n * math.log(N) - n * math.log(
                    2.0 * np.pi) / 2.0 - n * dimension * math.log(sigma) / 2.0 - (n - K) / 2.0

    return sum(scores)

