import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#plt.interactive(False)
import sys
import itertools
from timeit import default_timer as timer
import seaborn as sns
from multiprocessing import Pool
import subprocess
import modules

############ ALGORITHM ###########

def MultiX(path, M=100, pp=0.85, pg=0.85, eta=1, type="MNDL", verbose=0 ):
    # M = number of runs
    # pp = percentage of dimesions
    # pg = percentage of data
    # eta =
    # type =
    # verbose =

    # Read data #
    genes = [modules.GeneExpression(x) for x in open(str(path))]

    variances = sorted([np.var(x.expr) for x in genes])
    idx = round(0.9 * len(variances))
    thres = variances[idx]
    subset = [x for x in genes if modules.all_same(x.expr) == False and np.var(x.expr) > thres]

    print('There are', len(subset), 'high variance data over', len(genes), 'total data.')

    S = [(x.expr - np.mean(x.expr)) / np.std(x.expr) for x in subset]
    N = len(S)
    total_number_patients = len(S[0])

    sub_number_patients = int((2 * total_number_patients) / 3)
    print('We use', sub_number_patients, 'dimensions over', total_number_patients, 'to train the model.')

    build = sorted(random.sample(range(0, total_number_patients), sub_number_patients))
    perform = [a for a in list(range(0, total_number_patients)) if a not in build]

    # gene expression matrix for building the model:
    #b_ixgrid = np.ix_(range(0, N), build)
    #subS = np.matrix(S)
    # submatrix = subS[b_ixgrid]
    submatrix = [list(map(lambda x: S[i][x], build)) for i in range(0, N)]

    # gene expression matrix for computing the performance of the model:
    submatrix_perf = [list(map(lambda x: S[i][x], perform)) for i in range(0, N)]

    # M runs of X-Means #
    start_xmeans = timer()
    rexpressions = {}
    maskgenes = {}
    clusters = {}
    for m in range(M):
        print(m)
        rexpressions[m], maskgenes[m] = modules._randomNormalizedExpression((pg, pp, submatrix))
        # Nm = len(rexpressions[m])
        # K_min = round(np.sqrt(Nm))
        # K_max = round(Nm/4)
        K_min = 5
        K_max = 20

        rne_output = open("NormalizedExpressions_" + str(m) + ".tsv", "w")
        for r in np.matrix(rexpressions[m]):
            rne_output.write("\t".join([str(x) for x in np.array(r)[0]]) + "\n")
        rne_output.close()

        subprocess.run(
            ["./kmeans", "makeuni", "in", "NormalizedExpressions_" + str(m)  + ".tsv"])
        subprocess.run(["./kmeans", "kmeans", "-k" + str(K_min) + "", "-method", "blacklist", "-max_leaf_size", "80",
                        "-min_box_width", "0.1",
                        "-max_iter", "200", "-num_splits", "10", "-max_ctrs" + str(K_max) + "", "-cutoff_factor", "0.5",
                        "-in", "NormalizedExpressions_" + str(m) +".tsv", "-printclusters",
                        "clust" + str(m) + "",
                        "-save_ctrs", "ctrs_" + str(m) + ".out"])
        with open("clusters" + str(m) + ".out", "w") as f:
            subprocess.run(["./membership", "clust" + str(m)], stdout=f)

        # Read clusters
        clusters[m] = [modules.clusters_parser(x) for x in open("clusters" + str(m) + ".out")]


    end_xmeans = timer()
    time_xmeans = end_xmeans - start_xmeans

    if verbose == 1 :
        print("Time for ", M," runs of X-Means:", time_xmeans)

    #############################################

    # Computation of W #
    startW = timer()
    p = Pool(processes=16)
    result = p.map_async(modules.create_genepairs, [(iteration, maskgenes[iteration], clusters[iteration]) for iteration in range(0,M)])
    R = result.get()
    p.close()
    p.join()
    p.terminate()

    W_dict = {}
    for r in R:
        for e in r:
            if e in W_dict:
                W_dict[e] += 1
            else:
                W_dict[e] = 1
    W = np.zeros((N, N))
    for f in W_dict.keys():
        W[f[0]][f[1]] = W_dict[f]/float(M)

    end1 = timer()
    timeW = end1 - startW

    if verbose == 1 :
        print("Time for computation of W:", timeW)



    # cutoff and number of clusters
    startcutoff = timer()
    cutoff = []
    num_clusters = []

    for l in range(1, M):
        sys.stdout.write(str(l))
        sys.stdout.flush()
        subgraph = modules.clusterize(W, l, M, N, verbose)

        # c is the number of clusters, noise left out
        c = len(list(set([x for x in subgraph if subgraph.count(x) > eta])))
        num_clusters.append(c)

        # save the point for the cut-plot:
        cutoff.append(float(l) / float(M))
    endcutoff = timer()
    time_cutoff = endcutoff - startcutoff

    if verbose == 1 :
        print("Time for computing cut-plot:", time_cutoff)

    #############################################

    if type == "MNDL_all" :

        # ALL MNDL computation #
        start_all = timer()
        all_cutoffs = [cutoff[i] for i in range(len(cutoff)) if num_clusters[i] != 0 and num_clusters[i] != 1]
        p2 = Pool(processes=16)
        result2 = p2.map_async(modules.__validate_parallel, [(bc, M, W, submatrix_perf, 'MNDL', N, verbose) for bc in all_cutoffs])
        allMNDL = result2.get()
        p2.close()
        p2.join()
        p2.terminate()

        # Find the best cutoff #
        m = max([x for x in allMNDL if x != 0])
        maxindex = [i for i in list(range(len(allMNDL))) if allMNDL[i] == m][0]
        best_cutoff = all_cutoffs[maxindex]
        end_all = timer()
        time_all = end_all - start_all

        if verbose == 1 :
            print("Time for computing the best cut-off:", time_all)

        # Get the optimal configuration #
        final_subgraph = modules.clusterize(W, best_cutoff*M, M, N, verbose)
        not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) > eta]
        singletons = [x for x in final_subgraph if list(final_subgraph).count(x) <= eta]

        for i in range(len(subset)):
            if final_subgraph[i] in not_singletons:
                subset[i].clustera(final_subgraph[i])
            else:
                subset[i].clustera('N')

        # Order data according to the clusters #
        ordered_subset0 = sorted([x for x in subset if x.cluster != 'N'], key=lambda x: x.cluster) + [x for x in subset
                                                                                                      if
                                                                                                      x.cluster == 'N']
        ordered_subset = [x for x in ordered_subset0 if x.cluster in not_singletons]

        plt.figure(1)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)
        ax.set_title('Cut-plot without noise', fontsize=12, color='red')
        plt.step(cutoff, num_clusters)
        plt.xlabel('Cutoff', fontsize=10)
        plt.ylabel('Number of Clusters', fontsize=10)
        plt.axvline(best_cutoff, color='black')
        plt.savefig("Cut-plot.png")

        # Compute the correlation matrix and heatmap#
        correlations = modules.compute_correlations([x.expr for x in ordered_subset])
        distancematrix = modules.compute_distancematrix([x.expr for x in ordered_subset], correlations)

        a4_dims = (5, 5)
        fig, ax = plt.subplots(figsize=a4_dims, )
        sns_plot = sns.heatmap(distancematrix, ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)
        fig = sns_plot.get_figure()
        fig.savefig("Heatmap_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".png")

        # Save the result #
        # uber = name|chrom|ubertad
        uber = [(x.name, x.chrom, x.cluster) for x in ordered_subset] + [(x.name, x.chrom, 'N') for x in subset if
                                                                         x.cluster in singletons]

        output = open("Result_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".tsv", "w")
        for x in uber:
            output.write("\t".join([str(y) for y in x]) + "\n")
        output.close()

        return uber


    if type == "MNDL":

        # Long couples computation #
        startMNDL_l = timer()
        couples = [(cutoff[i], num_clusters[i]) for i in range(len(num_clusters)) if
                   num_clusters[i] != 0 and num_clusters[i] != 1]
        long_couples, NL = modules.long_interval(couples, verbose)

        list_set_of_best_cutoffs = []
        for lc in range(len(long_couples)):
            list_set_of_best_cutoffs.append([x[0] for x in long_couples[lc]])
        set_of_best_cutoffs = list(itertools.chain(*list_set_of_best_cutoffs))

        p2 = Pool(processes=16)
        result2 = p2.map_async(modules.__validate_parallel, [(bc, M, W, submatrix_perf, 'MNDL', N, verbose) for bc in set_of_best_cutoffs])
        MNDL = result2.get()
        p2.close()
        p2.join()
        p2.terminate()

        # Find the best cutoff #
        m = max([x for x in MNDL if x != 0])
        maxindex = [i for i in list(range(len(MNDL))) if MNDL[i]==m][0]
        best_cutoff = set_of_best_cutoffs[maxindex]
        endMNDL_l = timer()
        time_MNDL_l = endMNDL_l -startMNDL_l

        if verbose == 1 :
            print("The best cutoff is", best_cutoff)
            print("Time for computing best cut-off:", time_MNDL_l)

        # Result computation #
        final_subgraph = modules.clusterize(W, best_cutoff*M, M, N, verbose)
        not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) > eta]
        singletons = [x for x in final_subgraph if list(final_subgraph).count(x) <= eta]

        for i in range(len(subset)):
            if final_subgraph[i] in not_singletons:
                subset[i].clustera(final_subgraph[i])
            else:
                subset[i].clustera('N')

        # Order data according to the clusters #
        ordered_subset0 = sorted([x for x in subset if x.cluster != 'N'], key=lambda x: x.cluster) + [x for x in subset
                                                                                                      if
                                                                                                      x.cluster == 'N']
        ordered_subset = [x for x in ordered_subset0 if x.cluster in not_singletons]

        plt.figure(1)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)
        ax.set_title('Cutplot without noise', fontsize=12, color='red')
        plt.step(cutoff, num_clusters)
        plt.xlabel('Cutoff', fontsize=10)
        plt.ylabel('Number of Clusters', fontsize=10)
        plt.axvline(best_cutoff, color='black')
        plt.savefig("Cut-plot.png")

        # compute correlation matrix and heatmap #
        correlations = modules.compute_correlations([x.expr for x in ordered_subset])
        distancematrix = modules.compute_distancematrix([x.expr for x in ordered_subset], correlations)

        a4_dims = (5, 5)
        fig, ax = plt.subplots(figsize=a4_dims, )
        sns_plot = sns.heatmap(distancematrix, ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)
        fig = sns_plot.get_figure()
        fig.savefig("Heatmap_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".png")


        # Save the results #
        # uber = genename|chrom|ubertad
        uber = [(x.name, x.chrom, x.cluster) for x in ordered_subset] + [(x.name, x.chrom, 'N') for x in subset if
                                                                         x.cluster in singletons]

        output = open("Results_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".tsv", "w")
        for x in uber:
            output.write("\t".join([str(y) for y in x]) + "\n")
        output.close()

        return uber


    if type == "BIC":

        # Long couples computation #
        startBIC_l = timer()
        couples = [(cutoff[i], num_clusters[i]) for i in range(len(num_clusters)) if
                   num_clusters[i] != 0 and num_clusters[i] != 1]
        long_couples, NL = modules.long_interval(couples, verbose)

        list_set_of_best_cutoffs = []
        for lc in range(len(long_couples)):
            list_set_of_best_cutoffs.append([x[0] for x in long_couples[lc]])
        set_of_best_cutoffs = list(itertools.chain(*list_set_of_best_cutoffs))

        # Computation of W #
        p2 = Pool(processes=16)
        result2 = p2.map_async(modules.__validate_parallel, [(bc, M, W, submatrix_perf, 'BIC', N, verbose) for bc in set_of_best_cutoffs])
        BIC = result2.get()
        p2.close()
        p2.join()
        p2.terminate()



        # Find the best cutoff #
        m = max([x for x in BIC if x != 0])
        maxindex = [i for i in list(range(len(BIC))) if BIC[i] == m][0]
        best_cutoff = set_of_best_cutoffs[maxindex]
        endBIC_l = timer()
        time_BIC = endBIC_l - startBIC_l
        if verbose == 1 :
            print("The best cutoff is", best_cutoff)
            print("Time for computing best cut-off:", time_BIC)

        # Find the optimal configuration #
        final_subgraph = modules.clusterize(W, best_cutoff*M, M, N, verbose)
        not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) > eta]
        singletons = [x for x in final_subgraph if list(final_subgraph).count(x) <= eta]

        for i in range(len(subset)):
            if final_subgraph[i] in not_singletons:
                subset[i].clustera(final_subgraph[i])
            else:
                subset[i].clustera('N')

        # Order data according to the clusters
        ordered_subset0 = sorted([x for x in subset if x.cluster != 'N'], key=lambda x: x.cluster) + [x for x in subset
                                                                                                      if
                                                                                                      x.cluster == 'N']
        ordered_subset = [x for x in ordered_subset0 if x.cluster in not_singletons]

        plt.figure(1)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)
        ax.set_title('Cutplot without noise', fontsize=12, color='red')
        plt.step(cutoff, num_clusters)
        plt.xlabel('Cutoff', fontsize=10)
        plt.ylabel('Number of Clusters', fontsize=10)
        plt.axvline(best_cutoff, color='black')
        plt.savefig("Cut-plot.png")

        # Compute correlation matrix and heatmap #
        correlations = modules.compute_correlations([x.expr for x in ordered_subset])
        distancematrix = modules.compute_distancematrix([x.expr for x in ordered_subset], correlations)

        a4_dims = (5, 5)
        fig, ax = plt.subplots(figsize=a4_dims, )
        sns_plot = sns.heatmap(distancematrix, ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)
        fig = sns_plot.get_figure()
        fig.savefig("Heatmap_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".png")


        # save the result #
        # uber = genename|chrom|ubertad
        uber = [(x.name, x.chrom, x.cluster) for x in ordered_subset] + [(x.name, x.chrom, 'N') for x in subset if
                                                                         x.cluster in singletons]

        output = open("Results_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".tsv", "w")
        for x in uber:
            output.write("\t".join([str(y) for y in x]) + "\n")
        output.close()

        return uber

    if type == "longest" :

        # Long couples computation #
        startcouples_l = timer()
        couples = [(cutoff[i], num_clusters[i]) for i in range(len(num_clusters)) if
                   num_clusters[i] != 0 and num_clusters[i] != 1]
        the_best, nl = modules.longest_interval(couples, verbose)
        print("The longest pairs:", [x for x in the_best])
        list_the_best = []
        if isinstance(the_best[0], list):
        #if type(the_best[0]) is list:
            for lc in range(len(the_best)):
                list_the_best.append([x[0] for x in the_best[lc]])
            set_the_best = list(itertools.chain(*list_the_best))
        #if type(the_best[0]) == 'tuple':
        else:
            set_the_best = [x[0] for x in the_best]
        print("The longest cutoff:", set_the_best)

        # Computation of W #
        p2 = Pool(processes=16)
        result2 = p2.map_async(modules.__validate_parallel,
                               [(bc, M, W, submatrix_perf, 'MNDL', N, verbose) for bc in set_the_best])
        MNDL = result2.get()
        p2.close()
        p2.join()
        p2.terminate()

        # Find the best cutoff #
        m = min([x for x in MNDL if x != 0])
        maxindex = [i for i in list(range(len(MNDL))) if MNDL[i] == m][0]
        best_cutoff = set_the_best[maxindex]
        endMNDL_ls = timer()
        time_MNDL_ls = endMNDL_ls - startcouples_l

        if verbose == 1 :
            print("The best cutoff is ", best_cutoff)
            print("Time for computing best cut-off:", time_MNDL_ls)

        # Find the optimal configuration #
        final_subgraph = modules.clusterize(W, best_cutoff*M, M, N, verbose)
        not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) > eta]
        singletons = [x for x in final_subgraph if list(final_subgraph).count(x) <= eta]

        for i in range(len(subset)):
            if final_subgraph[i] in not_singletons:
                subset[i].clustera(final_subgraph[i])
            else:
                subset[i].clustera('N')

        # Order data according to the clusters #
        ordered_subset0 = sorted([x for x in subset if x.cluster != 'N'], key=lambda x: x.cluster) + [x for x in subset
                                                                                                      if
                                                                                                      x.cluster == 'N']
        ordered_subset = [x for x in ordered_subset0 if x.cluster in not_singletons]

        plt.figure(1)
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)
        ax.set_title('Cutplot without noise', fontsize=12, color='red')
        plt.step(cutoff, num_clusters)
        plt.xlabel('Cutoff', fontsize=10)
        plt.ylabel('Number of Clusters', fontsize=10)
        plt.axvline(best_cutoff, color='black')
        plt.savefig("Cut-plot.png")

        # Compute correlation matrix and heatmap #
        startdistance_ls = timer()
        correlations = modules.compute_correlations([x.expr for x in ordered_subset])
        distancematrix = modules.compute_distancematrix([x.expr for x in ordered_subset], correlations)

        a4_dims = (5, 5)
        fig, ax = plt.subplots(figsize=a4_dims, )
        sns_plot = sns.heatmap(distancematrix, ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)
        fig = sns_plot.get_figure()
        fig.savefig("Heatmap_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".png")
        enddistance_ls = timer()
        time_distance_ls = enddistance_ls - startdistance_ls

        # save the result #
        # uber = genename|chrom|ubertad
        uber = [(x.name, x.chrom, x.cluster) for x in ordered_subset] + [(x.name, x.chrom, 'N') for x in subset if
                                                                         x.cluster in singletons]

        output = open("Results_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".tsv", "w")
        for x in uber:
            output.write("\t".join([str(y) for y in x]) + "\n")
        output.close()

        return uber


    if type == "Peter" :

        # Find peter-configuration #
        final_subgraph = modules.clusterize(W, 0.8*M, M, N, verbose)
        not_singletons = [x for x in final_subgraph if list(final_subgraph).count(x) > eta]
        singletons = [x for x in final_subgraph if list(final_subgraph).count(x) <= eta]

        for i in range(len(subset)):
            if final_subgraph[i] in not_singletons:
                subset[i].clustera(final_subgraph[i])
            else:
                subset[i].clustera('N')

        # Order data according to the clusters #
        ordered_subset0 = sorted([x for x in subset if x.cluster != 'N'], key=lambda x: x.cluster) + [x for x in subset
                                                                                                      if
                                                                                                      x.cluster == 'N']
        ordered_subset = [x for x in ordered_subset0 if x.cluster in not_singletons]

        correlations = modules.compute_correlations([x.expr for x in ordered_subset])
        distancematrix = modules.compute_distancematrix([x.expr for x in ordered_subset], correlations)

        a4_dims = (5, 5)
        fig, ax = plt.subplots(figsize=a4_dims, )
        sns_plot = sns.heatmap(distancematrix, ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)
        fig = sns_plot.get_figure()
        fig.savefig("Heatmap_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".png")


        # save the result #
        # uber = genename|chrom|ubertad
        print("Start saving results:")
        uber = [(x.name, x.chrom, x.cluster) for x in ordered_subset] + [(x.name, x.chrom, 'N') for x in subset if
                                                                         x.cluster in singletons]

        output = open("Results_Nclusters" + str(len(set(not_singletons))) + "Nnoise" + str(
            len(subset) - len(ordered_subset)) + ".tsv", "w")
        for x in uber:
            output.write("\t".join([str(y) for y in x]) + "\n")
        output.close()

        return uber

