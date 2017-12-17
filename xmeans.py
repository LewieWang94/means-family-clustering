import ntpath
import random
import time

from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans, splitting_type

from pyclustering.utils import read_sample, timedcall

import pdb


def template_clustering(start_centers, path, tolerance=0.025, criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
                        ccore=False):
    sample = read_sample(path)
    start = time.clock()
    xmeans_instance = xmeans(sample, start_centers, 20, tolerance, criterion, ccore)
    (ticks, _) = timedcall(xmeans_instance.process)

    # clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    end = time.clock()
    print(end - start)

    criterion_string = "UNKNOWN"
    if criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION:
        criterion_string = "BAYESIAN INFORMATION CRITERION"
    elif criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH:
        criterion_string = "MINIMUM NOISELESS DESCRIPTION_LENGTH"

    print("Sample: ", ntpath.basename(path), "\nInitial centers: '", (start_centers is not None),
          "', Execution time: '", ticks, "', Number of clusters:", len(centers), ",", criterion_string, "\n")

    # visualizer = cluster_visualizer()
    # visualizer.set_canvas_title(criterion_string)
    # visualizer.append_clusters(clusters, sample)
    # visualizer.append_cluster(centers, None, marker='*')
    # visualizer.show()


def cluster_sample1():
    # Start with wrong number of clusters.
    start_centers = [[3.7, 5.5]]
    pdb.set_trace()
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE1,
                        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION)
    template_clustering(start_centers, SIMPLE_SAMPLES.SAMPLE_SIMPLE1,
                        criterion=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)


def cluster_sample1_without_initial_centers():
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE1,
                        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION)
    template_clustering(None, SIMPLE_SAMPLES.SAMPLE_SIMPLE1,
                        criterion=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)


if __name__ == '__main__':
    # cluster_sample1()
    # cluster_sample1_without_initial_centers()
    template_clustering(None, 'data/cluster_dim_16_10.txt',
                        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION)

