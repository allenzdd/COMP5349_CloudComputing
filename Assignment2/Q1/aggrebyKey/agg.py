
'''
edit by Dongdong Zhang
'''

from pyspark.sql import SparkSession
import argparse
from q1_utils import *
from q1_other_function import *
from pyspark.ml.feature import PCA
from pyspark.sql.functions import *
from time import time
from pyspark.accumulators import AccumulatorParam
import pyspark


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("COMP5349 Assignment 2 KNN") \
        .getOrCreate()

    # set init from arugment
    args = argument_set()

    # take the arugment as follow
    train_datafile = args.train_input
    test_datafile = args.test_input
    k_PCA = args.k_PCA
    k_knn = args.k_knn
    dist_knn = args.distance
    result_show_method = args.show_method

    # read data and assembler and PCA with pipeline
    train_vector_pca, test_vector_pca = pipeline_assembler_pca(
        spark, train_datafile, test_datafile, k=k_PCA)
