'''
edit by lab115_2 May 2018
'''

import numpy as np
import argparse
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.sql.functions import *
from pyspark.accumulators import AccumulatorParam
from pyspark.ml import Pipeline
import pyspark


# init the argument set
def argument_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", help="the train data path",
                        default='/share/MNIST/Train-label-28x28.csv')
    parser.add_argument("--test_input", help="the test data path",
                        default='/share/MNIST/Test-label-28x28.csv')
    parser.add_argument("--k_PCA", help="the k of PCA", default=50, type=int)
    parser.add_argument("--k_knn", help="the k of knn", default=5, type=int)
    parser.add_argument(
        "--distance", help="the distance method of knn", default="euclidean")
    parser.add_argument(
        "--show_method", help="the method of show result", default="accumulator")
    parser.add_argument(
        "--repartition_test", help="the repartition of test data", default=16, type=int)
    args = parser.parse_args()

    # set the test data repartition part in this file with global variables
    global repartition_test
    repartition_test = args.repartition_test
    return args


# the pipeline function with assembler and PCA
def pipeline_assembler_pca(Spark, trainDat, testDat, k=50):
    # read data
    train_df = Spark.read.csv(trainDat, header=False, inferSchema="true")
    test_df = Spark.read.csv(testDat, header=False, inferSchema="true")

    # assembler them
    assembler = VectorAssembler(
        inputCols=train_df.columns[1:], outputCol="features")
    # PCA init
    pca = PCA(k=k, inputCol="features", outputCol="features_pcas")

    # pipeline set
    pipeline = Pipeline(stages=[assembler, pca])
    # fit model
    model = pipeline.fit(train_df)

    # transform train data
    train_pca_result = model.transform(train_df).select(
        col(train_df.columns[0]).alias("label"), "features_pcas")
    # transform test data
    test_pca_result = model.transform(test_df).select(
        col(test_df.columns[0]).alias("label"), "features_pcas") \
        .repartition(repartition_test)  # add repartition
    return train_pca_result, test_pca_result


# init the accumulator from spark document
class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return [0.0] * len(value)

    def addInPlace(self, val1, val2):
        val1 += val2
        return val1


# knn of spark main class function
class knn_Spark(object):
    def __init__(self, Spark, train_data, k_knn=5, distance="euclidean", show_method="accumulator"):
        global trainLab
        global trainFea
        global K_knn
        global Dist

        # broadcast label of train data
        # broadcast features of train data
        list_trainDat = train_data.collect()
        trainLab = Spark.sparkContext.broadcast(
            np.array([x[0] for x in list_trainDat]))
        trainFea = Spark.sparkContext.broadcast(
            np.array([x[1] for x in list_trainDat]))

        # set the k value of knn
        K_knn = k_knn
        # set the distance type
        Dist = distance

        # set whether using accumulator by global variable
        global using_accumlator
        using_accumlator = False
        if show_method == "accumulator":
            global confMatrix
            confMatrix = Spark.sparkContext.accumulator(
                np.zeros((100,)), VectorAccumulatorParam())
            using_accumlator = True

    @staticmethod
    def knn_spark_fit(features, label):
        ##################################################################
        # the four distance method as follow:
        # eu distance with numpy
        if Dist == "euclidean":
            dist = np.linalg.norm(
                trainFea.value - np.array(features), axis=1).reshape(trainLab.value.shape)
        # manhattan distance with numpy
        elif Dist == "manhattan":
            dist = np.abs(trainFea.value - np.array(features)
                          ).sum(axis=1).reshape(trainLab.value.shape)
        # correlation with numpy
        elif Dist == "correlation":
            data_norm1 = (
                (np.array(features) - np.mean(np.array(features))) / np.std(np.array(features)))
            data_norm2 = (trainFea.value - trainFea.value.mean(axis=1).reshape(len(
                trainFea.value), 1)) / trainFea.value.std(axis=1).reshape(len(trainFea.value), 1)
            dist = -np.sum((data_norm1 * data_norm2 / float(
                np.array(features).shape[0])), axis=1).reshape(trainLab.value.shape)
        # chebyshev distance with numpy
        elif Dist == "chebyshev":
            dist = np.abs(trainFea.value - np.array(features)
                          ).max(axis=1).reshape(trainLab.value.shape)
        ###################################################################

        # take top k train data label after calculated distance
        topK_label = np.take(
            trainLab.value, np.argpartition(dist, K_knn, axis=0)[:K_knn])

        # select the most label in the top k label
        counts_test = np.bincount(topK_label)
        Most_test = np.argmax(counts_test)

        # judge whether using accumlator
        if using_accumlator is True:
            # using accumulator to calculate confusion matrix
            global confMatrix
            m = np.zeros((100,))
            m[int(Most_test) * 10 + int(label)] = 1
            confMatrix += m

        return int(Most_test)

    def predict(self, test_data):
        func = udf(self.knn_spark_fit)

        # using withcolumn operation in spark for KNN
        result = test_data \
            .withColumn('predict_label', func(test_data.columns[1], test_data.columns[0])) \
            .select('predict_label', 'label')

        # unpersist broadcast variables
        trainLab.unpersist()
        trainFea.unpersist()

        # return kn result (predict label, real label) and confusion matrix
        if using_accumlator is True:
            return result, confMatrix
        else:
            return result


# show precision recall fscore by pyspark dataframe
def evaluate_precision_recall_fscore(resultDf):
    col_p = "predict_label"
    col_l = "label"
    precisionF = udf(lambda pred, lab_list:
                     list(map(float, lab_list)).count(float(pred)) / float(len(lab_list)) * 100)

    res_precision = resultDf.groupby(col_p).agg(collect_list(col_l).alias("label_list")).repartition(1).cache() \
        .withColumn("precision", round(precisionF(col_p, 'label_list'), 1)).select(col_p, "precision")

    recallF = udf(lambda pred_list, lab:
                  list(map(float, pred_list)).count(float(lab)) / float(len(pred_list)) * 100)

    res_recall = resultDf.groupby(col_l).agg(collect_list(col_p).alias("predict_list")).repartition(1).cache() \
        .withColumn("recall", round(recallF("predict_list", col_l), 1)).select(col_l, "recall")

    fscoreF = udf(lambda precision, recall: (
        (2 * float(precision) * float(recall)) / (float(precision) + float(recall))))

    evaluation_result = res_precision \
        .join(res_recall, res_precision.predict_label == res_recall.label) \
        .select(col_p, "precision", "recall") \
        .withColumn("F-Measure", round(fscoreF("precision", "recall"), 1)).sort(col_p)

    return evaluation_result
