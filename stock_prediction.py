""""
    Name:       Stock Price Prediction
    Authors:    Théo Perinet, Marc Monteil, Mathieu Rivier
    Version:    1.0

    This is a machine learning model to predict the price of the stocks in the
    data folder
"""

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

# To SPlit the dataset
from sklearn.model_selection import train_test_split


class stockPricePredict(object):
    def __init__(self, df):
        self.df = df
        self.trainDF, self.testDF = self._train_split()

    def _train_split(self):
        trainDF, testDF = self.df.randomSplit([0.8, 0.2], seed=42)

    def linear_regression(self):

        # Load and parse the data file.
        # data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
        data = self.df

        # Split the data into training and test sets (30% held out for testing)
        (trainingData, testData) = data.randomSplit([0.7, 0.3])

        # Train a GradientBoostedTrees model.
        #  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
        #         (b) Use more iterations in practice.
        model = GradientBoostedTrees.trainClassifier(
            trainingData, categoricalFeaturesInfo={}, numIterations=3
        )

        # Evaluate model on test instances and compute test error
        predictions = model.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label
                                           ).zip(predictions)
        testErr = labelsAndPredictions.filter(lambda lp: lp[0] != lp[1]).count(
        ) / float(testData.count())
        print("Test Error = " + str(testErr))
        print("Learned classification GBT model:")
