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

import matplotlib.pyplot as plt

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression


class StockPrediction(object):
    def __init__(self, stock):
        self.stock = stock
        self.predict = self.stock.predict

    def _train_split(self):
        return self.predict.fullDF.randomSplit([0.8, 0.2], seed=42)

    def linear_regression(self, predicting_col="Close"):
        self.predict.load_insights(predicting_col)
        self.trainDF, self.testDF = self._train_split()

        rFormula = RFormula(
            formula="next_" + predicting_col + " ~ . - Date",
            featuresCol="features"
        )

        lr = LinearRegression(
            labelCol="next_" + predicting_col,
            predictionCol="pred_next_" + predicting_col
        )

        pipeline = Pipeline(stages=[rFormula, lr])
        model = pipeline.fit(self.trainDF)

        return model

    def exe_linear_regression(self, predicting_col="Close"):
        return self.linear_regression(predicting_col).transform(self.testDF)

    def draw(self, predicting_col="Close"):
        pred = self.exe_linear_regression(predicting_col)

        real = pred.select("next_" + predicting_col)
        predicted = pred.select("pred_next_" + predicting_col)

        plt.plot(real.toPandas(), label="Real next " + predicting_col)
        plt.plot(predicted.toPandas(), label="Predicted next " + predicting_col)

        plt.show()
