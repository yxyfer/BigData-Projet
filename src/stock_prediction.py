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
from pyspark.ml.feature import MinMaxScaler

import pyspark.sql.functions as func
from pyspark.sql.functions import when


class StockPrediction(object):
    def __init__(self, stock):
        self.stock = stock
        self.predict = self.stock.predict
        self.pred_name = None
        self.pred = None

        self.trainDF = None
        self.testDF = None

    def _train_split(self):
        self.trainDF, self.testDF = self.predict.fullDF.randomSplit(
            [0.8, 0.2], seed=42
        )

    def linear_regression(self):
        self._train_split()

        predicting_col = self.predict.col_to_pred

        rFormula = RFormula(
            formula="next_" + predicting_col + " ~ . - Date",
            featuresCol="features_not_scaled"
        )

        scaler = MinMaxScaler(
            inputCol="features_not_scaled", outputCol="features"
        )

        lr = LinearRegression(
            labelCol="next_" + predicting_col,
            predictionCol="pred_next_" + predicting_col
        )

        pipeline = Pipeline(stages=[rFormula, scaler, lr])
        model = pipeline.fit(self.trainDF)

        return model

    def exec_linear_regression(self):
        if self.pred_name != "linear_reg":
            self.pred_name = "linear_reg"
            self.pred = self.linear_regression().transform(self.testDF)
        return self.pred

    def print_linear_regression(self):
        predicting_col = self.predict.col_to_pred
        pred = self.exec_linear_regression()

        pred.select("next_" + predicting_col,
                    "pred_next_" + predicting_col).show()

    def plot_linear_regression(self):
        predicting_col = self.predict.col_to_pred
        pred = self.exec_linear_regression()

        real = pred.select("next_" + predicting_col)
        predicted = pred.select("pred_next_" + predicting_col)

        plt.plot(real.toPandas(), label="Real next " + predicting_col)
        plt.plot(predicted.toPandas(), label="Predicted next " + predicting_col)

        plt.show()

    def get_up_down(self):
        pred = self.exec_linear_regression()

        predicting_col = self.predict.col_to_pred
        next_name = "next_" + predicting_col
        pred_next_name = "pred_next_" + predicting_col

        pred = pred.withColumn(
            "real_UP", pred[next_name] > pred[predicting_col]
        )
        pred = pred.withColumn(
            "pred_UP", pred[pred_next_name] > pred[predicting_col]
        )
        pred = pred.withColumn("pred_good", pred["real_UP"] == pred["pred_UP"])

        return pred

    def percent_good_up_down(self):
        pred = self.get_up_down()
        nb_value = pred.count()

        nb_good = pred.where(pred["pred_good"] == True).count()

        print(
            "Percent of good prediction of UP/DOWN : " +
            str(nb_good / nb_value) + "%"
        )

    def gain_predict(self):
        pred = self.get_up_down()

        predicting_col = self.predict.col_to_pred
        next_name = "next_" + predicting_col

        pred = pred.withColumn(
            "price_change", func.abs(pred[next_name] - pred[predicting_col])
        )
        pred = pred.withColumn(
            "price_sign",
            when(pred["pred_good"] == True, 1).otherwise(-1)
        )
        pred = pred.withColumn(
            "gain", pred["price_change"] * pred["price_sign"]
        )

        return pred

    def plot_gain_predict(self):
        pred = self.gain_predict()

        df = pred.select("gain").toPandas()
        fig = plt.figure(figsize=(30, 10))
        df.plot.bar()
        plt.show()

    def print_stonks_or_not(self):
        pred = self.gain_predict()
        gain = pred.select(func.sum("gain")).collect()[0][0]

        predicting_col = self.predict.col_to_pred
        start_price = pred.select(predicting_col).first()[0]

        print(
            "With an initial buy at " + str(start_price) + "$ we won " +
            str(gain) + "$"
        )

    def rmse(self):
        # Complete the evaluator code
        predicting_col = self.predict.col_to_pred
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="next_" + predicting_col,
            predictionCol="pred_next_" + predicting_col
        )

        return evaluator

    def eval_rmse(self):
        evaluator = self.rmse()
        print(f"Metric name: {evaluator.getMetricName()}")
        print(f"Label Column: {evaluator.getLabelCol()}")
        print(f"Prediction column: {evaluator.getPredictionCol()}")

        reg = self.exec_linear_regression()
        RMSE = evaluator.evaluate(reg)

        print(f"RMSE value: {RMSE}\n\n")

    def normalised_rmse(self):
        # Normalized RMSE = RMSE / (max value – min value)
        val_max = self.stock.df.select(func.max(self.stock.df['Close'])
                                      ).first()[0]
        val_min = self.stock.df.select(func.min(self.stock.df['Close'])
                                      ).first()[0]

        print(f"Max Value: {val_max}")
        print(f"Min Value: {val_min}")

        evaluator = self.rmse()
        reg = self.exec_linear_regression()
        RMSE = evaluator.evaluate(reg)

        normalised_rmse = RMSE / (val_max - val_min)
        print(f"Normalised RMSE : {normalised_rmse}\n\n")
