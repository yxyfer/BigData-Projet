from src.stock import Stock
from src.stock_prediction import StockPrediction
import pyspark.sql.functions as func
from pyspark.sql.functions import col

import glob

# only used to print informations
import pandas as pd
import numpy as np


class Stocks(object):
    def __init__(
        self,
        files=None,
        header=False,
        delimiter=";",
        schema=None,
        col_to_pred="Close"
    ):

        # load and save stocks
        self.stocks = self._load_stocks(files, header, delimiter, schema)

        # load and save prediction objects
        self.preds = self._load_preds()

        # load insights for prediction
        for stock in self.stocks:
            stock.predict.load_insights(col_to_pred)

    def _load_stocks(self, files, header, delimiter, schema):
        # load stocks contained in files

        # use all the csv files in stocks_data/ if files argument not provided
        if not files:
            files = glob.glob("stocks_data/*.csv")

        dfs = []
        for file in files:
            # call the Stock class to load each stock/file
            dfs.append(
                Stock(file, header=header, delimiter=delimiter, schema=schema)
            )

        return dfs

    def _load_preds(self):
        dfs = []
        for stock in self.stocks:
            # call the StockPrediction class to load each prediction of each stock
            dfs.append(StockPrediction(stock))

        return dfs

    def get_max_daily_return(self):
        # get the maximum daily return for each stocks
        max_daily_returns = {}
        for stock in self.stocks:
            cmpny_name = stock.df.select("company_name").first()[0]
            max_daily_returns[cmpny_name] = (
                stock.analysis.get_daily_return_max()
            )

        return dict(
            sorted(
                max_daily_returns.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

    def get_correlation(self, stock1, stock2):
        # get correlation between two stocks

        # rename columns
        df_stock1 = stock1.select(
            [
                col(col_name).alias(col_name + "_stock1")
                for col_name in stock1.columns
            ]
        )

        df_stock2 = stock2.select(
            [
                col(col_name).alias(col_name + "_stock2")
                for col_name in stock2.columns
            ]
        )

        # join dataframes
        df = df_stock1.join(
            df_stock2, df_stock1.Date_stock1 == df_stock2.Date_stock2, "inner"
        )

        # get columns with double type
        cols = [
            col_info[0] for col_info in stock1.dtypes if col_info[1] == "double"
        ]

        # get correlation
        res = pd.DataFrame(
            1, index=["correlation"], columns=cols, dtype=np.double
        )
        for col_name in cols:
            res[col_name] = df.stat.corr(
                col_name + "_stock1", col_name + "_stock2"
            )

        return res

    def print_correlations(self):
        # print correlation between each stock
        corrs = []

        print("Correlation between each stock:\n")

        for x in range(len(self.stocks)):
            for y in range(x + 1, len(self.stocks)):
                corr = self.get_correlation(
                    self.stocks[x].df, self.stocks[y].df
                )
                corr.index = [
                    self.stocks[x].get_name() + " x " +
                    self.stocks[y].get_name()
                ]
                corrs.append(corr)

        print(pd.concat(corrs))

    def period_return_rate(self, start_date, period="month", nb_days=None):
        return_rates = {}
        for stock in self.stocks:
            cmpny_name = stock.df.select("company_name").first()[0]
            return_rates[cmpny_name] = stock.analysis.get_window_return_rate(
                start_date, period, nb_days
            )

        return_rates = dict(
            sorted(
                return_rates.items(), key=lambda item: item[1], reverse=True
            )
        )
        return return_rates

    def max_period_return_rate(self, start_date, period="month", nb_days=None):
        return_rates = self.period_return_rate(start_date, period, nb_days)
        return list(return_rates.items())[0][0]

    def call_explore_function(self, function_name):
        # call a specific function to each explore object of each stock
        for stock in self.stocks:
            stock.print_name()
            func = getattr(stock.explore, function_name)
            func()

    def call_analysis_function(self, function_name):
        # call a specific function to each analysis object of each stock
        for stock in self.stocks:
            stock.print_name()
            func = getattr(stock.analysis, function_name)
            func()

    def call_insight_function(self, function_name):
        # call a specific function to each insight object of each stock
        for stock in self.stocks:
            stock.print_name()
            func = getattr(stock.insight, function_name)
            func()

    def call_prediction_function(self, function_name):
        # call a specific function to each prediction object of each stock
        for pred in self.preds:
            pred.stock.print_name()
            func = getattr(pred, function_name)
            func()
