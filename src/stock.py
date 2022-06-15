from pyspark.sql import SparkSession

import matplotlib.pyplot as plt

import pyspark.sql.functions as func
from pyspark.sql.functions import (
    col,
    isnan,
    when,
    count,
    mean,
    lag,
    lead,
    isnull,
    datediff,
    asc,
    lit,
    desc,
    monotonically_increasing_id,
    lit,
)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    DateType,
    StructType,
    StructField,
)

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.window import Window

# only used to print informations
import pandas as pd
import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression

spark_application_name = "WannaFlop_Project"
spark = SparkSession.builder.appName(spark_application_name).getOrCreate()


class Stock(object):
    def __init__(self, file_path, header=False, delimiter=";", schema=None):
        # save attributs
        self.file_path = file_path
        self.header = header
        self.delimiter = delimiter
        self.schema = schema

        # load the file and save the Spark DataFrame
        self.df = self._load_df()

        # save the Explore and Analysis objects of the current stock
        self.explore = self.Explore(self)
        self.analysis = self.Analysis(self)
        self.insight = self.Insight(self)
        self.predict = self.Predict(self)

    def get_name(self):
        return self.file_path.split(".")[-2].split("/")[-1]

    def print_name(self):
        # print the name of the stock
        name = self.get_name()
        print("###########" + "#" * len(name) + "###########")
        print("########## " + name + " ##########")
        print("###########" + "#" * len(name) + "###########\n")

    def _get_num_cols(self):
        # get column names with double or integer type
        num_cols = [
            f.name for f in self.df.schema.fields
            if isinstance(f.dataType, DoubleType) or
            isinstance(f.dataType, IntegerType)
        ]

        return num_cols

    def _get_rounded_df(self):
        # round each double/integer values of each columns
        df = self.df
        for col in self._get_num_cols():
            df = df.withColumn(col, func.round(col))

        return df

    def _handle_csv(self):
        # Read the csv file and return a Spark DataFrame
        return (
            spark.read.option("inferSchema",
                              "true").option("nullValue", "null").csv(
                                  self.file_path,
                                  sep=self.delimiter,
                                  schema=self.schema,
                                  header=self.header,
                              )
        )

    def _handle_json(self):
        # Read the json file and return a Spark DataFrame
        return spark.read.schema(self.schema).json(self.file_path)

    def _load_df(self):
        # load the file depending of the extension
        df = None

        extension = self.file_path.split("."
                                        )[-1] if "." in self.file_path else ""
        if extension == "json":
            df = self._handle_json()
        elif extension == "csv":
            df = self._handle_csv()
        else:
            raise Exception(
                "Current file format is not handled yet. We support csv and json formats."
            )

        return df

    class Explore:
        def __init__(self, stock):
            # save attributs
            self.stock = stock

        def __repr__(self):
            # print every infos on the stock
            return f"""
            {self._print_schema()}\n
            {self.get_df_abstract()}\n
            {self._nb_rows()}\n
            {self.get_stats()}\n
            {self.get_missing()}"""

        def get_missing(self):
            # count the number of missing data on each colums
            print("Missing Data per column:")
            self._count_missing().show()

        def get_df_abstract(self):
            # print the first and last 40 rows of the stock
            rounded_df = self.stock._get_rounded_df()

            # First 40 rows
            print("First 40 rows:")
            rounded_df.show(40)

            # Last 40 rows
            print("Last 40 rows:")
            rounded_df.orderBy(desc("Date")).show(40)

        def get_stats(self):
            # get stat for each columns
            rounded_df = self.stock.df
            summary = rounded_df.summary()
            cols = self.stock._get_num_cols()
            for col in cols:
                summary = summary.withColumn(col, func.round(col))
            print("Stock Stats:")
            summary.show()

        def nb_rows(self):
            # Number of total rows
            print("Number of rows: " + str(self.stock.df.count()) + "\n")

        def correlation(self):
            # print correlation between each double/int columuns
            cols = self.stock._get_num_cols()

            res = pd.DataFrame(1, index=cols, columns=cols, dtype=np.double)

            for x in range(len(cols)):
                for y in range(x + 1, len(cols)):
                    col_1, col_2 = cols[x], cols[y]
                    cor = self.stock.df.stat.corr(col_1, col_2)
                    res.loc[col_1, col_2] = cor
                    res.loc[col_2, col_1] = cor

            print(res)
            print("")

        def period(self):
            # get the period between data points : "day", "week", "month", "year"
            my_window = Window.partitionBy().orderBy("Date")
            # add infos

            df = self.stock.df
            df = df.withColumn("prev_value", lag(df.Date).over(my_window))
            df = df.withColumn(
                "diff",
                when(isnull(datediff(df.Date, df.prev_value)),
                     0).otherwise(datediff(df.Date, df.prev_value)),
            )
            mean_diff = df.select(mean("diff")).first()[0]

            if mean_diff < 4:
                return "day"
            elif mean_diff < 8:
                return "week"
            elif mean_diff < 40:
                return "month"
            return "year"

        def print_period(self):
            print("Period: " + self.period())

        def _print_schema(self):
            self.stock.df.printSchema()

        def _count_missing(self):
            # count the number of missing values
            cols = self.stock.df.columns
            cols.remove("Date")
            return self.stock.df.select(
                [count(when(isnan(c), c)).alias(c) for c in cols]
            )

    class Analysis:
        def __init__(self, stock):
            self.stock = stock
            self.df = stock.df

        def get_oc_avg(self, period):
            # get average opening / closing price
            close = self._compute_avg(self.df, "Close", period)
            opening = self._compute_avg(self.df, "Open", period)

            if period == "week":
                return (
                    close.join(
                        opening,
                        (opening.Date == close.Date) &
                        (opening.Week_number == close.Week_number),
                        "inner",
                    ).select(
                        close.Date,
                        close.Week_number,
                        opening.Open_mean,
                        close.Close_mean,
                    ).orderBy("Date", "Week_number")
                )
            return (
                close.join(opening, opening.Date == close.Date, "inner").select(
                    close.Date, opening.Open_mean, close.Close_mean
                ).orderBy("Date")
            )

        def print_get_oc_avg(self):
            # print average opening / closing price for each period
            for period in ["week", "month", "year"]:
                self.get_oc_avg(period).show()

        def get_daily_return(self, period="day"):
            # get daily return by period
            df = self.get_oc_avg(period)
            return df.withColumn(
                "daily_return_" + period, (df["Close_mean"] - df["Open_mean"])
            )

        def print_daily_return(self):
            self.get_daily_return("day").drop("Open_mean", "Close_mean").show()

        def print_daily_return_avg(self):
            for period in ["week", "month", "year"]:
                if period == "week":
                    self.get_daily_return(period).select(
                        "Date",
                        "Week_number",
                        col("daily_return_" +
                            period).alias("avg_daily_return_" + period),
                    ).show()
                else:
                    self.get_daily_return(period).select(
                        "Date",
                        col("daily_return_" +
                            period).alias("avg_daily_return_" + period),
                    ).show()

        def get_daily_return_rate(
            self, period="day", start_price=None, nb_shares=1
        ):
            df = self.get_daily_return(period)
            if not start_price:
                start_price = df["Open_mean"] * nb_shares

            daily_r = df.withColumn(
                "daily_return_rate_" + period,
                ((df["daily_return_" + period] * nb_shares) / start_price) *
                100,
            ).drop("Open_mean", "Close_mean", "daily_return_" + period)
            return daily_r

        def print_daily_return_rate_avg(self):
            for period in ["week", "month", "year"]:
                self.get_daily_return_rate(period).show()

        def get_price_change(self, period="day"):
            # compute the price change by a period
            df = self.df

            # get periods value for each row
            date = {"day": "yyyy-MM-dd", "month": "yyyy-MM", "year": "yyyy"}
            df = df.withColumn(
                "Date_period", func.date_format(func.col("Date"), date[period])
            )

            # get first and last row of periods
            w = Window.partitionBy("Date_period")

            df_first = (
                df.withColumn("Date_first",
                              func.min("Date").over(w)).where(
                                  col("Date") == col("Date_first")
                              ).select("Date_period", "Open")
            )

            df_last = (
                df.withColumn("Date_last",
                              func.max("Date").over(w)).where(
                                  col("Date") == col("Date_last")
                              ).select(
                                  col("Date_period").alias("Date_period_last"),
                                  "Close"
                              )
            )

            # join results
            df = (
                df_first.join(
                    df_last, df_first.Date_period == df_last.Date_period_last,
                    "inner"
                ).select("Date_period", "Open", "Close").orderBy("Date_period")
            )

            return df.withColumn(
                "price_change", (df["Close"] - df["Open"])
            ).select(
                col("Date_period").alias("Date_period_" + period),
                "price_change"
            )

        def print_price_change(self):
            for period in ["day", "month", "year"]:
                self.get_price_change(period).show()

        def get_daily_return_max(self, period="day"):
            # get maximum daily return by period
            df = self.get_daily_return(period)
            return df.select(func.max("daily_return_" + period)).first()[0]

        def get_daily_return_rate_max(
            self, period="day", start_price=None, nb_shares=1
        ):
            df = self.get_daily_return(period, start_price, nb_shares)
            return df.select(func.max("diff")).first()[0]

        def moving_average(self, df=None, col="Close", nb_points=5):
            SECS_IN_DAY = 86400
            period = nb_points * SECS_IN_DAY

            if df is None:
                df = self.stock.df

            df = df.withColumn("Date2", df.Date.cast("timestamp"))
            window = (
                Window().partitionBy(lit("Date2")).orderBy(
                    func.col("Date2").cast("long")
                ).rangeBetween(-period, 0)
            )

            df = df.withColumn(
                "moving_average_" + col,
                func.avg(col).over(window)
            )

            return df

        def print_moving_average(self, df=None, col="Close", nb_points=5):
            self.moving_average(df, col, nb_points).select(
                "Date", "moving_average_" + col
            ).show()

        def get_window_return_rate(
            self, start_date, period="month", nb_days=None
        ):
            """
            # return is in % already !!!!!! 
            """
            df = self.stock.df
            period_days = {"month": 30, "year": 365}
            if not nb_days:
                nb_days = period_days[period]

            nb_days += 1    # For the exclusive end
            date_from = func.to_date(lit(start_date))
            date_from = func.date_add(date_from, -1)
            date_to = func.date_add(date_from, nb_days)
            df = df.filter(df.Date > date_from
                          ).filter(df.Date < date_to
                                  ).select("Date", "Open", "Close")

            SECS_IN_DAY = 86400 * nb_days
            period = period * SECS_IN_DAY
            df = df.withColumn("window_return", (df["Close"] - df["Open"]))

            start_price = df.first()['Open']
            daily_r = df.withColumn(
                "eee",
                ((df["window_return"]) / start_price) * 100,
            )    #.drop("Open_mean", "Close_mean", "daily_return_" + period)

            # return is in % already !!!!!!
            return daily_r.agg(func.avg('eee').alias('avg')).first()[0]

        def _compute_avg(self, df, col, period):
            # compute the average value of a column by a period
            if period == "week":
                # for week, we une the number of the week
                df = (
                    df.withColumn(
                        "Week_number", func.weekofyear(func.col("Date"))
                    ).withColumn(
                        "Date", func.date_format(func.col("Date"), "yyyy")
                    ).groupBy("Date", "Week_number").agg(
                        mean(col).alias(col + "_mean")
                    ).orderBy("Date", "Week_number")
                )
                return df

            date = {"day": "yyyy-MM-dd", "month": "yyyy-MM", "year": "yyyy"}
            df = (
                df.withColumn(
                    "Date", func.date_format(func.col("Date"), date[period])
                ).groupBy("Date").agg(mean(col).alias(col + "_mean")
                                     ).orderBy("Date")
            )
            return df

    class Insight:
        def __init__(self, stock):
            # save attributs
            self.stock = stock

        def get_r_de_williams(self, n_days=14, df=None):
            if df is None:
                df = self.stock.df

            SECS_IN_DAY = 86400
            period = n_days * SECS_IN_DAY

            df = df.withColumn("Date2", df.Date.cast("timestamp"))

            w = Window().partitionBy(lit("Date2"))\
                    .orderBy(col("Date2").cast("long"))\
                    .rangeBetween(-period, 0)

            df = df.withColumn("PBn", func.min("Low").over(w))
            df = df.withColumn("PHn", func.max("High").over(w))
            df = df.withColumn(
                "R_de_williams", 100 -
                ((df["PHn"] - df["Close"]) / (df["PHn"] - df["PBn"])) * 100
            ).drop("PHn", "PBn", "Date2")

            return df

        def print_r_de_williams(self, n_days=14):
            self.get_r_de_williams(n_days).select("Date",
                                                  "R_de_williams").show()

        def plot_r_de_williams(self, n_days=30):
            df = self.get_r_de_williams(n_days).select("Date", "R_de_williams"
                                                      ).toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.axhline(20, color='r')
            plt.axhline(50, color='g')
            plt.axhline(80, color='r')
            plt.show()

        def get_momentum_roc(self, n_days=12, is_momentum=True, df=None):
            if df is None:
                df = self.stock.df

            SECS_IN_DAY = 86400
            period = n_days * SECS_IN_DAY

            df = df.withColumn("Date2", df.Date.cast("timestamp"))

            w = Window().partitionBy(lit("Date2"))\
                    .orderBy(col("Date2").cast("long"))\
                    .rangeBetween(-period, 0)

            df = df.withColumn("n_close", func.first("Close").over(w))
            if is_momentum:
                df = df.withColumn("momentum", df["Close"] -
                                   df["n_close"]).drop("n_close", "Date2")
            else:
                df = df.withColumn("roc", df["Close"] /
                                   df["n_close"]).drop("n_close", "Date2")

            return df

        def print_momentum(self, n_days=12):
            self.get_momentum_roc(n_days).select("Date", "momentum").show()

        def plot_momentum(self, n_days=30):
            df = self.get_momentum_roc(n_days).select("Date",
                                                      "momentum").toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.show()

        def print_roc(self, n_days=25):
            self.get_momentum_roc(n_days,
                                  is_momentum=False).select("Date",
                                                            "roc").show()

        def plot_roc(self, n_days=30):
            df = self.get_momentum_roc(n_days, is_momentum=False).select(
                "Date", "roc"
            ).toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.show()

        def get_cci(self, n_days=14, df=None):
            if df is None:
                df = self.stock.df

            SECS_IN_DAY = 86400
            period = n_days * SECS_IN_DAY

            df = df.withColumn("Date2", df.Date.cast("timestamp"))

            w = Window().partitionBy(lit("Date2"))\
                    .orderBy(col("Date2").cast("long"))\
                    .rangeBetween(-period, 0)

            df = df.withColumn("TP", (df["High"] + df["low"] + df["Close"]) / 3)
            df = df.withColumn("SMATP", func.avg("TP").over(w))

            df = df.withColumn("D", func.abs(df["TP"] - df["SMATP"]))
            df = df.withColumn("M", func.avg("D").over(w) * 0.015)

            df = df.withColumn("CCI", (df["TP"] - df["SMATP"]) / df["M"])

            return df

        def print_cci(self, n_days=14):
            self.get_cci(n_days).select("Date", "CCI").show()

        def plot_cci(self, n_days=14):
            df = self.get_cci(n_days).select("Date", "CCI").toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.axhline(-100, color='r')
            plt.axhline(0, color='g')
            plt.axhline(100, color='r')
            plt.show()

        def get_gc(self, n_days=14, df=None):
            if df is None:
                df = self.stock.df

            SECS_IN_DAY = 86400
            period = n_days * SECS_IN_DAY

            df = df.withColumn("Date2", df.Date.cast("timestamp"))

            w = Window().partitionBy(lit("Date2"))\
                    .orderBy(col("Date2").cast("long"))\
                    .rangeBetween(-period, 0)

            df = df.withColumn("F", (func.count("Date").over(w) + 1) / 2)
            df = df.withColumn("A", func.dense_rank().over(w) * df["Close"])
            df = df.withColumn("GC", df["F"] - (df["A"] / df["Close"]))

            return df

        def print_gc(self, n_days=14):
            self.get_gc(n_days).select("Date", "GC").show()

        def plot_gc(self, n_days=14):
            df = self.get_gc(n_days).select("Date", "GC").toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.show()

        def get_bb(self, n_days = 20, df = None):
            if df is None:
                df = self.stock.df

            SECS_IN_DAY = 86400
            period = n_days*SECS_IN_DAY

            df = df.withColumn("Date2", df.Date.cast("timestamp"))

            w = Window().partitionBy(lit("Date2"))\
                    .orderBy(col("Date2").cast("long"))\
                    .rangeBetween(-period, 0)

            df = df.withColumn("BB_MM", func.avg("Close").over(w))
            df = df.withColumn("BB_std", func.stddev("Close").over(w))
            df = df.withColumn("BB_UB", df["BB_MM"] + 2 * df["BB_std"])
            df = df.withColumn("BB_LB", df["BB_MM"] - 2 * df["BB_std"])

            return df

        def print_bb(self, n_days = 20):
            self.get_bb(n_days).select("Date", "Close", "BB_MM", "BB_UB", "BB_LB").show()


        def plot_bb(self, n_days = 20):
            df = self.get_bb(n_days).select("Date", "Close", "BB_MM", "BB_UB", "BB_LB").toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.show()

        def get_kb(self, n_days = 20, df = None):
            if df is None:
                df = self.stock.df

            SECS_IN_DAY = 86400
            period = n_days*SECS_IN_DAY

            df = df.withColumn("Date2", df.Date.cast("timestamp"))

            w = Window().partitionBy(lit("Date2"))\
                    .orderBy(col("Date2").cast("long"))\
                    .rangeBetween(-period, 0)

            my_window = Window.partitionBy().orderBy("Date")

            df = df.withColumn("KB_MM", func.avg("Close").over(w))
            df = df.withColumn("KB_LC", lag("Close").over(my_window))
            df = df.withColumn("KB_TR", max(abs(DF["High"] - DF["Low"]), abs(DF["High"] - DF["KB_LC"]), abs(DF["Low"] - DF["KB_LC"])))
            df = df.withColumn("KB_ATR", func.avg("KB_TR").over(w))
            df = df.withColumn("KB_UB", df["KB_MM"] + 2 * df["KB_ATR"])
            df = df.withColumn("KB_LB", df["KB_MM"] - 2 * df["KB_ATR"])

            return df

        def print_kb(self, n_days = 20):
            self.get_bb(n_days).select("Date", "Close", "KB_MM", "KB_UB", "KB_LB").show()


        def plot_kb(self, n_days = 20):
            df = self.get_bb(n_days).select("Date", "Close", "KB_MM", "KB_UB", "KB_LB").toPandas()
            df.set_index("Date", inplace=True)

            fig = plt.figure(figsize=(30, 10))
            df.plot()
            plt.show()


    class Predict:
        def __init__(self, stock):
            # save attributs
            self.stock = stock

            self.fullDF = None
            self.trainDF = None
            self.testDF = None
            self.predDF = None

            self.col_to_pred = None

        def add_next_day(self, col_name, df=None):
            # add the value of the next day in the column "next_" + col_name
            my_window = Window.partitionBy().orderBy("Date")

            if df is None:
                df = self.fullDF

            df = df.withColumn(
                "next_" + col_name,
                lead(col_name).over(my_window)
            )

            return df

        def remove_unused_col(self, df=None):
            # remove useless columns
            if df is None:
                df = self.fullDF

            df = df.drop("company_name")

            return df

        def add_insights(self, df=None):
            # add insights
            if df is None:
                df = self.fullDF

            # add cci
            df = self.stock.insight.get_cci(
                df=df
            ).drop("Date2", "TP", "SMATP", "D", "M")
            # add roc
            df = self.stock.insight.get_momentum_roc(is_momentum=False,
                                                     df=df).drop(
                                                         "Date2", "n_close"
                                                     )
            # add momentum
            df = self.stock.insight.get_momentum_roc(is_momentum=True,
                                                     df=df).drop(
                                                         "Date2", "n_close"
                                                     )
            # add r_de_williams
            df = self.stock.insight.get_r_de_williams(
                df=df
            ).drop("Date2", "PBn", "PHn")

            return df

        def add_analysis(self, df=None):
            # add analysis infos
            if df is None:
                df = self.fullDF

            # add moving_average
            df = self.stock.analysis.moving_average(df=df).drop("Date2")

            # add daily_return
            daily_return = self.stock.analysis.get_daily_return().drop(
                "Open_mean", "Close_mean"
            ).withColumnRenamed("Date", "Date2")
            df = df.join(
                daily_return, df.Date == daily_return.Date2, how='inner'
            ).drop("Date2").orderBy("Date")

            return df

        def load_insights(self, col_to_pred="Close"):
            self.fullDF = self.stock.df

            # self.col_to_pred = col_to_pred

            # add infos
            self.fullDF = self.remove_unused_col()
            self.fullDF = self.add_next_day(col_to_pred)
            self.fullDF = self.add_insights()
            self.fullDF = self.add_analysis()

            # remove row with null
            self.fullDF = self.fullDF.na.drop()
