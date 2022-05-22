from stock import Stock

import glob

class Stocks(object):
    def __init__(self, files=None, header=False, delimiter=";", schema=None):
        
        # load and save stocks
        self.stocks = self._load_stocks(files, header, delimiter, schema)

    def _load_stocks(self, files, header, delimiter, schema):
        # load stocks contained in files
        
        # use all the csv files in stocks_data/ if files argument not provided
        if not files:
            files = glob.glob("stocks_data/*.csv")

        dfs = []
        for file in files:
            # call the Stock class to load each stock/file
            dfs.append(Stock(file, header=header, delimiter=delimiter,
                             schema=schema))
        
        return dfs

    def get_max_daily_return(self):
        # get the maximum daily return for each stocks
        max_daily_returns = {}
        for stock in self.stocks:
            cmpny_name = stock.df.select("company_name").first()[0]
            max_daily_returns[cmpny_name] = (stock.analysis.get_daily_return_max())

        return dict(sorted(max_daily_returns.items(), key=lambda item: item[1], reverse=True))
    
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
