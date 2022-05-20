from stock import Stock

import glob

class Stocks(object):
    def __init__(self, files=None, schema=None): 
        self.stocks = self._load_stocks(files, schema)

    def _load_stocks(self, files, schema):
        if not files:
            files = glob.glob("stocks_data/*.csv")

        dfs = []
        for file in files:
            dfs.append(Stock(file, header=True, delimiter=',', schema=schema))
        
        return dfs

    def get_max_daily_return(self):
        max_daily_returns = {}
        for stock in self.stocks:
            cmpny_name = stock.df.select("company_name").first()[0]
            max_daily_returns[cmpny_name] = (stock.analysis.get_daily_return_max())

        return dict(sorted(max_daily_returns.items(), key=lambda item: item[1], reverse=True))

