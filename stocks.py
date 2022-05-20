from stock import Stock

import glob

class Stocks(object):
    def __init__(self, files=None, schema=None): 
        self.dfs = self._load_dfs(files, schema)

    def _load_dfs(self, files, schema):
        if not files:
            files = glob.glob("stocks_data/*.csv")

        dfs = []
        for file in files:
            dfs.append(Stock(file, header=True, delimiter=',', schema=schema))
        
        return dfs
