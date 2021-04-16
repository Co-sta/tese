import pandas as pd
import math
from copy import deepcopy

MAX_DIS_TIME = pd.to_timedelta('90 days')  # max distance from current date to option expiration in order do buy
MIN_DIS_TIME = pd.to_timedelta('40 days')  # min distance from current date to option expiration in order to buy
TYPE_STRIKE = 1 # 0:out of the money | 1:in the money
PRICE_RANGE = [10, 150]
CAPITAL_MODE = 0 # 1:max capital | 0:unlimited capital

trading_dic = {
1: {'tra_type': 'long',
    'opt_type': 'call',
    'action': {1: 'open',
               0: 'nothing',
              -1: 'close'}},
2: {'tra_type': 'long',
    'opt_type': 'put',
    'action': {1: 'open',
               0: 'nothing',
              -1: 'close'}},
3: {'tra_type': 'short',
    'opt_type': 'call',
    'action': {1: 'close',
               0: 'nothing',
              -1: 'open'}},
4: {'tra_type': 'short',
    'opt_type': 'put',
    'action': {1: 'close',
               0: 'nothing',
              -1: 'open'}},
}


def trade(eval_start, eval_end, orders, tickers, type):
    global CASE_STUDY
    CASE_STUDY = type

    port = Portfolio(eval_start, eval_end, tickers)
    while port.new_day():
        print(port.get_current_date())
        daily_orders = orders[port.get_current_date()].copy()
        for ticker in port.get_tickers():
            action = daily_orders.at[ticker]
            position = trading_dic[CASE_STUDY]['action'][action]
            transation_type = trading_dic[CASE_STUDY]['tra_type']
            option_type = trading_dic[CASE_STUDY]['opt_type']
            if port.close_VIX():
                port.close_all_positions(trading_dic[CASE_STUDY]['tra_type'])
            else:
                if position == 'open':
                    root = port.search_root(ticker, option_type)
                    if root:
                        if control_xema(ticker, root, action, port.get_current_date()):
                            continue
                        n_options = port.get_n_options(root)[0]
                        port.open_position(transation_type, root, n_options)
                elif position == 'close':
                    port.close_position(transation_type, ticker=ticker)
        port.clean_portfolio()
        port.update_holdings()
        port.update_ROI()
        port.update_CAPITAL()
    return port

class Transaction:

    def __init__(self, date, root, ty, value, quantity):
        self.root = root
        self.type = ty
        self.quantity = quantity
        self.init_value = value
        self.init_date = date
        self.final_value = -1
        self.final_date = -1
        self.result = -1

    def get_root(self):
        return self.root
    def get_type(self):
        return self.type
    def get_quantity(self):
        return self.quantity
    def get_init_value(self):
        return self.init_value
    def get_init_date(self):
        return self.init_date
    def get_final_value(self):
        return self.final_value
    def get_final_date(self):
        return self.final_date
    def get_result(self):
        return self.result
    def set_final_value(self, value):
        self.final_value = value
    def set_final_date(self, date):
        self.final_date = date
    def set_root_init_value_quantity(self, new_root, new_init_value, new_quantity):
        self.root = new_root
        self.init_value = new_init_value
        self.quantity = new_quantity
    def check_result(self):
        profit = self.get_final_value()-self.get_init_value()
        if (self.get_type() == 'long' and profit > 0) or (self.get_type() == 'short' and profit < 0):
            self.result = 'positive'
            return 'positive'
        else:
            self.result = 'negative'
            return 'negative'


class Option:

    def __init__(self, root, quantity=1):
        self.root = root
        self.strike = int(root[-8::]) / 1000
        self.type = root[-9:-8]
        self.year = root[-15:-13]
        self.month = root[-13:-11]
        self.day = root[-11:-9]
        self.expiration_date = pd.to_datetime('/'.join([self.month, self.day, self.year]))
        self.company = root[0:-15]
        self.quantity = quantity

    def get_root(self):
        return self.root
    def get_strike(self):
        return self.strike
    def get_type(self):
        return self.type
    def get_expiration_date(self):
        return self.expiration_date
    def get_company(self):
        return self.company
    def get_quantity(self):
        return self.quantity
    def add_quantity(self, quantity):
        self.quantity += quantity
    def set_root_strike_nr(self, new_root, new_strike, new_quantity):
        self.root = new_root
        self.strike = new_strike
        self.quantity = new_quantity


class Portfolio:

    def __init__(self, start_date, end_date, tickers, initial_capital=0):
        self.max_position = 100000  # o valor máximo por posição é de 100000
        self.current_capital = initial_capital
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.filenames = open('data/Options/option_dataset_filenames.txt').readlines()
        self.dataset = self.get_option_dataset(self.get_start_date())[0]
        self.current_date = self.get_start_date()
        self.portfolio = {}
        self.to_clean_portfolio = []
        self.holdings = pd.DataFrame({'net_value': initial_capital, 'capital': initial_capital}, index=[start_date])

        self.tickers = tickers
        self.log = {self.current_date: []}
        roi_dic = { i : 0 for i in tickers }
        roi_dic['total'] = 0
        self.ROI = pd.DataFrame(roi_dic, index=[start_date])
        self.CAPITAL = pd.DataFrame(roi_dic, index=[start_date])
        self.stock_splits = pd.read_csv('data/stock_splits/stock_splits.csv', parse_dates=True)

        self.nr_pos_trades = 0
        self.nr_neg_trades = 0
        self.VIX = pd.read_csv('data/VIX/VIX30D.csv', index_col='Date', parse_dates=True)


    #  GETTERS  #
    def get_max_position(self):
        return self.max_position
    def get_current_capital(self):
        return self.current_capital
    def get_initial_capital(self):
        return self.initial_capital
    def get_start_date(self):
        return self.start_date
    def get_end_date(self):
        return self.end_date
    def get_filenames(self):
        return self.filenames
    def get_dataset(self):
        return self.dataset
    def get_current_date(self):
        return self.current_date
    def get_portfolio(self):
        return self.portfolio
    def get_to_clean_portfolio(self):
        return self.to_clean_portfolio
    def get_holdings(self):
        return self.holdings
    def get_log(self):
        return self.log
    def get_ROI(self):
        return self.ROI
    def get_CAPITAL(self):
        return self.CAPITAL
    def get_stock_splits(self):
        return self.stock_splits
    def get_nr_pos_trades(self):
        return self.nr_pos_trades
    def get_nr_neg_trades(self):
        return self.nr_neg_trades
    def get_VIX(self):
        return self.VIX
    def get_tickers(self):
        return self.tickers

    # GENERAL
    def new_day(self):
        current_date = self.get_current_date()
        while True:
            current_date = current_date + pd.to_timedelta(1, unit='d')
            [dataset, date_exists] = self.get_option_dataset(current_date)
            if date_exists:
                self.current_date = current_date
                self.dataset = dataset
                self.log[current_date] = []
                self.check_stock_split()
                self.holdings_new_day()
                self.verify_mature_options()
                if current_date >= self.get_end_date():  # sees if it's already in the final day
                    self.close_all_positions(trading_dic[CASE_STUDY]['tra_type'])
                    self.update_holdings()
                    self.update_ROI()
                    self.update_CAPITAL()
                    self.evaluate_trades()
                    return 0

                else:
                    return 1

    def search_root(self, ticker, option_type):
        if ticker not in self.get_dataset().UnderlyingSymbol.values:
            return False

        underlying_price = \
            self.get_dataset().loc[self.get_dataset()['UnderlyingSymbol'] == ticker]['UnderlyingPrice'].iloc[-1]

        # print(underlying_price)
        t1 = self.get_current_date() + MIN_DIS_TIME
        t2 = self.get_current_date() + MAX_DIS_TIME

        options_roots = self.get_dataset().loc[(self.get_dataset()['UnderlyingSymbol'] == ticker) &
                                         (self.get_dataset()['Type'] == option_type) &
                                         (t1 < self.get_dataset()['Expiration']) &
                                         (self.get_dataset()['Expiration'] < t2) &
                                         (self.get_dataset()['Ask'] >= PRICE_RANGE[0]) &
                                         (self.get_dataset()['Ask'] <= PRICE_RANGE[1])
                                         ]['OptionRoot'].values

        options_strikes = self.get_dataset().loc[(self.get_dataset()['UnderlyingSymbol'] == ticker) &
                                           (self.get_dataset()['Type'] == option_type) &
                                           (t1 < self.get_dataset()['Expiration']) &
                                           (self.get_dataset()['Expiration'] < t2) &
                                           (self.get_dataset()['Ask'] >= PRICE_RANGE[0]) &
                                           (self.get_dataset()['Ask'] <= PRICE_RANGE[1])
                                           ]['Strike'].values
        for i in range(len(options_strikes)):
            if option_type == 'put':
                if TYPE_STRIKE:  # 1-in the money
                    if options_strikes[i] > underlying_price:
                        return options_roots[i]
                else:  # 0-out of the money
                    if options_strikes[i] > underlying_price:
                        return options_roots[i - 1]
            else:
                if TYPE_STRIKE:  # 1-in the money
                    if options_strikes[i] > underlying_price:
                        return options_roots[i - 1]
                else:  # 0-out of the money
                    if options_strikes[i] > underlying_price:
                        return options_roots[i]
        # in case it doesnt find an option
        return False

    def verify_mature_options(self):
        current_date = self.get_current_date()
        for option in self.portfolio.values():
            if option.get_expiration_date() <= current_date + pd.to_timedelta(MIN_DIS_TIME):
                self.close_position(trading_dic[CASE_STUDY]['tra_type'], root=option.get_root())

    def get_option_dataset(self, date): # TODO VOLUE = 0?
        str_date = date.strftime('%Y%m%d')
        option_dataset = []
        for filename in self.filenames:
            if str_date in filename:
                option_dataset = pd.read_csv(filename.rstrip('\n'))  # rstrip removes \n from the end of string
                # TODO VERIFICAR SE É PRECISO VOLTAR RETIRAR OS VOLUMES = 0
                # option_dataset = option_dataset.drop(option_dataset[option_dataset.Volume == 0].index)  # drops from
                # data all rows with Volume = 0
                option_dataset['Expiration'] = pd.to_datetime(option_dataset['Expiration'])
                return option_dataset, 1
        return option_dataset, 0

    def close_transations(self, root, price):
        for daily_transactions in self.get_log().values():
            for transaction in daily_transactions:
                if transaction.get_root() == root:
                    print('7777777777777')
                    print(price)
                    print(self.get_current_date())
                    print('7777777777777')
                    transaction.set_final_value(price)
                    transaction.set_final_date(self.get_current_date())

    def update_holdings(self):
        capital = self.holdings.at[self.current_date, "capital"]
        options_value = 0

        for option in self.portfolio.values():
            value = self.get_dataset().loc[self.get_dataset()['OptionRoot'] == option.get_root()].iloc[0]['Ask'] * \
                    option.get_quantity()
            options_value += value

        if CASE_STUDY == 1 or CASE_STUDY == 2: # long
            self.holdings.at[self.current_date, 'net_value'] = capital + options_value
        elif CASE_STUDY == 3 or CASE_STUDY == 4: # short
            self.holdings.at[self.current_date, 'net_value'] = capital - options_value

    def check_stock_split(self):
        stock_splits = self.get_stock_splits()
        portfolio_copy = deepcopy(self.get_portfolio())
        new_entry = {}
        date = self.get_current_date()
        stock_splits['date'] = pd.to_datetime(stock_splits['date'])
        if date in stock_splits['date'].values:
            # rectifies portfolio
            print('1111111')
            print(self.get_portfolio().keys())
            print('1111111')
            for key in self.get_portfolio().keys():
                option = self.get_portfolio()[key]
                for i in stock_splits.index:
                    if option.get_company() == stock_splits.at[i, 'ticker'] and date == stock_splits.at[i, 'date']:
                        ratio = stock_splits.loc[(stock_splits['date']==date) &
                                                 (stock_splits['ticker']==option.get_company()),
                                                 'ratio'].values[0]
                        [ratio_u, ratio_d] = ratio.split('/')
                        ratio_u = int(ratio_u)
                        ratio_d = int(ratio_d)
                        new_quantity = (option.get_quantity() * ratio_u) / ratio_d
                        root_strike = '00000000' + str(int(round((option.strike * ratio_d) / ratio_u, 2) * 1000))
                        root_strike = root_strike[-8::]
                        new_root = option.company + option.year + option.month + option.day + option.type + root_strike
                        new_strike = int(root_strike) / 1000
                        option.set_root_strike_nr(new_root, new_strike, new_quantity)
                        new_entry[new_root] = option
                        portfolio_copy.pop(key)
            new_entry.update(portfolio_copy)
            self.portfolio = deepcopy(new_entry)
            print('44444444444')
            print(self.get_portfolio().keys())
            print('44444444444')

            # rectifies transactions
            for daily_transactions in self.get_log().values():
                for transaction in daily_transactions:
                    if transaction.get_final_value() == -1:
                        for i in stock_splits.index:
                            option = Option(transaction.get_root())
                            if option.get_company() == stock_splits.at[i, 'ticker'] and date == stock_splits.at[i, 'date']:
                                print(option.get_company())
                                ratio = stock_splits.loc[(stock_splits['date']==date) &
                                                         (stock_splits['ticker']==option.get_company()),
                                                         'ratio'].values[0]
                                [ratio_u, ratio_d] = ratio.split('/')
                                ratio_u = int(ratio_u)
                                ratio_d = int(ratio_d)
                                new_quantity = (transaction.get_quantity() * ratio_u) / ratio_d
                                root_strike = '00000000' + str(int(round((option.strike * ratio_d) / ratio_u, 2) * 1000))
                                root_strike = root_strike[-8::]
                                new_root = option.company + option.year + option.month + option.day + option.type + root_strike
                                new_init_value = (transaction.get_init_value() * ratio_d) / ratio_u
                                transaction.set_root_init_value_quantity(new_root, new_init_value, new_quantity)

    def update_ROI(self):
        roi = self.get_ROI()
        old_data = pd.DataFrame(roi[-1:].values,
                                     index=[self.current_date],
                                     columns=roi.columns)
        roi = roi.append(old_data)
        for daily_transactions in self.get_log().values():
            for transaction in daily_transactions:
                if transaction.get_final_date() == self.get_current_date():
                    ticker = Option(transaction.get_root()).get_company()
                    if transaction.get_type() == 'short':
                        profit = transaction.get_init_value() - transaction.get_final_value()
                        investment = transaction.get_final_value()
                    else: #transaction.type == long
                        profit = transaction.get_final_value() - transaction.get_init_value()
                        investment = transaction.get_init_value()
                    roi.at[self.current_date, ticker] += (profit/investment)*transaction.get_quantity()
                    roi.at[self.current_date, 'total'] += (profit/investment)*transaction.get_quantity()

        self.ROI = roi

    def update_CAPITAL(self):
        print('UPDATE CAPITAL init')
        capital = self.get_CAPITAL()
        old_data = pd.DataFrame(capital[-1:].values,
                                     index=[self.current_date],
                                     columns=capital.columns)
        capital = capital.append(old_data)
        for daily_transactions in self.get_log().values():
            for transaction in daily_transactions:
                if transaction.get_final_date() == self.get_current_date():
                    ticker = Option(transaction.get_root()).get_company()
                    print(ticker)
                    print(transaction.get_type())
                    print(transaction.get_init_value())
                    print(transaction.get_final_value())
                    print()
                    if transaction.get_type() == 'short':
                        profit = transaction.get_init_value() - transaction.get_final_value()
                        print(profit)
                    else: #transaction.type == long
                        profit = transaction.get_final_value() - transaction.get_init_value()
                        print(profit)
                    capital.at[self.current_date, ticker] += profit*transaction.get_quantity()
                    capital.at[self.current_date, 'total'] += profit*transaction.get_quantity()

        self.CAPITAL = capital
        print('UPDATE CAPITAL end')

    def close_VIX(self):
        vix = self.get_VIX().at[self.current_date, 'close']
        if vix >= 20:
            return True
        else:
            return False

    def evaluate_trades(self):
        for daily_transactions in self.get_log().values():
            for txn in daily_transactions:
                result = txn.check_result()
                if result == 'positive':
                    self.nr_pos_trades += 1
                elif result == 'negative':
                    self.nr_neg_trades += 1
                else:
                    raise 'log with invalid result value'

    def clean_portfolio(self):
        to_clean = self.get_to_clean_portfolio()
        for root in to_clean:
            self.portfolio.pop(root)
        self.to_clean_portfolio = []

    def holdings_new_day(self):
        self.holdings.at[self.current_date] = [self.holdings.iloc[len(self.holdings) - 1].net_value,
                                               self.holdings.iloc[len(self.holdings) - 1].capital]

    def get_n_options(self, root):
        option_price = self.get_dataset().loc[self.get_dataset()['OptionRoot'] == root].iloc[0]['Ask']
        if option_price > self.current_capital and CAPITAL_MODE:
            # print('Not enough money to buy this option: ' + root)
            return 0, option_price
        n_options = int(float(self.get_max_position()) / float(option_price))
        return n_options, option_price

    def open_position(self, type, root, n_options=0):
        print('opening... ' + type + ': ' + root)
        [_n_options, option_price] = self.get_n_options(root)
        if not n_options:
            n_options = _n_options
        if n_options:
            if type == 'long':
                self.current_capital -= option_price * n_options
            elif type == 'short':
                self.current_capital += option_price * n_options
            print('----------OPEN POSITION UPDATED')
            print(self.holdings.at[self.current_date, 'capital'])
            self.holdings.at[self.current_date, 'capital'] = self.current_capital  # updates the capital
            print(self.holdings.at[self.current_date, 'capital'])
            print('----------OPEN POSITION UPDATED')
            self.log[self.current_date].append(Transaction(self.current_date, root, type, option_price, n_options))
            # adds the transaction to logs
            if root in self.portfolio:
                # updates the number of options of a company
                self.portfolio[root].add_quantity(n_options)
            else:
                # adds the option to the portfolio
                self.portfolio[root] = Option(root, n_options)

    def close_position(self, type, root=0, ticker=0):
        print('55555555555555555555')
        print(root)
        to_close = []
        if root: # closes position of a specific root
            if root not in self.to_clean_portfolio:
                to_close.append(root)

        else: # closes all positions from that company
            for option in self.get_portfolio().values():
                if option.get_company() == ticker:
                    root = option.get_root()
                    if root not in self.to_clean_portfolio:
                        to_close.append(root)
        print(to_close)
        print('portfolio:::::::::::::')
        print(self.portfolio.keys())
        print('55555555555555555555')
        for root in to_close:
            print('closing... ' + type + ': ' + root)
            print(self.get_dataset().loc[self.get_dataset()['OptionRoot'] == root])
            option_price = self.get_dataset().loc[self.get_dataset()['OptionRoot'] == root].iloc[0]['Ask']
            n_options = self.portfolio[root].get_quantity()
            if type == 'long':
                print('long')
                self.current_capital += option_price * n_options
            elif type == 'short':
                print('short')
                print(option_price * n_options)
                self.current_capital -= option_price * n_options
            print('----------CLOSE POSITION UPDATED')
            print(self.holdings.at[self.current_date, 'capital'])
            self.holdings.at[self.current_date, 'capital'] = self.current_capital
            print(self.holdings.at[self.current_date, 'capital'])
            print('----------CLOSE POSITION UPDATED')
            self.close_transations(root, option_price)
            self.to_clean_portfolio.append(root)

    def close_all_positions(self, type):
        for root in self.portfolio.keys():
            self.close_position(type, root=root)

def control_xema(ticker, root, action, date):
    filepath = 'data/options_xma/' + ticker + '.csv'
    option_dataset = pd.read_csv(filepath, index_col='Date', parse_dates=['Date'], usecols=[root, 'Date'])
    option_action = option_dataset.at[date, root]
    print(ticker)
    print(action)
    print(option_action)
    print((action * option_action) != 1)
    return (action * option_action) != 1
