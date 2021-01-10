import pandas as pd
import math

MAX_DIS_TIME = pd.to_timedelta('90 days')  # max distance from current date to option expiration in order do buy
MIN_DIS_TIME = pd.to_timedelta('40 days')  # min distance from current date to option expiration in order to buy
TYPE_STRIKE = 1 # 0:out of the money | 1:in the money
PRICE_RANGE = [30, 80]
TRADING_TYPE = 1 # 1:buy(calls/puts) | 2:sell(puts) | 3:sell(calls)

def trade(eval_start, eval_end, orders): # TODO REVIEW
    port = Portfolio(eval_start, eval_end)
    while port.new_day():
        daily_orders = orders[port.current_date].copy()
        for ticker in daily_orders.index:
            action = daily_orders.at[ticker]  # 1:comprar, 0:nada, -1:vender
            if TRADING_TYPE == 1: # buy(calls/puts)
                if action == 1:
                    put_root = port.search_root(ticker, 'put')
                    call_root = port.search_root(ticker, 'call')
                    if put_root and call_root:
                        n_options = min(port.get_n_options(put_root)[0], port.get_n_options(call_root)[0])
                        if n_options:
                            port.buy_options(put_root, n_options)  # buys a put and a call so that the return is not influenced
                            port.buy_options(call_root, n_options)  # by the variation of the underlying price
                elif action == -1:
                    port.sell_exercise_options(ticker=ticker)
            elif TRADING_TYPE == 2: # sell(puts)
                if action == 1 or close_VIX(): # CONTINUAR AQUI
                    port.buyback_options(ticker=ticker)
                elif action == -1:
                    put_root = port.search_root(ticker, 'put')
                    port.create_options(put_root)
            elif TRADING_TYPE == 3: # sell(calls)
                pass
        port.clean_portfolio()
        port.update_holdings()
        port.update_ROI()
    return port


class Transaction:

    def __init__(self, date, root, ty, value, quantity):
        self.date = date
        self.root = root
        self.type = ty
        self.init_value = value
        self.quantity = quantity
        self.final_value = -1

    def get_date(self):
        return self.date
    def get_root(self):
        return self.root
    def get_type(self):
        return self.type
    def get_init_value(self):
        return self.init_value
    def get_quantity(self):
        return self.quantity
    def get_final_value(self):
        return self.final_value
    def set_final_value(self, value):
        self.final_value = value
    def set_root_init_value_quantity(self, new_root, new_init_value, new_quantity):
        self.root = new_root
        self.init_value = new_init_value
        self.quantity = new_quantity


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

    def __init__(self, start_date, end_date, initial_capital=pow(1000000, 1000000)):
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

        self.log = {self.current_date: []}
        self.ROI = pd.DataFrame(data={'total': 0}, index=[start_date])
        self.stock_splits = pd.read_csv('data/stock_splits/stock_splits.csv', parse_dates=True)

        self.nr_pos_trades = {}
        self.nr_neg_trades = {}
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
    def empty_to_clean(self):
        self.to_clean_portfolio = []
    def get_holdings(self):
        return self.holdings
    def get_ROI(self):
        return self.ROI
    def get_stock_splits(self):
        return self.stock_splits
    def get_nr_pos_trades(self):
        return self.nr_pos_trades
    def get_nr_neg_trades(self):
        return self.nr_neg_trades
    def get_VIX(self):
        return self.VIX

    # GENERAL
    def new_day(self): # TODO REVIEW TYPE 3
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
                    if TRADING_TYPE == 1: # buy(calls/puts)
                        self.sell_exercise_all_options()
                    elif TRADING_TYPE == 2: # sell(puts)
                        self.buyback_all_options()
                    elif TRADING_TYPE == 3: # sell(calls)
                        pass
                    self.update_holdings()
                    self.update_ROI()
                    return 0

                else:
                    return 1

    def search_root(self, ticker, option_type):
        underlying_price = \
            self.dataset.loc[self.dataset['UnderlyingSymbol'] == ticker]['UnderlyingPrice'].iloc[-1]

        # print(underlying_price)
        t1 = self.get_current_date() + MIN_DIS_TIME
        t2 = self.get_current_date() + MAX_DIS_TIME

        options_roots = self.dataset.loc[(self.dataset['UnderlyingSymbol'] == ticker) &
                                         (self.dataset['Type'] == option_type) &
                                         (t1 < self.dataset['Expiration']) &
                                         (self.dataset['Expiration'] < t2) &
                                         (self.dataset['Ask'] >= PRICE_RANGE[0]) &
                                         (self.dataset['Ask'] <= PRICE_RANGE[1])
                                         ]['OptionRoot'].values

        options_strikes = self.dataset.loc[(self.dataset['UnderlyingSymbol'] == ticker) &
                                           (self.dataset['Type'] == option_type) &
                                           (t1 < self.dataset['Expiration']) &
                                           (self.dataset['Expiration'] < t2) &
                                           (self.dataset['Ask'] >= PRICE_RANGE[0]) &
                                           (self.dataset['Ask'] <= PRICE_RANGE[1])
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

    def verify_mature_options(self): # TODO REVIEW TYPE 3
        current_date = self.get_current_date()
        for option in self.portfolio.values():
            if TRADING_TYPE == 1: # buy(calls/puts)
                if option.get_expiration_date() <= current_date + pd.to_timedelta('1 day'):
                    self.sell_exercise_options(root=option.get_root())
            elif TRADING_TYPE == 2: # sell(puts)
                if option.get_expiration_date() <= current_date + pd.to_timedelta(MIN_DIS_TIME):
                    self.buyback_options(root=option.get_root())
            elif TRADING_TYPE == 3: # sell(calls)
                pass

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
        for daily_transactions in self.log.values():
            for transaction in daily_transactions:
                if transaction.get_root() == root:
                    transaction.set_final_value(value)

    def update_holdings(self): # TODO REVIEW TYPE 3
        capital = self.holdings.at[self.current_date, "capital"]
        options_value = 0

        for option in self.portfolio.values():
            value = self.dataset.loc[self.dataset['OptionRoot'] == option.get_root()].iloc[0]['Ask'] * \
                    option.get_quantity()
            options_value += value

        if TRADING_TYPE == 1: # buy(calls/puts)
            self.holdings.at[self.current_date, 'net_value'] = capital + options_value
        elif TRADING_TYPE == 2: # sell(puts)
            self.holdings.at[self.current_date, 'net_value'] = capital - options_value
        elif TRADING_TYPE == 3: # sell(calls)
            pass

    def check_stock_split(self):
        stock_splits = self.get_stock_splits()
        date = self.get_current_date()
        if date in stock_splits['date'].tolist():
            # rectifies portfolio
            for option in self.get_portfolio().values():
                for ticker in stock_splits.at[date, 'ticker']:
                    if option.get_company() == ticker:
                        ratio = stock_splits.loc[(stock_splits['date']==date) &
                                                 (stock_splits['ticker']==ticker),
                                                 'ratio'].values[0]
                        ratio = stock_splits.at[date, 'ratio']
                        [ratio_u, ratio_d] = ratio.split('/')
                        ratio_u = int(ratio_u)
                        ratio_d = int(ratio_d)
                        new_quantity = (option.get_quantity() * ratio_u) / ratio_d
                        root_strike = '00000000' + str(int(round((option.strike * ratio_d) / ratio_u, 2) * 1000))
                        root_strike = root_strike[-8::]
                        new_root = option.company + option.year + option.month + option.day + option.type + root_strike
                        new_strike = int(root_strike) / 1000
                        option.set_root_strike_nr(new_root, new_strike, new_quantity)
            # rectifies transactions
            for daily_transactions in self.log().values():
                for transaction in daily_transactions:
                    if transaction.get_final_value() != -1:
                        for ticker in stock_splits.at[date, 'ticker']:
                            option = Option(transaction.get_root())
                            if option.get_company() == ticker
                                ratio = stock_splits.loc[(stock_splits['date']==date) &
                                                         (stock_splits['ticker']==ticker),
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
        last_transactions = self.log.values()[-1]
        for transaction in last_transactions:
            if transaction.get_final_value() != -1:
                ticker = Option(transaction.get_root()).get_company()
                if transaction.get_type() == 'sell':
                    profit = transaction.get_init_value() - transaction.get_final_value()
                    investment = transaction.get_final_value()
                else: #transaction.type == buy
                    profit = transaction.get_final_value() - transaction.get_init_value()
                    investment = transaction.get_init_value()
                roi.at[self.current_date, ticker] += profit/investment
                roi.at[self.current_date, 'total'] += profit/investment

        self.ROI = roi

    def close_VIX(self):
        vix = self.get_VIX().at[self.current_date, 'close']
        if vix >= 20:
            return True
        else:
            return False


    # TYPE 1 (BUY)
    def holdings_new_day(self): # TODO REVIEW
        self.holdings.at[self.current_date] = [self.holdings.iloc[len(self.holdings) - 1].net_value,
                                               self.holdings.iloc[len(self.holdings) - 1].capital]

    def buy_options(self, root, n_options=0): # TODO REVIEW
        print('buying.... ' + root)
        [_n_options, option_price] = self.get_n_options(root)
        if not n_options:
            n_options = _n_options
        if n_options:
            position = option_price * n_options
            self.current_capital -= position
            self.holdings.at[self.current_date, 'capital'] = self.current_capital  # updates the capital
            self.log[self.current_date].append(Transaction(self.current_date, root, 'buy', option_price, n_options))
            # adds the transaction to logs
            if root in self.portfolio:
                self.portfolio[root].add_quantity(n_options)  # updates the number of options of a company
            else:
                self.portfolio[root] = Option(root, n_options)  # adds the option to the portfolio

    def get_n_options(self, root): # TODO REVIEW
        option_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['Ask']
        if option_price > self.current_capital:
            # print('Not enough money to buy this option: ' + root)
            return 0, option_price
        n_options = int(float(self.get_max_position()) / float(option_price))
        return n_options, option_price

    def sell_options(self, root): # TODO REVIEW
        if root not in self.to_clean_portfolio:
            print('selling.... ' + root)
            option_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['Ask']
            n_options = self.portfolio[root].get_quantity()
            self.current_capital += option_price * n_options
            self.holdings.at[self.current_date, 'capital'] = self.current_capital  # updates the capital
            self.log[self.current_date].append(Transaction(self.current_date, root, 'sell', option_price, n_options))
            self.to_clean_portfolio.append(root)

    def exercise_options(self, root): # TODO REVIEW
        print('exercising.... ' + root)
        stock_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['UnderlyingPrice']
        strike = self.portfolio[root].get_strike()

        if self.portfolio[root].get_type() == 'C':  # call options
            profit = stock_price - strike
        else:  # put options
            profit = strike - stock_price

        n_options = self.portfolio[root].get_quantity()
        self.current_capital += profit * n_options
        self.holdings.at[self.current_date, 'capital'] = self.current_capital  # updates the capital
        self.to_clean_portfolio.append(root)

    def sell_exercise_options(self, ticker=0, root=0): # TODO REVIEW
        if not root:  # sells or exercises all options from that company
            for option in self.portfolio.values():
                if option.get_company() == ticker:
                    root = option.get_root()
                    option_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['Ask']
                    stock_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['UnderlyingPrice']
                    strike = self.portfolio[root].get_strike()

                    if self.portfolio[root].get_type() == 'C':  # call options
                        exercise_profit = stock_price - strike
                    else:  # put options
                        exercise_profit = strike - stock_price

                    if option_price > exercise_profit:
                        self.sell_options(root)
                    else:
                        self.exercise_options(root)
        else:  # sells or exercises a specific option
            print('expiring:    ' + root)  # TODO TIRAR
            # print(self.dataset.loc[self.dataset['OptionRoot'] == root])
            option_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['Ask']
            stock_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['UnderlyingPrice']
            strike = self.portfolio[root].get_strike()

            if self.portfolio[root].get_type() == 'C':  # call options
                exercise_profit = stock_price - strike
            else:  # put options
                exercise_profit = strike - stock_price

            if option_price > exercise_profit:
                self.sell_options(root)
            else:
                self.exercise_options(root)

    def sell_exercise_all_options(self):
        for root in self.portfolio.keys():
            self.sell_exercise_options(root=root)

    def clean_portfolio(self): # TODO REVIEW
        to_clean = self.get_to_clean_portfolio()
        for root in to_clean:
            self.portfolio.pop(root)
        self.empty_to_clean()

    # TYPE 2/3 (SELL)
    def create_options(root, n_options):
        print('creating.... ' + root +' ('+str(n_options)+')')
        option_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['Ask']
        position = option_price * n_options
        self.current_capital += position
        self.holdings.at[self.current_date, 'capital'] = self.current_capital  # updates the capital
        self.log[self.current_date].append(Transaction(self.current_date, root, 'sell', option_price, n_options))
        if root in self.portfolio:
            self.portfolio[root].add_quantity(n_options)  # updates the number of options of a company
        else:
            self.portfolio[root] = Option(root, n_options)  # adds the option to the portfolio

    def buyback_options(self, ticker=0, root=0):
        to_buy_back = []
        if root: # buys back options of a specific root
            if root not in self.to_clean_portfolio:
                to_buy_back.append(root)

        else: # buys back all options from that company
            for option in self.portfolio.values():
                if option.get_company() == ticker:
                    root = option.get_root()
                    if root not in self.to_clean_portfolio:
                        to_buy_back.append(root)

        for root in to_buy_back:
            print('buying back.... ' + root)
            option_price = self.dataset.loc[self.dataset['OptionRoot'] == root].iloc[0]['Ask']
            n_options = self.portfolio[root].get_quantity()
            self.current_capital -= option_price * n_options
            self.holdings.at[self.current_date, 'capital'] = self.current_capital
            self.close_transations(root, option_price)
            self.to_clean_portfolio.append(root)

    def buyback_all_options(self):
        for root in self.portfolio.keys():
            self.buyback_options(root=root)
