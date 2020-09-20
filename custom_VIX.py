import pandas as pd
import numpy as np
from datetime import timedelta
# import statistics as st
import pickle
import os.path


def test():
    v = VIX('SPX')
    fig = 1
    # st.graph_custom_vix(v, title='SPX', fig=fig)
    # for ticker in open_sp500_tickers_to_list():
    #   fig += 1
    #   v = VIX(ticker)
    #   st.graph_custom_vix(v, title=ticker, fig=fig)
    return v


def create_interest_rates(interest_rates_file):
    interest_rates = pd.read_csv(interest_rates_file)
    for idx in interest_rates.index:
        interest_rates.at[idx, "DATE"] = pd.to_datetime(interest_rates.at[idx, 'DATE'])
        if interest_rates.at[idx, 'DGS10'] == '.':
            interest_rates.at[idx, "DGS10"] = interest_rates.at[idx - 1, 'DGS10']
        else:
            interest_rates.at[idx, "DGS10"] = float(interest_rates.at[idx, 'DGS10'])
    return interest_rates


def calculate_F(strike, interest_rate_R, time_maturity_T, call, put):
    F = strike + np.exp(interest_rate_R * time_maturity_T) * (call - put)
    return F


def sqr_volatility(T, midpoint, F, K):
    su = midpoint.sum(axis=0)[2]
    sqr_vol = ((2 / T) * su) - ((1 / T) * ((F / K) - 1) ** 2)
    return sqr_vol


def minutes_to_date(date1, date2):
    t = date2 - date1
    m = t.total_seconds() / 60
    return m


def days_to_date(date1, date2):
    days = (date2 - date1).days
    return days


def time_to_exp_T(date1, date2):
    minutes = minutes_to_date(date1, date2)
    m_year = 525600
    return minutes / m_year


def save_vix_dataset(ds, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(ds, f)


def load_vix_dataset(filepath):
    with open(filepath, 'rb') as f:
        ds = pickle.load(f)
    return ds


def save_vix(ds, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(ds, f)


def load_vix(filepath):
    with open(filepath, 'rb') as f:
        ds = pickle.load(f)
    return ds


def open_sp500_tickers_to_list():
    with open('data/Options/sp500_tickers_all.txt') as f:
        tickers = f.read().splitlines()
    return tickers


class VIX:
    def __init__(self,
                 ticker,
                 start_day='2011-01-03',
                 end_day='2014-01-01',
                 interest_rates_file='data/interest_rate.csv'):

        self.count = 1  # todo tirar
        self.start_day = pd.to_datetime(start_day)
        self.current_day = self.start_day
        self.end_day = pd.to_datetime(end_day)
        self.ticker = ticker
        self.value = pd.DataFrame()
        self.data = self.save_load_vix_dataset()
        self.expirations = self.update_expirations()
        print(self.expirations)
        print('%%%%%%%%%%%%%')

        self.dates = sorted(list(set(self.data.Date)))
        # print(self.dates)
        self.interest_rates = create_interest_rates(interest_rates_file)
        if self.dates[0] > self.current_day:
            self.current_day = self.dates[0]

        while 1:
            self.update_current_day()
            near_next_date = self.get_near_next_term_date()
            if near_next_date[0] != -1:  # there is data for either two fridays or two thursdays
                self.near_term_date = near_next_date[0]
                self.next_term_date = near_next_date[1]
                break

        self.near_term_options = self.create_n_term_dataset(self.near_term_date)
        self.next_term_options = self.create_n_term_dataset(self.next_term_date)
        self.update_put_call_difference()
        K0_F = self.get_K0_F()
        self.K0_near = K0_F[0]
        self.K0_next = K0_F[1]
        self.F_near = K0_F[2]
        self.F_next = K0_F[3]
        self.near_midpoint = self.create_midpoint_dataframe('near')
        self.next_midpoint = self.create_midpoint_dataframe('next')
        self.update_midpoint_contributions()
        self.vix = self.create_vix_dataframe()
        self.count = 1  # todo tirar
        self.save_load_vix()

    def compute_complete_vix(self):
        while True:
            if self.new_day() == -1:
                break

    def create_vix_dataframe(self):
        vix = pd.DataFrame({'Date': [self.current_day], 'Volatility': [self.day_vix()]})
        return vix

    def update_vix(self):
        self.vix = self.vix.append({'Date': self.current_day, 'Volatility': self.day_vix()}, ignore_index=True)

    def create_custom_option_dataset(self):
        filenames_list = open('data/Options/option_dataset_filenames.txt').readlines()
        option_dataset = pd.DataFrame()
        for name in filenames_list:
            # todo tirar
            print('creating ' + self.ticker + ' dataset: ' + str(self.count) + '/' + str(len(filenames_list)))
            self.count += 1  # todo tirar
            data = pd.read_csv(name.rstrip('\n'))  # rstrip removes \n from the end of string
            data = data.drop(data[data.Volume == 0].index)  # drops from data all rows with Volume = 0
            data = data.drop(data[data.UnderlyingSymbol != self.ticker].index)  # drops all rows that are not of that ticker
            data = data.drop(columns=['UnderlyingSymbol', 'UnderlyingPrice', 'Exchange',
                                      'OptionRoot', 'OptionExt', 'Volume', 'OpenInterest', 'T1OpenInterest'])
            data.rename(columns={' DataDate': 'Date', 'DataDate': 'Date'}, inplace=True)
            data['Expiration'] = pd.to_datetime(data['Expiration'])
            data['Date'] = pd.to_datetime(data['Date'])
            data['Strike'] = data['Strike'].astype(np.float16)
            data['Last'] = data['Last'].astype(np.float16)
            data['Bid'] = data['Bid'].astype(np.float16)
            data['Ask'] = data['Ask'].astype(np.float16)
            option_dataset = pd.concat([option_dataset, data], ignore_index=True, sort=False)
        return option_dataset

    def new_day(self):
        if self.current_day >= self.end_day or self.current_day >= self.dates[-40]:
            # se for depois disto já não garante ter uma next term date
            # sees if it's already in the final day
            return -1
        elif self.current_day < self.dates[0]:
            return 0
        self.update_current_day()
        self.expirations = self.update_expirations()
        print('creating ' + self.ticker + ' vix: ' + str(self.count) + '/' + str(len(self.dates)))  # todo tirar
        self.count += 1  # todo tirar
        self.update_near_next_term_options()
        # if len(self.next_term_options.index) < 20:  # TODO VERIFICAR COM O PROF O NUMERO DE OPCOES
        #    return -1
        self.update_K0_F()
        self.update_n_term_midpoint()
        self.update_vix()
        return 1

    def update_current_day(self):
        curr_day = self.current_day
        while True:
            curr_day = curr_day + timedelta(days=1)
            if curr_day in self.dates:
                self.current_day = curr_day
                return None

    def update_expirations(self):
        temp = self.data
        current_day_data = temp.drop(temp[temp.Date != self.current_day].index)
        expirations = sorted(list(set(current_day_data.Expiration)))
        return expirations

    def get_near_next_term_date(self):
        fridays = []
        thursdays = []
        for dates in self.expirations:
            # print('current day:')
            # print(self.current_day)
            # print(dates)
            # print(days_to_date(self.current_day, dates))
            # print('week day')
            # print(dates.isoweekday())
            # print('ººººººººººººººººººººººººººººººººººººº')
            if 62 <= days_to_date(self.current_day, dates) <= 124 \
                    and dates.isoweekday() == 6:
                fridays.append(dates)
            elif 62 <= days_to_date(self.current_day, dates) <= 124 \
                    and dates.isoweekday() == 5:
                thursdays.append(dates)
        # print('00000000000000000')
        # print(fridays)
        # print(thursdays)
        # print(self.current_day)
        # print('00000000000000000')
        if len(fridays) < 2:  # near or next term day is an holiday
            if len(thursdays) < 2:  # there is no data for neither two fridays nor two thursdays
                # print('~~~~~~~~~~~~~~~~~~')
                # print(fridays)
                # print(self.ticker)
                # print(thursdays)
                # print('~~~~~~~~~~~~~~~~~~')
                return -1, -1

            if fridays[0] > thursdays[0] and fridays[0] > thursdays[1]:  # near term day is an holiday
                fridays.append(fridays[0])
                fridays[0] = thursdays[0]
            else:  # next term day is an holiday
                fridays.append(thursdays[1])
        # print('^^^^^^^^^^^^^^^^^^^^')
        # print(fridays)
        # print(thursdays)
        # print('^^^^^^^^^^^^^^^^^^')
        return fridays[0], fridays[1]

    def update_near_next_term_date(self):
        near_next = self.get_near_next_term_date()
        if near_next[0] != -1:
            self.near_term_date = near_next[0]
            self.next_term_date = near_next[1]
        # if there aren't  neither two fridays nor two thursday (mostly because all options are volume = 0)
        # the near and next term dates will be the same as the previous day and the vix will be the same

    def update_near_next_term_options(self):
        self.update_near_next_term_date()
        self.near_term_options = self.create_n_term_dataset(self.near_term_date)
        self.next_term_options = self.create_n_term_dataset(self.next_term_date)
        self.update_put_call_difference()

    def create_n_term_dataset(self, date):  # todo
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print(self.current_day)
        # print(self.near_term_date)
        # print(self.next_term_date)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        n_term_options = pd.DataFrame({'Put': [], 'Call': [], 'Difference': [], 'Put bid': [], 'Call bid': []})
        n_term_options.index.name = 'Strike'
        # options = self.data.drop(self.data[pd.to_datetime(self.data.Expiration) != date].index)
        options = self.data.drop(self.data[self.data.Date != self.current_day].index)
        options = options.drop(options[options.Expiration != date].index)

        # options = options.drop(options[options.Date != self.current_day].index)
        options = options.sort_values(by=['Strike'])
        for idx in options.index:
            line = options.loc[idx]
            if line[3] not in n_term_options.index:
                n_term_options.loc[line[3]] = [np.nan, np.nan, np.nan, np.nan, np.nan]
            if line[0] == 'put':
                n_term_options.at[line[3], 'Put'] = (line[5] + line[6]) / 2
                n_term_options.at[line[3], 'Put bid'] = line[5]
            else:
                n_term_options.at[line[3], 'Call'] = (line[5] + line[6]) / 2
                n_term_options.at[line[3], 'Call bid'] = line[5]
        return n_term_options

    def update_put_call_difference(self):
        for strike in self.near_term_options.index:
            self.near_term_options.at[strike, 'Difference'] = abs(
                self.near_term_options.at[strike, 'Put'] - self.near_term_options.at[strike, 'Call'])
        for strike in self.next_term_options.index:
            self.next_term_options.at[strike, 'Difference'] = abs(
                self.next_term_options.at[strike, 'Put'] - self.next_term_options.at[strike, 'Call'])
        return None

    def find_min_diff_call_put(self):
        diff_near = 1000000000
        diff_next = 1000000000
        strike_near = np.nan
        strike_next = np.nan
        near_call = 0
        next_call = 0
        near_put = 0
        next_put = 0
        for strike in self.near_term_options.index:
            if self.near_term_options.at[strike, 'Difference'] < diff_near:
                diff_near = self.near_term_options.at[strike, 'Difference']
                strike_near = strike
                near_call = self.near_term_options.at[strike, 'Call']
                near_put = self.near_term_options.at[strike, 'Put']
        for strike in self.next_term_options.index:
            if self.next_term_options.at[strike, 'Difference'] < diff_next:
                diff_next = self.next_term_options.at[strike, 'Difference']
                strike_next = strike
                next_call = self.next_term_options.at[strike, 'Call']
                next_put = self.next_term_options.at[strike, 'Put']
        return strike_near, near_call, near_put, strike_next, next_call, next_put

    def find_near_next_R(self):
        near_date = self.near_term_date - timedelta(days=1)
        next_date = self.next_term_date - timedelta(days=1)
        near_R = self.interest_rates.loc[self.interest_rates['DATE'] == near_date]['DGS10'].iloc[0]
        # self.near/next_term_date - 1 day because the options stop trading at a friday but expire at a saturday
        next_R = self.interest_rates.loc[self.interest_rates['DATE'] == next_date]['DGS10'].iloc[0]
        return near_R, next_R

    def calculate_near_next_F(self):
        min_diff_call_put = self.find_min_diff_call_put()
        n_n_R = self.find_near_next_R()
        near_T = time_to_exp_T(self.current_day, self.near_term_date)
        next_T = time_to_exp_T(self.current_day, self.next_term_date)

        near_F = calculate_F(min_diff_call_put[0], n_n_R[0], near_T, min_diff_call_put[1], min_diff_call_put[2])
        next_F = calculate_F(min_diff_call_put[3], n_n_R[1], next_T, min_diff_call_put[4], min_diff_call_put[5])

        return near_F, next_F

    def get_K0_F(self):
        K0_near_F_diff = 100000  # todo mudar para um valor que faça sentido
        K0_next_F_diff = 100000  # todo mudar para um valor que faça sentido
        K0_near = np.nan
        K0_next = np.nan
        n_n_F = self.calculate_near_next_F()
        for strike in self.near_term_options.index:
            if 0 < (n_n_F[0] - strike) < K0_near_F_diff:
                K0_near_F_diff = n_n_F[0] - strike
                K0_near = strike
        for strike in self.next_term_options.index:
            if 0 < (n_n_F[1] - strike) < K0_next_F_diff:
                K0_next_F_diff = n_n_F[1] - strike
                K0_next = strike
        return K0_near, K0_next, n_n_F[0], n_n_F[1]

    def update_K0_F(self):
        K0_F = self.get_K0_F()
        self.K0_near = K0_F[0]
        self.K0_next = K0_F[1]
        self.F_near = K0_F[2]
        self.F_next = K0_F[3]

    def create_midpoint_dataframe(self, n_term):
        n_midpoint = pd.DataFrame({'Type': [], 'Midpoint Price': [], 'Contribution': []})
        n_midpoint.index.name = 'Strike'
        if n_term == 'near':
            n_term_options = self.near_term_options
            K0 = self.K0_near
        elif n_term == 'next':
            n_term_options = self.next_term_options
            K0 = self.K0_next
        else:
            return None
        print('++++++++++++++++')
        print(self.current_day)
        print(self.near_term_date)
        print(self.next_term_date)
        K0_mid_p_avg = abs((n_term_options.loc[K0]['Put'] + n_term_options.loc[K0]['Call']) / 2)
        n_midpoint.loc[K0] = ['PutCall', K0_mid_p_avg, np.nan]
        bid_0 = 0
        for strike in n_term_options.index:
            if strike > K0:
                if n_term_options.loc[strike]['Call bid'] == np.nan or n_term_options.loc[strike]['Call bid'] == 0:
                    bid_0 = bid_0 + 1
                else:
                    bid_0 = 0
                    n_midpoint.loc[strike] = ['Call', n_term_options.loc[strike]['Call'], np.nan]
                if bid_0 == 2:
                    break

        bid_0 = 0
        for strike in reversed(n_term_options.index):
            if strike < K0:
                if n_term_options.loc[strike]['Put bid'] == np.nan or n_term_options.loc[strike]['Put bid'] == 0:
                    bid_0 = bid_0 + 1
                else:
                    bid_0 = 0
                    n_midpoint.loc[strike] = ['Put', n_term_options.loc[strike]['Put'], np.nan]
                if bid_0 == 2:
                    break

        n_midpoint = n_midpoint.sort_index()
        return n_midpoint

    def update_n_term_midpoint(self):
        self.near_midpoint = self.create_midpoint_dataframe('near')
        self.next_midpoint = self.create_midpoint_dataframe('next')
        self.update_midpoint_contributions()

    def update_midpoint_contributions(self):
        # todo verificar se é mesmo - timedelta(days)=1. meti porque as opções expiram ao sabado embora o ultimo dia
        #  de trading seja na sexta anterior. como não exitem interest rates ao sábado estou a usar a sexta
        #  simediatamente antes
        R = float(self.interest_rates.loc[self.interest_rates['DATE'] == self.near_term_date - timedelta(days=1), 'DGS10'])
        T = time_to_exp_T(self.current_day, self.near_term_date)
        for strike in self.near_midpoint.index:
            self.near_midpoint.at[strike, "Contribution"] = self.calculate_contribution(strike, 'near', R, T)

        R = float(self.interest_rates.loc[self.interest_rates['DATE'] == self.next_term_date - timedelta(days=1), 'DGS10'])
        T = time_to_exp_T(self.current_day, self.next_term_date)
        for strike in self.next_midpoint.index:
            self.next_midpoint.at[strike, 'Contribution'] = self.calculate_contribution(strike, 'next', R, T)

    def calculate_contribution(self, strike, n_term, R, T):
        if n_term == 'near':
            midpoint = self.near_midpoint
        elif n_term == 'next':
            midpoint = self.next_midpoint
        else:
            return None
        if strike == midpoint.index[0]:
            delta_K = midpoint.index[1] - strike
        elif strike == midpoint.index[-1]:
            delta_K = strike - midpoint.index[-2]
        else:
            idx = midpoint.index.get_loc(strike)
            delta_K = (midpoint.index[idx + 1] - midpoint.index[idx - 1]) / 2
        contribution = (delta_K / strike ** 2) * np.exp(R * T) * midpoint.at[strike, 'Midpoint Price']
        return contribution

    def calculate_n_volatility(self, n_term):
        if n_term == 'near':
            T = time_to_exp_T(self.current_day, self.near_term_date)
            midpoint = self.near_midpoint
            F = self.F_near
            K = self.K0_near
        else:
            T = time_to_exp_T(self.current_day, self.next_term_date)
            midpoint = self.next_midpoint
            F = self.F_next
            K = self.K0_next
        sqr_vol = sqr_volatility(T, midpoint, F, K)
        return sqr_vol

    def day_vix(self):
        T1 = time_to_exp_T(self.current_day, self.near_term_date)
        T2 = time_to_exp_T(self.current_day, self.next_term_date)
        vol1 = self.calculate_n_volatility('near')
        vol2 = self.calculate_n_volatility('next')
        NT1 = minutes_to_date(self.current_day, self.near_term_date)
        NT2 = minutes_to_date(self.current_day, self.next_term_date)
        N93 = 133920
        N365 = 525600
        day_vix = 100 * np.sqrt(
            (T1 * vol1 * ((NT2 - N93) / (NT2 - NT1))
             + T2 * vol2 * ((N93 - NT1) / (NT2 - NT1)))
            * (N365 / N93))
        print(day_vix)
        return day_vix

    def save_load_vix_dataset(self):
        filepath = 'data/custom_vix/' + self.ticker + '_dataset.pickle'
        u = 0
        if os.path.exists(filepath):
            data = load_vix_dataset(filepath)
        else:
            data = self.create_custom_option_dataset()
            save_vix_dataset(data, filepath)
        return data

    def save_load_vix(self):
        filepath = 'data/custom_vix/' + self.ticker + '_vix.pickle'
        if os.path.exists(filepath):
            self.vix = load_vix(filepath)
        else:
            self.compute_complete_vix()
            save_vix_dataset(self.vix, filepath)