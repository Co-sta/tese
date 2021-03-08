import pandas as pd
from pandas_ods_reader import read_ods
import numpy as np
import pickle
import os.path as path
import os
import yfinance as yf
import math




def vix_to_dataframe(vix):
    date = vix['date'].to_list()
    for i, item in enumerate(date):
        temp = item.split('/')
        item = temp[2] + '-' + temp[0] + '-' + temp[1]  # invertes the string. M-D-Y to Y-M-D
        vix.loc[i]['date'] = pd.to_datetime(item)
    close = vix['close'].to_list()
    for i, item in enumerate(close):
        vix.loc[i]['close'] = float(item)
    return vix

def vix1_vix2_to_dataframe(vix1, vix2):
    vix = vix1.copy()
    y1 = vix1['close'].to_list()
    y2 = vix2['close'].to_list()
    y = []
    for i, item in enumerate(y1):
        y.append(float(item))
    for i, item in enumerate(y2):
        vix.loc[i]['close'] = y[i] - float(item)
    return vix

def load_option_dataset(tic):
    filepath = 'data/companies_options/' + tic + '_dataset.pickle'
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data
    else:
        print('There is no existing dataset for that ticker')
        return -1

def save_option_dataset(tic='none'):
    if tic == 'none':
        i = 0
        for ticker in open_sp500_tickers_to_list():
            i += 1
            print('doing... ' + str(i) + ' of ' + str(len(open_sp500_tickers_to_list())))
            filepath = 'data/companies_options/' + ticker + '_dataset.pickle'
            if os.path.exists(filepath):
                print('Dataset already exists. \n Continuing...')
            else:
                data = create_custom_option_dataset(ticker)
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
    else:
        filepath = 'data/companies_options/' + tic + '_dataset.pickle'
        if os.path.exists(filepath):
            print('Dataset already exists.')
        else:
            data = create_custom_option_dataset(tic)
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

def create_custom_option_dataset(ticker):
    filenames_list = open('data/' + 'Options/option_dataset_filenames.txt').readlines()
    option_dataset = pd.DataFrame()
    count = 0
    for name in filenames_list:
        # todo tirar
        # print('creating ' + ticker + ' dataset: ' + str(count) + '/' + str(len(filenames_list)))
        count += 1  # todo tirar
        data = pd.read_csv(name.rstrip('\n'))  # rstrip removes \n from the end of string
        data = data.drop(data[data.Volume == 0].index)  # drops from data all rows with Volume = 0
        data = data.drop(data[data.UnderlyingSymbol != ticker].index)  # drops all rows that are not of that ticker
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

def open_sp500_tickers_to_list():
    with open('data/SP500/some_tickers.txt') as f:
        tickers = f.read().splitlines()
    return tickers

def open_all_sp500_tickers_to_list():
    with open('data/SP500/all_tickers.txt') as f:
        tickers = f.read().splitlines()
    return tickers

def open_sp500_tickers_to_str():
    with open('data/SP500/some_tickers.txt') as f:
        tickers = f.read().replace('\n', ' ')  # transforms all strings in just one (separated by a space)
    return tickers

def create_stock_dataset(ds_start, ds_end):
    sp500_tickers = open_sp500_tickers_to_str()
    dataset = yf.download(sp500_tickers, ds_start, ds_end)
    dataset = dataset.Close
    save_dataset(dataset, 'stock_dataset')

def save_dataset(ds, filename):
    filepath = 'data/SP500/' + filename + '.pickle'
    with open(filepath, 'wb') as f:
        pickle.dump(ds, f)

def load_dataset(filename='stock_dataset'):
    filepath = 'data/companies_options/' + filename + '.pickle'
    with open(filepath, 'rb') as f:
        ds = pickle.load(f)
    return ds

def create_ivol_dataset():
    filename = 'data/implied_volatility/all_tickers_ivol.csv'
    if path.exists(filename):
        os.remove(filename)
        print("File Removed!")

    tickers = open_all_sp500_tickers_to_list()
    all_ivol = pd.DataFrame()
    for ticker in tickers:
        print(ticker)
        filepath = 'data/implied_volatility/HistoricalIV_' + ticker + '.csv'
        if path.exists(filepath):
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            data = data.sort_index()
            data.drop(data.columns.difference(['IV30']), 1, inplace=True)
            data = data.rename({'IV30': ticker}, axis='columns')
            all_ivol = pd.concat([all_ivol, data], axis=1)
    all_ivol.to_csv(filename)
    print("File Saved!")
    return all_ivol

def create_smooth_ivol_dataset(n):
    filename = 'data/implied_volatility/all_tickers_smooth_ivol_('+str(n)+').csv'
    if path.exists(filename):
        os.remove(filename)
        print("File Removed!")

    tickers = open_all_sp500_tickers_to_list()
    all_ivol = pd.DataFrame()
    for ticker in tickers:
        print(ticker)
        filepath = 'data/implied_volatility/HistoricalIV_' + ticker + '.csv'
        if path.exists(filepath):
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            data = data.sort_index()
            data['close'] = data['IV30']
            data = EMA(data, n)
            data.drop(data.columns.difference(['value']), 1, inplace=True)
            data = data.rename({'value': ticker}, axis='columns')
            all_ivol = pd.concat([all_ivol, data], axis=1)
    all_ivol.to_csv(filename)
    print("File Saved!")
    return all_ivol

def EMA(raw_signal, n=14):
    calc = raw_signal.copy()
    calc['value'] = calc['close']
    k = 2 / (n-1)
    sum = 0

    for i in range(n):
        sum += calc.iloc[i]['close']

    calc.at[calc.index[n], 'value'] = sum/n
    for i in range(n+1, len(calc)):
        calc.at[calc.index[i], 'value'] = calc.iloc[i]['close'] * k + calc.iloc[i-1]['value'] * (1-k)
    return calc

def MA(raw_signal, n=14):
    calc = raw_signal.copy()
    calc['value'] = calc['close']
    sum = 0

    for i in range(n):
        sum += calc.iloc[i]['close']

    calc.at[calc.index[n], 'value'] = sum/n
    for i in range(n+1, len(calc)):
        if math.isnan(calc.iloc[i]['close']):
            calc.at[calc.index[i], 'value'] = calc.iloc[i-1]['value']
        else:
            calc.at[calc.index[i], 'value'] = (calc.iloc[i]['close'] + calc.iloc[i-1]['value'] * (n-1))/n
    return calc

def create_yfinance_dataset():
    tickers = open_all_sp500_tickers_to_list()
    filename = 'data/yfinance/all_tickers_yfinance.csv'
    if path.exists(filename):
        os.remove(filename)
        print("File Removed!")
    all_yfiance = yf.download(tickers, start="2011-01-03", end="2016-01-01")
    all_yfiance = all_yfiance.drop(['Low', 'Adj Close', 'Open', 'Volume', 'High'], axis=1)
    all_yfiance.columns = all_yfiance.columns.get_level_values(1)
    all_yfiance.to_csv(filename)
    print("File Saved!")
    return all_yfiance

def save_best(best, title):
    now = pd.to_datetime("now")
    now = now.strftime('%d-%m-%Y:%r')
    filepath = 'data/results/train/' + title + '-' + now + '.pickle'
    pickle.dump(best, open( filepath, "wb" ))
    print('best population saved')
    return filepath

def save_portfolio(portfolio, time_period=0, filepath=0):
    if not filepath:
        now = pd.to_datetime("now")
        now = now.strftime('%d-%m-%Y:%r')
        filepath = 'data/results/test/' + time_period + '-' + now + '.pickle'
    pickle.dump(portfolio, open( filepath, "wb" ))
    print('portfolio saved')

def compress_dataframe():
    filenames = open('data/Options/option_dataset_filenames(some).txt').readlines()
    data = pd.DataFrame()
    for filename in filenames:
        print(filename)
        with open(filename.rstrip('\n')) as file:
            option_dataset = pd.read_csv(file, usecols=["UnderlyingSymbol",
                                                        "UnderlyingPrice",
                                                        "OptionRoot",
                                                        "Type",
                                                        "Expiration",
                                                        "DataDate",
                                                        "Strike",
                                                        "Ask"])

        option_dataset = option_dataset.rename(columns={"DataDate": "Date"})
        option_dataset = option_dataset.astype({"UnderlyingSymbol": 'string',
                                                    "UnderlyingPrice": 'float16',
                                                    "OptionRoot": 'string',
                                                    "Type": 'category',
                                                    "Expiration": 'category',
                                                    "Date": 'category',
                                                    "Strike": 'float16',
                                                    "Ask": 'float16'})
        option_dataset.to_csv(filename.rstrip('\n'), index=False)

def create_options_xma(ticker):
    n1 = 5
    n2 = 32
    filenames = open('data/Options/option_dataset_filenames.txt').readlines()
    options = pd.DataFrame(index=[pd.to_datetime('01-03-2011')])
    data = pd.DataFrame()

    print('aggregating option dataframe:')
    for filename in filenames:
        print(filename)
        date = date_from_filename(filename)
        roots = roots_from_df(filename)
        roots = [elem for elem in roots if ticker in elem]
        values = value_from_df(filename, roots)

        for (root, value) in zip(roots, values):
            options.at[date, root] = value
    options.to_csv('data/options_xma/temp.csv')

    print('applying MA5 and MA32 to each column:')
    for (columnName, columnData) in options.iteritems():
        print('option root: ', columnName)
        data['close'] = options[columnName]
        print('5 days ma:')
        short_ma = MA(data, n1)
        print('32 days ma:')
        long_ma = MA(data, n2)

        print('creating xma:')
        for i in range(n2):
            options.at[options.index[i], columnName] = 0
        for i in range(n2, len(data)):
            if math.isnan(long_ma.iloc[i]['value']):
                options.at[options.index[i], columnName] = 0
            else:
                if short_ma.iloc[i]['value'] > long_ma.iloc[i]['value']:
                    options.at[options.index[i], columnName] = 1
                elif short_ma.iloc[i]['value'] < long_ma.iloc[i]['value']:
                    options.at[options.index[i], columnName] = -1
                else:
                    options.at[options.index[i], columnName] = 0
    filepath = 'data/options_xma/' + ticker + '.csv'
    options.to_csv(filepath)




def roots_from_df(filename):
    with open(filename.rstrip('\n')) as file:
        option_dataset = pd.read_csv(file, usecols=["OptionRoot"])
        roots = option_dataset["OptionRoot"].tolist()
    return roots

def value_from_df(filename, roots):
    values = []
    with open(filename.rstrip('\n')) as file:
        option_dataset = pd.read_csv(file, usecols=["OptionRoot", "Ask"])
        for root in roots:
            values.append(option_dataset[option_dataset["OptionRoot"] == root]["Ask"].iloc[0])
    return values

def date_from_filename(filename):
    date_txt = filename.split('_')[-1].split('.')[0]
    date = pd.to_datetime(date_txt, format='%Y%m%d')
    return date


create_options_xma('AAPL')
