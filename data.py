# import statistics as st
import pandas as pd
from pandas_ods_reader import read_ods
import numpy as np
import pickle
import os.path as path
import os
import yfinance as yf

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
