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


def vix_info():
    path_VIX9D = "data/VIX/VIX9D.ods"
    path_VIX30D = "data/VIX/VIX30D.ods"
    path_VIX3M = "data/VIX/VIX3M.ods"
    path_VIX6M = "data/VIX/VIX6M.ods"

    VIX9D = read_ods(path_VIX9D, 1)  # takes data from spreadsheet
    VIX9D = vix_to_dataframe(VIX9D)  # transforms data into a dataframename
    VIX30D = read_ods(path_VIX30D, 1)
    VIX30D = vix_to_dataframe(VIX30D)
    VIX3M = read_ods(path_VIX3M, 1)
    VIX3M = vix_to_dataframe(VIX3M)
    VIX6M = read_ods(path_VIX6M, 1)
    VIX6M = vix_to_dataframe(VIX6M)

    # st.graph_vix(VIX9D, "VIX 9 Days", 1)
    # st.graph_vix(VIX30D, "VIX 30 Days", 2)
    # st.graph_vix(VIX3M, "VIX 3 Months", 3)
    # st.graph_vix(VIX6M, "VIX 6 Months", 4)

    VIX_9D_30D = vix1_vix2_to_dataframe(VIX9D, VIX30D)
    VIX_9D_3M = vix1_vix2_to_dataframe(VIX9D, VIX3M)
    VIX_9D_6M = vix1_vix2_to_dataframe(VIX9D, VIX6M)
    VIX_30D_9D = vix1_vix2_to_dataframe(VIX30D, VIX9D)
    VIX_30D_3M = vix1_vix2_to_dataframe(VIX30D, VIX3M)
    VIX_30D_6M = vix1_vix2_to_dataframe(VIX30D, VIX6M)
    VIX_3M_9D = vix1_vix2_to_dataframe(VIX3M, VIX9D)
    VIX_3M_30D = vix1_vix2_to_dataframe(VIX3M, VIX30D)
    VIX_3M_6M = vix1_vix2_to_dataframe(VIX3M, VIX6M)
    VIX_6M_9D = vix1_vix2_to_dataframe(VIX6M, VIX9D)
    VIX_6M_30D = vix1_vix2_to_dataframe(VIX6M, VIX30D)
    VIX_6M_3M = vix1_vix2_to_dataframe(VIX6M, VIX3M)

    st.graph_vix(VIX_9D_30D, 'VIX 9 days minus VIX 30 days', 1)
    # st.graph_vix(VIX_9D_3M, 'VIX 9 days minus VIX 3 months', 2)
    # st.graph_vix(VIX_9D_6M, 'VIX 9 days minus VIX 6 months', 3)
    # st.graph_vix(VIX_30D_9D, 'VIX 30 days minus VIX 9 days', 4)
    # st.graph_vix(VIX_30D_3M, 'VIX 30 days minus VIX 3 months', 5)
    # st.graph_vix(VIX_30D_6M, 'VIX 30 days minus VIX 6 months', 6)
    # st.graph_vix(VIX_3M_9D, 'VIX 3 months minus VIX 9 days', 7)
    # st.graph_vix(VIX_3M_30D, 'VIX 3 months minus VIX 30 days', 8)
    # st.graph_vix(VIX_3M_6M, 'VIX 3 months minus VIX 6 months', 9)
    # st.graph_vix(VIX_6M_9D, 'VIX 6 months minus VIX 9 days', 10)
    # st.graph_vix(VIX_6M_30D, 'VIX 6 months minus VIX 30 days', 11)
    # st.graph_vix(VIX_6M_3M, 'VIX 6 months minus VIX 3 months', 12)


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
    with open('data/Options/sp500_tickers_all.txt') as f:
        tickers = f.read().splitlines()
    return tickers


def open_sp500_tickers_to_str():
    with open('data/SP500/sp500_tickers_all.txt') as f:
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
    file = 'data/implied_volatility/all_tickers_ivol.csv'
    if path.exists(file):
        os.remove(file)
        print("File Removed!")

    tickers = open_sp500_tickers_to_list()
    all_ivol = pd.DataFrame()
    for ticker in tickers:
        filepath = 'data/implied_volatility/HistoricalIV_' + ticker + '.csv'
        if path.exists(filepath):
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
            data = data.sort_index()
            data.drop(data.columns.difference(['IV30']), 1, inplace=True)
            data = data.rename({'IV30': ticker}, axis='columns')
            all_ivol = pd.concat([all_ivol, data], axis=1)
    all_ivol.to_csv('data/implied_volatility/all_tickers_ivol.csv')
    return all_ivol

def save_best(best, title):
    now = pd.to_datetime("now")
    now = now.strftime('%r:%d-%m-%Y')
    filepath = 'data/results/' + title + '-' + now + '.pickle'
    pickle.dump(best, open( filepath, "wb" ))
    print('result saved')
