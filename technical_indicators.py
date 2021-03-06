import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import data as d
import os.path
import math
from multiprocessing import Pool


def compute_technical_signals(n):
    tickers = d.open_all_sp500_tickers_to_list()  # TODO METER LISTA COMPLETA DE TECHNICAL INDICATORS
    # rsi_stock_signals = pd.DataFrame()
    # roc_stock_signals = pd.DataFrame()
    # sto_stock_signals = pd.DataFrame()

    rsi_ivol_signals = pd.DataFrame()
    roc_ivol_signals = pd.DataFrame()
    sto_ivol_signals = pd.DataFrame()
    macd_ivol_signals = pd.DataFrame()

    data = pd.DataFrame()

    # exists_vix_rsi = os.path.isfile('data/technical_indicators/' + str(n) + '_vix_rsi.csv')
    # exists_vix_roc = os.path.isfile('data/technical_indicators/' + str(n) + '_vix_roc.csv')

    # exists_stock_rsi = os.path.isfile('data/technical_indicators/' + str(n) + '_stock_rsi.csv')
    # exists_stock_roc = os.path.isfile('data/technical_indicators/' + str(n) + '_stock_roc.csv')
    # exists_stock_sto = os.path.isfile('data/technical_indicators/' + str(n) + '_stock_sto.csv')

    exists_ivol_rsi = os.path.isfile('data/technical_indicators/' + str(n) + '_ivol_rsi.csv')
    exists_ivol_roc = os.path.isfile('data/technical_indicators/' + str(n) + '_ivol_roc.csv')
    exists_ivol_sto = os.path.isfile('data/technical_indicators/' + str(n) + '_ivol_sto.csv')
    exists_ivol_macd = os.path.isfile('data/technical_indicators/' + str(n) + '_ivol_macd.csv')


    # Vix signal
    # filepath = 'data/VIX/VIX30D.csv'
    # data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    # if not exists_vix_rsi:
    #     rsi_vix = RSI(data, n).rename(columns={'value': 'vix_rsi', 'Unnamed: 0':'Date'})
    #     save_technical_indicator(rsi_vix, str(n)+'_vix_rsi')
    # if not exists_vix_roc:
    #     roc_vix = ROC(data, n).rename(columns={'value': 'vix_roc', 'Unnamed: 0':'Date'})
    #     save_technical_indicator(roc_vix, str(n)+'_vix_roc')

    # Stock signals
    # filepath = 'data/yfinance/all_tickers_yfinance.csv'
    # data_all_tic = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    # for tic in tickers:
    #     data['close'] = data_all_tic[tic]
    #     if not exists_stock_rsi:
    #         print('stock rsi: ' + tic + ' - ' + str(n))
    #         rsi_tic = RSI(data, n, stock=True).rename(columns={'value': 'stock_' + tic + '_rsi', 'Unnamed: 0':'Date'})
    #         rsi_stock_signals = pd.concat([rsi_stock_signals, rsi_tic], axis=1)
    #     if not exists_stock_roc:
    #         print('stock roc: ' + tic + ' - ' + str(n))
    #         roc_tic = ROC(data, n, stock=True).rename(columns={'value': 'stock_' + tic + '_roc', 'Unnamed: 0':'Date'})
    #         roc_stock_signals = pd.concat([roc_stock_signals, roc_tic], axis=1)
    #     if not exists_stock_sto:
    #         print('stock sto: ' + tic + ' - ' + str(n))
    #         sto_tic = StO(data, n, stock=True).rename(columns={'value': 'stock_' + tic + '_sto', 'Unnamed: 0':'Date'})
    #         sto_stock_signals = pd.concat([sto_stock_signals, sto_tic], axis=1)
    #
    # if not exists_stock_rsi: save_technical_indicator(rsi_stock_signals, str(n)+'_stock_rsi')
    # if not exists_stock_roc: save_technical_indicator(roc_stock_signals, str(n)+'_stock_roc')
    # if not exists_stock_sto: save_technical_indicator(sto_stock_signals, str(n)+'_stock_sto')

    # Implied volatility
    # filepath = 'data/implied_volatility/all_tickers_ivol.csv'
    filepath = 'data/implied_volatility/all_tickers_smooth_ivol_(12).csv'
    data_all_tic = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for tic in tickers:
        data['close'] = data_all_tic[tic]
        if not exists_ivol_rsi:
            print('ivol rsi: ' + tic + ' - ' + str(n))
            rsi_tic = RSI(data, n).rename(columns={'value': 'ivol_' + tic + '_rsi', 'Unnamed: 0':'Date'})
            rsi_ivol_signals = pd.concat([rsi_ivol_signals, rsi_tic], axis=1)
        if not exists_ivol_roc:
            print('ivol roc: ' + tic + ' - ' + str(n))
            roc_tic = ROC(data, n).rename(columns={'value': 'ivol_' + tic + '_roc', 'Unnamed: 0':'Date'})
            roc_ivol_signals = pd.concat([roc_ivol_signals, roc_tic], axis=1)
        if not exists_ivol_sto:
            print('ivol sto: ' + tic + ' - ' + str(n))
            sto_tic = StO(data, n).rename(columns={'value': 'ivol_' + tic + '_sto', 'Unnamed: 0':'Date'})
            sto_ivol_signals = pd.concat([sto_ivol_signals, sto_tic], axis=1)
        if not exists_ivol_macd:
            print('ivol macd: ' + tic + ' - ' + str(n))
            macd_tic = MACD(data, n, n+14).rename(columns={'value': 'ivol_' + tic + '_macd', 'Unnamed: 0':'Date'})
            macd_ivol_signals = pd.concat([macd_ivol_signals, macd_tic], axis=1)
    if not exists_ivol_rsi: save_technical_indicator(rsi_ivol_signals, str(n)+'_ivol_rsi')
    if not exists_ivol_roc: save_technical_indicator(roc_ivol_signals, str(n)+'_ivol_roc')
    if not exists_ivol_sto: save_technical_indicator(sto_ivol_signals, str(n)+'_ivol_sto')
    if not exists_ivol_macd: save_technical_indicator(macd_ivol_signals, str(n)+'_ivol_macd')

def compute_xema_signals(n):
    tickers = d.open_all_sp500_tickers_to_list()  # TODO METER LISTA COMPLETA DE TECHNICAL INDICATORS

    xema_ivol_signals = pd.DataFrame()
    data = pd.DataFrame()

    exists_ivol_xema = os.path.isfile('data/technical_indicators/' + str(n[0]) +'-'+ str(n[1]) + '_xema_rsi.csv')

    filepath = 'data/implied_volatility/all_tickers_smooth_ivol_(12).csv'
    data_all_tic = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for tic in tickers:
        data['close'] = data_all_tic[tic]
        if not exists_ivol_xema:
            print('ivol xema: ' + tic + ' - ' + str(n[0]) +'-'+ str(n[1]))
            xema_tic = XEMA(data, n[0], n[1]).rename(columns={'value': 'ivol_' + tic + '_xema', 'Unnamed: 0':'Date'})
            xema_ivol_signals = pd.concat([xema_ivol_signals, xema_tic], axis=1)
    if not exists_ivol_xema: save_technical_indicator(xema_ivol_signals, str(n[0])+'-'+str(n[1])+'_ivol_xema')

############################
#       extra methods      #
############################
def compute_all_technical_signals(min=30, max=60, n_threads= 5):
    short_n = np.arange(2, 19)
    long_n = np.arange(20, 101, 5)
    n = []
    for i in short_n:
        for j in long_n:
            n.append([i,j])
    with Pool(n_threads) as p:
        p.map(compute_xema_signals, n)

    n = np.arange(min, max+1)
    with Pool(n_threads) as p:
        p.map(compute_technical_signals, n)


def normalization(signal, s_min=0, s_max=0):
    signal_norm = signal.copy()
    if not(s_max or s_min):
        s_max = signal_norm['value'].max()
        s_min = signal_norm['value'].min()

    for i in signal.index:
        x = signal_norm.at[i, 'value']
        signal_norm.at[i, 'value'] = ((x - s_min) / (s_max - s_min)) * 100
    return signal_norm


def save_technical_indicator(signal, filename):
    filepath = 'data/technical_indicators/' + filename + '.csv'
    signal.to_csv(filepath)


def load_technical_indicators():
    ti = pd.DataFrame()
    fp_vix_rsi = 'data/technical_indicators/vix_rsi.csv'
    fp_vix_roc = 'data/technical_indicators/vix_roc.csv'
    fp_stock_rsi = 'data/technical_indicators/stock_rsi.csv'
    fp_stock_roc = 'data/technical_indicators/stock_roc.csv'
    fp_ivol_rsi = 'data/technical_indicators/ivol_rsi.csv'
    fp_ivol_roc = 'data/technical_indicators/ivol_roc.csv'

    vix_rsi = pd.read_csv(fp_vix_rsi, index_col='Date', parse_dates=True)
    vix_roc = pd.read_csv(fp_vix_roc, index_col='Date', parse_dates=True)
    stock_rsi = pd.read_csv(fp_stock_rsi, index_col='Date', parse_dates=True)
    stock_roc = pd.read_csv(fp_stock_roc, index_col='Date', parse_dates=True)
    ivol_rsi = pd.read_csv(fp_ivol_rsi, index_col='Date', parse_dates=True)
    ivol_roc = pd.read_csv(fp_ivol_roc, index_col='Date', parse_dates=True)

    ti = pd.concat([vix_rsi, vix_roc,
                    stock_rsi, stock_roc,
                    ivol_rsi, ivol_roc], axis=1)
    return ti


def nan_to_50(value):
    if np.isnan(value):
        value = 50
    return value


############################
#   Technical Indicators   #
############################
def RSI(raw_signal, n=14, stock=False):
    calc = raw_signal.copy()
    calc['up'] = np.nan
    calc['down'] = np.nan
    calc['avg_up'] = np.nan
    calc['avg_down'] = np.nan
    calc['rsi'] = np.nan

    for i in range(1, len(calc)):
        change = calc.iloc[i].at['close'] - calc.iloc[i-1].at['close']
        if change > 0:
            calc.at[calc.index[i], 'up'] = abs(change)
            calc.at[calc.index[i], 'down'] = 0
        elif change < 0:
            calc.at[calc.index[i], 'up'] = 0
            calc.at[calc.index[i], 'down'] = abs(change)
        else:
            calc.at[calc.index[i], 'up'] = 0
            calc.at[calc.index[i], 'down'] = 0

    sum_u = 0
    sum_d = 0
    for i in range(1, n):
        sum_u += calc.iloc[i].at['up']
        sum_d += calc.iloc[i].at['down']

    calc.at[calc.index[n], 'avg_up'] = sum_u / n
    calc.at[calc.index[n], 'avg_down'] = sum_d / n
    if calc.iloc[n].at['avg_down']:
        calc.at[calc.index[n], 'rsi'] = 100 - 100 / (1 + calc.iloc[n].at['avg_up'] / calc.iloc[n].at['avg_down'])
    else:
        calc.at[calc.index[n], 'rsi'] = 100

    for i in range(n + 1, len(calc)):
        calc.at[calc.index[i], 'avg_up'] = ((n - 1) * calc.iloc[i-1].at['avg_up'] + calc.iloc[i].at['up']) / n
        calc.at[calc.index[i], 'avg_down'] = ((n - 1) * calc.iloc[i-1].at['avg_down'] + calc.iloc[i].at['down']) / n
        if calc.iloc[n].at['avg_down']:
            calc.at[calc.index[i], 'rsi'] = 100 - 100 / (1 + calc.iloc[i].at['avg_up'] / calc.iloc[i].at['avg_down'])
        else:
            calc.at[calc.index[i], 'rsi'] = 100


    if stock:
        for i in range(n, len(calc)):
            value = calc.at[calc.index[i], 'rsi']
            calc.at[calc.index[i], 'rsi'] = abs(value)

    signal = pd.DataFrame(index=calc.index)
    signal['value'] = calc['rsi']
    return signal


def ROC(raw_signal, n=14, stock=False):
    calc = raw_signal.copy()
    calc['value'] = np.nan

    for i in range(n, len(calc)):
        calc.at[calc.index[i], 'value'] = (calc.iloc[i]['close'] / calc.iloc[i-n]['close'] - 1) * 100

    if stock:
        for i in range(n, len(calc)):
            calc.at[calc.index[i], 'value'] = abs(calc.at[calc.index[i], 'value'])
    calc = normalization(calc, -50, 100) # the limits are 2x(100) and 1/2x(-50)

    signal = pd.DataFrame(index=calc.index)
    signal['value'] = calc['value']
    return signal


def StO(raw_signal, n=14, stock=False):
    calc = raw_signal.copy()
    calc['value'] = np.nan
    values = np.array([calc.iloc[0:n]['close']])
    for i in range(n, len(calc)):
        values = np.append(values, calc.iloc[i]['close'])
        max = values.max()
        min = values.min()
        if max and min and max!=min:
            calc.at[calc.index[i], 'value'] = ((calc.iloc[i]['close'] - min) / (max - min)) * 100
        else:
            calc.at[calc.index[i], 'value'] = 50 # no data to compute
        values = np.delete(values, 0)

    if stock:
        for i in range(n, len(calc)):
            value = calc.at[calc.index[i], 'value']
            calc.at[calc.index[i], 'value'] = abs(value)

    signal = pd.DataFrame(index=calc.index)
    signal['value'] = calc['value']
    return signal


def EMA(raw_signal, n=14):
    calc = raw_signal.copy()
    calc['value'] = calc['close']
    k = 2 / (n-1)
    sum = 0

    for i in range(n):
        sum += calc.iloc[i]['close']

    calc.at[calc.index[n], 'value'] = sum/n
    for i in range(n+1, len(calc)):
        if math.isnan(calc.iloc[i]['close']):
            calc.at[calc.index[i], 'value'] = calc.iloc[i-1]['value']
        else:
            calc.at[calc.index[i], 'value'] = calc.iloc[i]['close'] * k + calc.iloc[i-1]['value'] * (1-k)
    return calc


def MACD(raw_signal, n1=12, n2=26):
    calc = raw_signal.copy()
    ema1 = EMA(calc, n1)
    ema2 = EMA(calc, n2)

    for i in range(n2, len(ema2)):
        calc.at[calc.index[i], 'close'] = ema1.iloc[i]['value'] - ema2.iloc[i]['value']

    signal_line = EMA(calc, 9)
    for i in range(n2, len(calc)):
        macd = calc.iloc[i]['close']
        sig_l = signal_line.iloc[i]['value']
        if macd > sig_l:
            calc.at[calc.index[i], 'value'] = 50+(abs(macd-sig_l)/abs(sig_l))*25
            if calc.at[calc.index[i], 'value'] > 100: calc.at[calc.index[i], 'value'] = 100
        else:
            calc.at[calc.index[i], 'value'] = 50-(abs(sig_l-macd)/abs(macd))*25
            if calc.at[calc.index[i], 'value'] < 0: calc.at[calc.index[i], 'value'] = 0

    signal = pd.DataFrame(index=calc.index)
    signal['value'] = calc['value']
    return signal


def XEMA(raw_signal, n1=2, n2=20):
    calc = raw_signal.copy()
    short_ema = EMA(calc, n1)
    long_ema = EMA(calc, n2)

    for i in range(n2, len(calc)):
        if short_ema.iloc[i]['value'] > long_ema.iloc[i]['value']:
            calc.at[calc.index[i], 'value'] = 100
        else:
            calc.at[calc.index[i], 'value'] = 0

    signal = pd.DataFrame(index=calc.index)
    signal['value'] = calc['value']
    return signal

def MA(raw_signal, n=14):
    calc = raw_signal.copy()
    calc['value'] = calc['close']
    sum = 0

    for i in range(n):

        sum += calc.iloc[i]['close']
    print(sum)

    calc.at[calc.index[n], 'value'] = sum/n
    for i in range(n+1, len(calc)):
        if math.isnan(calc.iloc[i]['close']):
            calc.at[calc.index[i], 'value'] = calc.iloc[i-1]['value']
        else:
            calc.at[calc.index[i], 'value'] = (calc.iloc[i]['close'] + calc.iloc[i-1]['value'] * (n-1))/n
    return calc

# compute_all_technical_signals(min=5, max=60, n_threads= 10)
