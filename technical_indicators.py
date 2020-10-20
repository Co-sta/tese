import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import data as d


def compute_technical_signals():
    tickers = d.open_all_sp500_tickers_to_list()
    rsi_stock_signals = pd.DataFrame()
    roc_stock_signals = pd.DataFrame()
    rsi_ivol_signals = pd.DataFrame()
    roc_ivol_signals = pd.DataFrame()

    # Vix signal
    filepath = 'data/VIX/VIX30D.csv'
    data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    rsi_vix = RSI(data).rename(columns={'value': 'vix_rsi'})
    roc_vix = ROC(data).rename(columns={'value': 'vix_roc'})
    save_technical_indicator(rsi_vix, 'vix_rsi')
    save_technical_indicator(roc_vix, 'vix_roc')

    # Stock signals
    filepath = 'data/yfinance/all_tickers_yfinance.csv'
    data_all_tic = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for tic in tickers:    # TODO METER LISTA COMPLETA DE TECHNICAL INDICATORS
        print('stock: ' + tic)
        data['close'] = data_all_tic[tic]
        rsi_tic = RSI(data).rename(columns={'value': 'stock_' + tic + '_rsi'})
        roc_tic = ROC(data).rename(columns={'value': 'stock_' + tic + '_roc'})
        rsi_stock_signals = pd.concat([rsi_stock_signals, rsi_tic], axis=1)
        roc_stock_signals = pd.concat([roc_stock_signals, roc_tic], axis=1)
    save_technical_indicator(rsi_stock_signals, 'stock_rsi')
    save_technical_indicator(roc_stock_signals, 'stock_roc')

    # Implied volatility
    filepath = 'data/implied_volatility/all_tickers_ivol.csv'
    data_all_tic = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for tic in tickers:    # TODO METER LISTA COMPLETA DE TECHNICAL INDICATORS
        print('ivol: ' + tic)
        data['close'] = data_all_tic[tic]
        rsi_tic = RSI(data).rename(columns={'value': 'ivol_' + tic + '_rsi'})
        roc_tic = ROC(data).rename(columns={'value': 'ivol_' + tic + '_roc'})
        rsi_ivol_signals = pd.concat([rsi_ivol_signals, rsi_tic], axis=1)
        roc_ivol_signals = pd.concat([roc_ivol_signals, roc_tic], axis=1)
    save_technical_indicator(rsi_ivol_signals, 'ivol_rsi')
    save_technical_indicator(roc_ivol_signals, 'ivol_roc')

    # Other signals


############################
#       extra methods      #
############################
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
    signal = signal.applymap(nan_to_50)
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
# TODO CORRIGIR A FUNÇÃO
def RSI(raw_signal, n=14):
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
        calc.at[calc.index[i], 'avg_up'] = (n - 1 * calc.iloc[i-1].at['avg_up'] + calc.iloc[i].at['up']) / n
        calc.at[calc.index[i], 'avg_down'] = (n - 1 * calc.iloc[i-1].at['avg_down'] + calc.iloc[i].at['down']) / n
        calc.at[calc.index[i], 'rsi'] = 100 - 100 / (1 + calc.iloc[i].at['avg_up'] / calc.iloc[i].at['avg_down'])

    signal = pd.DataFrame()
    signal['value'] = calc['rsi']
    return signal


def ROC(raw_signal, n=14):
    calc = raw_signal.copy()
    calc['value'] = np.nan

    for i in range(n, len(calc)):
        calc.at[calc.index[i], 'value'] = (calc.iloc[i]['close'] / calc.iloc[i-n]['close'] - 1) * 100
    calc = normalization(calc, -100, 100)
    signal = pd.DataFrame()
    signal['value'] = calc['value']
    return signal
