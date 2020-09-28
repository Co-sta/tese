import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import data as d


def compute_technical_signals():
    tickers = d.open_sp500_tickers_to_list()

    # Vix signal
    filename = 'data/VIX/VIX30D.csv'
    data = pd.read_csv(filename, index_col='date', parse_dates=True)
    rsi_vix = RSI(data).rename(columns={'value': 'vix_rsi'})
    roc_vix = ROC(data).rename(columns={'value': 'vix_roc'})
    tec_signals = pd.concat([rsi_vix, roc_vix], axis=1)

    # Stock signals
    data_all_tic = yf.download(tickers, start="2011-01-03", end="2016-01-01")
    data_all_tic = data_all_tic.drop(['Low', 'Adj Close', 'Open', 'Volume', 'High'], axis=1)
    data_all_tic.columns = data_all_tic.columns.get_level_values(1)
    for tic in tickers:    # TODO METER LISTA COMPLETA DE TECHNICAL INDICATORS
        print(tic)
        data['close'] = data_all_tic[tic]
        rsi_tic = RSI(data).rename(columns={'value': tic + '_rsi'})
        roc_tic = ROC(data).rename(columns={'value': tic + '_roc'})
        tec_signals = pd.concat([tec_signals, rsi_tic, roc_tic], axis=1)

    # Other signals


    tec_signals = tec_signals.applymap(nan_to_50)
    save_technical_indicators(tec_signals)
    print(tec_signals.to_string())
    return tec_signals


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


def save_technical_indicators(signal):
    filepath = 'data/technical_indicators/technical_indicators.pickle'
    with open(filepath, 'wb') as f:
        pickle.dump(signal, f)


def load_technical_indicators():
    filepath = 'data/technical_indicators/technical_indicators.pickle'
    with open(filepath, 'rb') as f:
        signal = pickle.load(f)
    return signal


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
    calc.at[calc.index[n], 'rsi'] = 100 - 100 / (1 + calc.iloc[n].at['avg_up'] / calc.iloc[n].at['avg_down'])

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


compute_technical_signals()
