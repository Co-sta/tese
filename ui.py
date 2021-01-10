import genetic_algorithm_2 as ga2
import genetic_algorithm_1 as ga1
import data as data
import pandas as pd
import technical_indicators as ti
import pickle

import plotly.express as px
import plotly.graph_objects as go

###########################
#          TRAIN          #
###########################
def print_result(filename, ga1_pop_size, ga1_gene_size):
    print('printing results...')
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    genes2 = best_chromo.get_gene_list()
    [ga1_n_parents,ga1_n_children,ga1_crov_w,ga1_mutation_rate,
    ga1_mutation_std,ga1_method_1pop,ga1_method_ps,ga1_method_crov] = \
    ga2.unnorm(genes2, ga1_pop_size, ga1_gene_size)

    print('\nG2-Chromossome :')
    print('   score=' + str(best_chromo.get_score()) + '\n')
    print('    N parents: ' + str(ga1_n_parents) + ' (' + str(genes2[0].get_value()) + ')')
    print('   N children: ' + str(ga1_n_children) + ' (' + str(genes2[1].get_value()) + ')')
    print('       Crov w: ' + str(ga1_crov_w) + ' (' + str(genes2[2].get_value()) + ')')
    print(' Mutaion rate: ' + str(ga1_mutation_rate) + ' (' + str(genes2[3].get_value()) + ')')
    print(' Mutation std: ' + str(ga1_mutation_std) + ' (' + str(genes2[4].get_value()) + ')')
    print('    First pop: ' + str(ga1_method_1pop) + ' (' + str(genes2[5].get_value()) + ')')
    print(' Parent selct: ' + str(ga1_method_ps) + ' (' + str(genes2[6].get_value()) + ')')
    print('    Crossover: ' + str(ga1_method_crov) + ' (' + str(genes2[7].get_value()) + ')\n')
    for i in range(len(best_chromo.get_sub_pop().get_h_fame()[0].get_gene_list())):
        chr1 = best_chromo.get_sub_pop().get_h_fame()[0].get_gene_list()[i]
        if i == 0: print('\n      stock_rsi_weight: ' + str(chr1.get_value()))
        if i == 1: print('      stock_roc_weight: ' + str(chr1.get_value()))
        if i == 2: print('      stock_sto_weight: ' + str(chr1.get_value()))
        if i == 3: print('       ivol_rsi_weight: ' + str(chr1.get_value()))
        if i == 4: print('       ivol_roc_weight: ' + str(chr1.get_value()))
        if i == 5: print('       ivol_sto_weight: ' + str(chr1.get_value()))
        if i == 6: print('         ivol_macd_roc: ' + str(chr1.get_value()))

        if i == 7: print('           n_stock_rsi: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 8: print('           n_stock_roc: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 9: print('           n_stock_sto: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 10: print('            n_ivol_rsi: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 11: print('            n_ivol_roc: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 12: print('            n_ivol_sto: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 13: print('           n_ivol_macd: ' + str(ga1.unnorm_ti(chr1.get_value())))

        if i == 14: print('          n_prediction: ' + str(ga1.unnorm_ti(chr1.get_value())))

def print_train_stats(filename):
    print('printing train statistics...')
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))

    nr_trading_days =  best_chromo.get_sub_pop().get_h_fame()[0].nr_trading_days
    nr_correct_days = best_chromo.get_sub_pop().get_h_fame()[0].nr_correct_days
    nr_up_days = best_chromo.get_sub_pop().get_h_fame()[0].nr_up_days
    nr_stay_days = best_chromo.get_sub_pop().get_h_fame()[0].nr_stay_days
    nr_down_days = best_chromo.get_sub_pop().get_h_fame()[0].nr_down_days
    nr_correct_ups = best_chromo.get_sub_pop().get_h_fame()[0].nr_correct_ups
    nr_correct_stays = best_chromo.get_sub_pop().get_h_fame()[0].nr_correct_stays
    nr_correct_downs = best_chromo.get_sub_pop().get_h_fame()[0].nr_correct_downs

    print('------Trading Days: ' + str(nr_trading_days))
    print('-----------Up Days: ' + str(nr_up_days))
    print('---------Stay Days: ' + str(nr_stay_days))
    print('---------Down Days: ' + str(nr_down_days))
    print('------Correct Days: ' + str(nr_correct_days))
    print('---Correct Up Days: ' + str(nr_correct_ups))
    print('-Correct Stay Days: ' + str(nr_correct_stays))
    print('-Correct Down Days: ' + str(nr_correct_downs))

    print('---% of Correct Up Days: ' + str(100*nr_correct_ups/nr_up_days) + ' %')
    print('-% of Correct Stay Days: ' + str(100*nr_correct_stays/nr_stay_days) + ' %')
    print('-% of Correct Down Days: ' + str(100*nr_correct_downs/nr_down_days) + ' %')

def graph_score(filename):
    print('printing score...')
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    score_evol =  best_chromo.get_sub_pop().get_max_score()
    fig = px.line(score_evol, x="epoch", y="score")
    fig.show()

def graph_TI(filename):
    print('printing technical indicators graphs...')
    tickers = data.open_sp500_tickers_to_list()
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    chromo = best_chromo.get_sub_pop().get_h_fame()[0]
    gene_list = chromo.get_gene_list()

    n_fp_stock_rsi = ga1.unnorm_ti(gene_list[-8].get_value())
    n_fp_stock_roc = ga1.unnorm_ti(gene_list[-7].get_value())
    n_fp_stock_sto = ga1.unnorm_ti(gene_list[-6].get_value())
    n_fp_ivol_rsi = ga1.unnorm_ti(gene_list[-5].get_value())
    n_fp_ivol_roc = ga1.unnorm_ti(gene_list[-4].get_value())
    n_fp_ivol_sto = ga1.unnorm_ti(gene_list[-3].get_value())
    n_fp_ivol_macd = ga1.unnorm_ti(gene_list[-2].get_value())


    fp_stock_rsi = 'data/technical_indicators/' + str(n_fp_stock_rsi) + '_stock_rsi.csv'
    fp_stock_roc = 'data/technical_indicators/' + str(n_fp_stock_roc) + '_stock_roc.csv'
    fp_stock_sto = 'data/technical_indicators/' + str(n_fp_stock_sto) + '_stock_sto.csv'
    fp_ivol_rsi = 'data/technical_indicators/' + str(n_fp_ivol_rsi) + '_ivol_rsi.csv'
    fp_ivol_roc = 'data/technical_indicators/' + str(n_fp_ivol_roc) + '_ivol_roc.csv'
    fp_ivol_sto = 'data/technical_indicators/' + str(n_fp_ivol_sto) + '_ivol_sto.csv'
    fp_ivol_macd = 'data/technical_indicators/' + str(n_fp_ivol_macd) + '_ivol_macd.csv'

    stock_rsi = pd.read_csv(fp_stock_rsi, index_col='Date', parse_dates=True)
    stock_roc = pd.read_csv(fp_stock_roc, index_col='Date', parse_dates=True)
    stock_sto = pd.read_csv(fp_stock_sto, index_col='Date', parse_dates=True)
    ivol_rsi = pd.read_csv(fp_ivol_rsi, index_col='Date', parse_dates=True)
    ivol_roc = pd.read_csv(fp_ivol_roc, index_col='Date', parse_dates=True)
    ivol_sto = pd.read_csv(fp_ivol_sto, index_col='Date', parse_dates=True)
    ivol_macd = pd.read_csv(fp_ivol_macd, index_col='Date', parse_dates=True)

    ti_signals = pd.concat([stock_rsi, stock_roc, stock_sto, ivol_rsi,
                                ivol_roc, ivol_sto, ivol_macd], axis=1)

    for ticker in tickers:
        fig = px.line(ti_signals,
                      x=ti_signals.index,
                      y=['stock_' + ticker + '_rsi', 'stock_' + ticker + '_roc',
                         'stock_' + ticker + '_sto', 'ivol_' + ticker + '_rsi',
                         'ivol_' + ticker + '_roc',  'ivol_' + ticker + '_sto',
                         'ivol_' + ticker + '_macd'],
                      title="Technical Indicators")
        fig.data[0].name = 'stock_' + ticker + '_rsi (' + str(n_fp_stock_rsi) + ')'
        fig.data[1].name = 'stock_' + ticker + '_roc (' + str(n_fp_stock_roc) + ')'
        fig.data[2].name = 'stock_' + ticker + '_sto (' + str(n_fp_stock_sto) + ')'
        fig.data[3].name = 'ivol_' + ticker + '_rsi (' + str(n_fp_ivol_rsi) + ')'
        fig.data[4].name = 'ivol_' + ticker + '_roc (' + str(n_fp_ivol_roc) + ')'
        fig.data[5].name = 'ivol_' + ticker + '_sto (' + str(n_fp_ivol_sto) + ')'
        fig.data[6].name = 'ivol_' + ticker + '_macd (' + str(n_fp_ivol_macd) + ')'


        fig.show()

def graph_forecast(filename):
    fig = go.Figure()
    print('printing forecast...')
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    forecast =  best_chromo.get_sub_pop().get_h_fame()[0].forecast
    for i in range(len(forecast)):
        fig.add_trace(go.Scatter(x=forecast.columns.tolist(), y=forecast.iloc[i].tolist()))
        fig.data[i].name = forecast.index[i]
    fig.show()

def graph_orders(filename):
    fig = go.Figure()
    print('printing orders...')
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    orders =  best_chromo.get_sub_pop().get_h_fame()[0].orders
    for i in range(len(orders)):
        fig.add_trace(go.Scatter(x=orders.columns.tolist(), y=orders.iloc[i].tolist()))
        fig.data[i].name = orders.index[i]
    fig.show()

def graph_forecast_ivol(filename):
    fig = go.Figure()
    filepath = 'data/results/train/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    forecast =  best_chromo.get_sub_pop().get_h_fame()[0].forecast
    for i in range(len(forecast)):
        fig.add_trace(go.Scatter(x=forecast.columns.tolist(), y=forecast.iloc[i].tolist()))
        fig.data[i].name = 'forecasted IVol: ' + forecast.index[i]

    tickers = data.open_sp500_tickers_to_list()
    filepath = 'data/implied_volatility/all_tickers_ivol.csv'
    iv_signals = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for ticker in tickers:
        i += 1
        fig.add_trace(go.Scatter(x=iv_signals.index, y=iv_signals[ticker]))
        fig.data[i].name = 'real IVol: ' + ticker
    fig.show()


###########################
#         SIGNALS         #
###########################
def graph_IVol():
    fig = go.Figure()
    cnt = 0
    tickers = data.open_sp500_tickers_to_list()
    filepath = 'data/implied_volatility/all_tickers_ivol.csv'
    iv_signals = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for ticker in tickers:
        fig.add_trace(go.Scatter(x=iv_signals.index, y=iv_signals[ticker]))
        fig.data[cnt].name = ticker
        cnt += 1
    fig.show()

def graph_Stocks():
    tickers = data.open_sp500_tickers_to_list()
    filepath = 'data/yfinance/all_tickers_yfinance.csv'
    stock_signals = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    for ticker in tickers:
        fig = px.line(stock_signals, x=stock_signals.index, y=ticker,
                      title="Stock Value of " + ticker)
        fig.show()

def graph_VIX():
    filepath = 'data/VIX/VIX_all.csv'
    vix_signals = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    fig = px.line(vix_signals, x=vix_signals.index, y=['9 days', '30 days', '3 months', '6 months'], title='Vix signals')
    fig.show()


##########################
#          TEST          #
##########################
def graph_ROI(filename):
    print('printing ROI graph...')
    filepath = 'data/results/test/' + filename
    port = pickle.load( open( filepath, "rb" ))
    roi_evol = port.get_ROI()
    fig = px.line(roi_evol, x=roi_evol.index, y="value", title=' Rate of Income (ROI)')
    fig.show()

def graph_trades(filename):
    print('printing trades...')
    filepath = 'data/results/test/' + filename
    port = pickle.load( open( filepath, "rb" ))
    for ticker in port.get_tickers():
        data = []
        layout= go.Layout(title=go.layout.Title(text=ticker + ' Trades:'),
                xaxis={'title':'option value'},
                yaxis={'title':'date'})
        for daily_transactions in port.get_log().values():
            for txn in daily_transactions:
                if txn.get_company() == ticker and txn.get_result() == 'positive':
                    line = go.Scatter(x=[txn.get_init_date(), txn.get_final_date()],
                                      y=[txn.get_init_value(), txn.get_final_value()],
                                      name="Positive",
                                      legendgroup="Positive")
                else:
                    line = go.Scatter(x=[txn.get_init_date(), txn.get_final_date()],
                                      y=[txn.get_init_value(), txn.get_final_value()],
                                      name="Negative",
                                      legendgroup="Negative")
                data.append(line)
        go.Figure(data=point_plot, layout=layout1).show()

def print_nr_trades(filename):
    print('printing nr of positive and negative trades...')
    filepath = 'data/results/test/' + filename
    port = pickle.load( open( filepath, "rb" ))
    nr_pos_trades = port.get_nr_pos_trades()
    nr_neg_trades = port.get_nr_neg_trades()
    print('\n   Nr of total trades: ' + str(nr_pos_trades+nr_neg_trades))
    print('\n   Nr of positive trades: ' + str(nr_pos_trades))
    print('\n   Nr of negative trades: ' + str(nr_neg_trades))
