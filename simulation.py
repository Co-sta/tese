import genetic_algorithm_2 as ga2
import genetic_algorithm_1 as ga1
import pandas as pd
import trading_simulator as ts
import ui
import data
import pickle

# GA 2 - UPPER
ga2_pop_size = 10 # MEXER
ga2_chromo_size = 8
ga2_gene_size = 100000

ga2_n_parents = 4  # range [2 G] # MEXER
ga2_n_children = 7  # range [G-n_parents G]  # MEXER
ga2_crow_w = 0.5  # range [0 1] # MEXER
ga2_mutation_rate = 0.3  # range [0 1] # MEXER
ga2_mutation_std = 15000  # range [0 15000] # MEXER
ga2_method_1pop = 2  # 1st generation creation methods. [1,2,3] # MEXER
ga2_method_ps = 2  # parent selection methods. [1,2,3,4] # MEXER
ga2_method_crov = 5  # crossover methods. [1,2,3,4,5] # MEXER

# GA 1 - LOWER
ga1_pop_size = 100  # MEXER
ga1_chr_size = 15    # 7 INDICADORES PARA CADA EMPRESA + 7 GENES PARA O 'N' DE CADA INDICADOR +1 para a dist de forecast
ga1_gene_size = 100000

################################################################################
def train():
    best_pop = ga2.simulate(ga2_pop_size, ga2_chromo_size, ga2_gene_size, ga2_n_parents, ga2_n_children, ga2_crow_w,
                        ga2_mutation_rate, ga2_mutation_std, ga2_method_1pop, ga2_method_ps, ga2_method_crov,
                        ga1_pop_size, ga1_chr_size, ga1_gene_size, eval_start, eval_end)

    filepath = data.save_best(best_pop, time_period)
    return [best_pop, filepath]

def test(chromo, filepath=False):
    tickers = data.open_sp500_tickers_to_list()
    # tickers = data.open_all_sp500_tickers_to_list()
    [forecast, orders] = ga1.forecast_orders(chromo.get_gene_list(), tickers, ga1_chr_size, eval_start, eval_end)
    print(forecast)
    forecast.to_csv('forecast.csv')
    print(orders)
    portfolio = ts.trade(eval_start, eval_end, orders)
    data.save_portfolio(portfolio, time_period=time_period)
    print('ROI: '+ str(portfolio.get_ROI()['value'].iloc[-1]))



################################################################################
# 1st EVALUATION
# eval_start = pd.to_datetime('01-02-2011')   # COMECA SEMPRE UM DIA DEPOIS DE eval_star
# eval_end = pd.to_datetime('12-31-2011')
# time_period = '1st_period'

# 2nd EVALUATION
eval_start = pd.to_datetime('01-02-2012')   # COMECA SEMPRE UM DIA DEPOIS DE eval_star
eval_end = pd.to_datetime('12-31-2012')
time_period = '2nd_period'

################################################################################
# print('STARTING FULL TRAIN AND TEST')
# [best_chromo, filepath] = train()
# test(best_chromo, filepath=filepath)

################################################################################
# print('STARTING TEST')
# file = '1st_period-28-11-2020:02:03:50 AM.pickle'
# test_filepath = 'data/results/test/' + file
# train_filepath = 'data/results/train/' + file
# best_pop = pickle.load( open( train_filepath, "rb" ))
# best_chromo = best_pop.get_sub_pop().get_h_fame()[0]
# test(best_chromo, filepath=test_filepath)

################################################################################
# print('STARTING UI ENVIRONMENT')
# TRAIN
# ui.print_result('2nd_period-01-01-2021:09:12:17 AM.pickle', ga1_pop_size, ga1_gene_size)
# ui.graph_score('2nd_period-01-01-2021:09:12:17 AM.pickle')
# ui.graph_TI('2nd_period-01-01-2021:09:12:17 AM.pickle')
# ui.graph_forecast('2nd_period-01-01-2021:09:12:17 AM.pickle')
ui.graph_orders('2nd_period-01-01-2021:09:12:17 AM.pickle')

# SIGNALS
# ui.graph_IVol()
# ui.graph_Stocks()
# ui.graph_VIX()

# TEST
# ui.graph_ROI('2nd_period-01-01-2021:09:28:22 AM.pickle')
