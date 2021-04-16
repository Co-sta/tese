import genetic_algorithm_2 as ga2
import genetic_algorithm_1 as ga1
import pandas as pd
import trading_simulator as ts
import ui
import data
import pickle
from multiprocessing import Pool
from functools import partial

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
ga1_chr_size = 10    # 5 INDICADORES PARA CADA EMPRESA + 5 GENES PARA O 'N' DE CADA INDICADOR
ga1_gene_size = 100000

################################################################################
def train(start, end):
    best_pop = ga2.simulate(ga2_pop_size, ga2_chromo_size, ga2_gene_size, ga2_n_parents, ga2_n_children, ga2_crow_w,
                        ga2_mutation_rate, ga2_mutation_std, ga2_method_1pop, ga2_method_ps, ga2_method_crov,
                        ga1_pop_size, ga1_chr_size, ga1_gene_size, start, end)

    filepath = data.save_best(best_pop, start, end)
    return [best_pop, filepath]

def test(type, chromo, start, end):
    tickers = data.open_sp500_tickers_to_list()
    # tickers = data.open_all_sp500_tickers_to_list()
    [forecast, orders] = ga1.forecast_orders(chromo.get_gene_list(), tickers, ga1_chr_size, start, end)
    portfolio = ts.trade(start, end, orders, tickers, type)
    data.save_portfolio(portfolio, type, start, end)

################################################################################
train_period = {1:{'start': pd.to_datetime('01-02-2011'), 'end': pd.to_datetime('12-31-2011')},
                2:{'start': pd.to_datetime('01-02-2012'), 'end': pd.to_datetime('12-31-2012')},
                3:{'start': pd.to_datetime('01-02-2013'), 'end': pd.to_datetime('12-31-2013')}}

test_period = {1:{'start': pd.to_datetime('01-02-2012'), 'end': pd.to_datetime('12-31-2013')},
               2:{'start': pd.to_datetime('01-02-2013'), 'end': pd.to_datetime('12-31-2014')},
               3:{'start': pd.to_datetime('01-02-2014'), 'end': pd.to_datetime('12-31-2015')}}

################################################################################
# STARTING FULL TRAIN AND TEST
# types = [1,2,3,4]  # 1:long calls | 2:long puts | 3:short calls | 4:short puts
# periods = [1,3]
# for period in periods:
#     [best_chromo, filepath] = train(train_period[period]['start'], train_period[period]['end'])
#     for type in types:
#         test(best_chromo, type, test_period[period]['start'], test_period[period]['end'])

################################################################################
# STARTING TEST
period = 3
types = [1,2,3,4]  # 1:long calls | 2:long puts | 3:short calls | 4:short puts

filenames = {1:'(02-01-2011:12:00:00 AM)--(31-12-2011:12:00:00 AM).pickle',
             2:'(02-01-2012:12:00:00 AM)--(31-12-2012:12:00:00 AM).pickle',
             3:'(02-01-2013:12:00:00 AM)--(31-12-2013:12:00:00 AM).pickle'}
file = filenames[period]
train_filepath = 'data/results/train/' + file
best_pop = pickle.load( open( train_filepath, "rb" ))
best_chromo = best_pop.get_sub_pop().get_h_fame()[0]

for type in types:
    test(type, best_chromo, test_period[period]['start'], test_period[period]['end'])

################################################################################
# STARTING UI ENVIRONMENT

# TRAIN
# train_filename = '(02-01-2011:12:00:00 AM)--(31-12-2011:12:00:00 AM).pickle'
# ui.print_result(train_filename, ga1_pop_size, ga1_gene_size)
# ui.print_train_stats(train_filename)
# ui.graph_score(train_filename)
# ui.graph_TI(train_filename)
# ui.graph_forecast(train_filename)
# ui.graph_orders(train_filename)
# ui.graph_orders_correct_orders(train_filename)
# ui.graph_forecast_ivol(train_filename)

# SIGNALS
# ui.graph_IVol()
# ui.graph_smooth_IVol(12)
# ui.graph_Stocks()
# ui.graph_VIX()

# TEST
# test_filename = '(02-01-2013:12:00:00 AM)--(31-12-2014:12:00:00 AM)--(4).pickle'
# period = 2
# eval_start = test_period[period]['start']
# eval_end = test_period[period]['end']
# ui.graph_ROI(test_filename)
# ui.graph_CAPITAL(test_filename)
# ui.graph_holdings(test_filename)
# ui.graph_trades(test_filename)
# ui.print_nr_trades(test_filename)
# ui.options_graph(test_filename, eval_start, eval_end)

# EXTRA
