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
ga1_chr_size = 10    # 5 INDICADORES PARA CADA EMPRESA + 5 GENES PARA O 'N' DE CADA INDICADOR
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
    portfolio = ts.trade(eval_start, eval_end, orders, tickers)
    data.save_portfolio(portfolio, time_period=time_period)

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
# STARTING FULL TRAIN AND TEST
[best_chromo, filepath] = train()
test(best_chromo, filepath=filepath)

################################################################################
# STARTING TEST
file = '2nd_period-01-02-2021:03:22:37 AM.pickle'
# test_filepath = 'data/results/test/' + file
# train_filepath = 'data/results/train/' + file
# best_pop = pickle.load( open( train_filepath, "rb" ))
# best_chromo = best_pop.get_sub_pop().get_h_fame()[0]
# test(best_chromo, filepath=test_filepath)

################################################################################
# STARTING UI ENVIRONMENT
# TRAIN
train_filename = '2nd_period-22-02-2021:10:25:57 PM.pickle'
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
test_filename = '2nd_period-22-02-2021:10:33:33 PM.pickle'
# ui.graph_ROI(test_filename)
# ui.graph_trades(test_filename)
# ui.print_nr_trades(test_filename)
# ui.options_graph(test_filename, eval_start, eval_end)
