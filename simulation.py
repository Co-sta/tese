import genetic_algorithm_2 as ga2
import pandas as pd
import ui
import data

# GA 2 - UPPER
ga2_pop_size = 10 # MEXER
ga2_chromo_size = 8
ga2_gene_size = 100000

ga2_n_parents = 6  # range [2 G] # MEXER
ga2_n_children = 8  # range [G-n_parents G]  # MEXER
ga2_crow_w = 0.5  # range [0 1] #M EXER
ga2_mutation_rate = 0.8  # range [0 1] # MEXER
ga2_mutation_std = 10000  # range [0 15000] # MEXER
ga2_method_1pop = 1  # 1st generation creation methods. [1,2,3] # MEXER
ga2_method_ps = 2  # parent selection methods. [1,2,3,4] # MEXER
ga2_method_crov = 2  # crossover methods. [1,2,3,4,5] # MEXER


# GA 1 - LOWER
ga1_pop_size = 10  # MEXER
ga1_chr_size = 4
ga1_gene_size = 100000


# 1st EVALUATION
# eval_start = pd.to_datetime('01-02-2011')   # COMECA SEMPRE UM DIA DEPOIS DE eval_star
# eval_end = pd.to_datetime('12-31-2011')
# time_period = '1st_period'

# 2nd EVALUATION
eval_start = pd.to_datetime('01-02-2012')   # COMECA SEMPRE UM DIA DEPOIS DE eval_star
eval_end = pd.to_datetime('12-31-2012')
time_period = '2nd_period'



def tese():
    best_chromo = ga2.simulate(ga2_pop_size, ga2_chromo_size, ga2_gene_size, ga2_n_parents, ga2_n_children, ga2_crow_w,
                        ga2_mutation_rate, ga2_mutation_std, ga2_method_1pop, ga2_method_ps, ga2_method_crov,
                        ga1_pop_size, ga1_chr_size, ga1_gene_size, eval_start, eval_end)

    data.save_best(best_chromo, time_period)

# tese()
ui.print_result('1st_period-01:19:40 AM:12-10-2020.pickle', ga1_pop_size, ga1_gene_size)
ui.graph_score('1st_period-01:19:40 AM:12-10-2020.pickle')
