import random
import numpy as np
import pandas as pd
import trading_simulator as ts
from random import randrange
import math
from math import floor
from math import sqrt
from scipy.stats import truncnorm
from copy import deepcopy
import data
import time


# o que interessa da implied volatility é a varaiação absoludta e não a variação precentual a implied volatility

def unnorm_ti(ti_norm, n_min=5, n_max=60):
    pos_ti = np.arange(n_min, n_max + 1)
    step_ti = (GENE_SIZE + 1) / len(pos_ti)
    i = int(np.floor(ti_norm / step_ti))
    return int(pos_ti[i])

def unnorm_xema(ti_norm):
    short_n = np.arange(2, 19)
    long_n = np.arange(20, 101, 5)
    pos_ti = []
    for i in short_n:
        for j in long_n:
            pos_ti.append([i,j])
    step_ti = (GENE_SIZE + 1) / len(pos_ti)
    i = int(np.floor(ti_norm / step_ti))
    return pos_ti[i]


############################
#     Global Variables     #
############################
GENE_SIZE = 100000
END_VALUE = 0.9  # MEXER
MAX_N_GEN = 100  # max of generations per simulation # MEXER
N_TOP = 5 # MEXER
MAX_NO_EVOL = 15 # MEXERNO_EVOL_STD_INCREASE
NO_EVOL_STD_INCREASE = 5 # MEXER
DEFAULT_MUTATION_STD = 1 # initialization

H_FAME_SIZE = 5 # MEXER

N_PARENTS = 1  # initialization
N_CHILDREN = 1  # initialization
CROV_W = 1.0  # initialization
MUTATION_RATE = 1.0  # initialization
MUTATION_STD = 1  # initialization
NO_EVOL_STD_INCREASE
METHOD_1POP = 1  # initialization
METHOD_PS = 1  # initialization
METHOD_CROV = 1  # initialization

FORECAST_DIST = 10 # forecast at 10 days
IVOL_CHANGE_STEP = 2  # ivol minimum change to consider change # TODO PERGUNTAR AO RUI NEVES SE É ESTE O VALOR


def set_global_var(gene_size, n_parents, n_children, crow_w, mutation_rate,
                   mutation_std, method_1pop, method_ps, method_crov):
    global GENE_SIZE
    GENE_SIZE = gene_size
    global N_PARENTS
    N_PARENTS = n_parents
    global N_CHILDREN
    N_CHILDREN = n_children
    global CROV_W
    CROV_W = crow_w
    global MUTATION_RATE
    MUTATION_RATE = mutation_rate
    global MUTATION_STD
    MUTATION_STD = mutation_std
    global DEFAULT_MUTATION_STD
    DEFAULT_MUTATION_STD = mutation_std
    global METHOD_1POP
    METHOD_1POP = method_1pop
    global METHOD_PS
    METHOD_PS = method_ps
    global METHOD_CROV
    METHOD_CROV = method_crov


def simulate(pop_size, chromo_size, gene_size, n_parents, n_children, crow_w,
             mutation_rate, mutation_std, method_1pop,
             method_ps, method_crov, eval_start, eval_end):
    set_global_var(gene_size, n_parents, n_children, crow_w, mutation_rate,
                   mutation_std, method_1pop, method_ps, method_crov)
    pop = Population(pop_size, chromo_size)
    epoch = 1
    while True:
        print('G1 generation nr: ' + str(pop.get_generation() + 1))
        pop.evaluation_phase(eval_start, eval_end, use_trading=False)
        pop.update_h_fame()
        end = pop.check_end_phase()

        if end:
            return pop
        else:
            pop.max_score = pop.max_score.append({'epoch':epoch, 'score':pop.get_h_fame()[0].get_score()}, ignore_index=True)
            print(pop.max_score)
            pop.parent_selection_phase()
            # print('N_PAreNTS: ' + str(N_PARENTS))
            # print('n_parents: ' + str(len(pop.get_parents())))
            pop.crossover_phase()
            pop.mutation_phase()
            pop.increase_gen()
            epoch += 1


def get_GENE_SIZE():
    global GENE_SIZE
    return GENE_SIZE
def get_END_VALUE():
    global END_VALUE
    return END_VALUE
def get_MAX_N_GEN():
    global MAX_N_GEN
    return MAX_N_GEN
def get_N_TOP():
    global N_TOP
    return N_TOP
def get_N_PARENTS():
    global N_PARENTS
    return N_PARENTS
def get_N_CHILDREN():
    global N_CHILDREN
    return N_CHILDREN
def get_CROV_W():
    global CROV_W
    return CROV_W
def get_MUTATION_RATE():
    global MUTATION_RATE
    return MUTATION_RATE
def get_MUTATION_STD():
    global MUTATION_STD
    return MUTATION_STD
def get_DEFAULT_MUTATION_STD():
    global DEFAULT_MUTATION_STD
    return DEFAULT_MUTATION_STD
def get_MAX_NO_EVOL():
    global MAX_NO_EVOL
    return MAX_NO_EVOL
def get_NO_EVOL_STD_INCREASE():
    global NO_EVOL_STD_INCREASE
    return NO_EVOL_STD_INCREASE
def get_IVOL_CHANGE_STEP():
    global IVOL_CHANGE_STEP
    return IVOL_CHANGE_STEP

class Gene:
    def __init__(self, value):
        self.value = value

    ###########################
    #     general methods     #
    ###########################
    def get_value(self):
        return self.value
    def set_value(self, value):
        self.value = value

    ###########################
    #     custom methods      #
    ###########################
    def mutate(self):
        lower = 0
        upper = get_GENE_SIZE()
        std = get_MUTATION_STD()
        mean = self.value
        a, b = (lower - mean) / std, (upper - mean) / std
        new_value = int(truncnorm.rvs(a, b, loc=mean, scale=std, size=1))
        self.set_value(new_value)


class Chromosome:
    def __init__(self, gene_list):
        self.gene_list = gene_list
        self.size = len(gene_list)
        self.score = 0
        self.forecast = pd.DataFrame()
        self.orders = pd.DataFrame()
        self.nr_trading_days = 0
        self.nr_correct_days = 0
        self.nr_up_days = 0
        self.nr_stay_days = 0
        self.nr_down_days = 0
        self.nr_correct_ups = 0
        self.nr_correct_stays = 0
        self.nr_correct_downs = 0

    ###########################
    #     general methods     #
    ###########################
    def set_score(self, score):
        self.score = score
    def get_score(self):
        return self.score
    def get_size(self):
        return self.size
    def get_gene_list(self):
        return self.gene_list

    ###########################
    #     custom methods      #
    ###########################


class Population:
    def __init__(self,
                 pop_size,
                 chr_size):

        self.crov_w = CROV_W
        self.method_1pop = METHOD_1POP
        self.method_ps = METHOD_PS
        self.method_crov = METHOD_CROV

        self.pop_size = pop_size
        self.chromo_size = chr_size
        self.mutation_rate = MUTATION_RATE

        self.arg_verification()
        self.chromo_list = self.pop_generation_phase()

        self.parents = []
        self.h_fame = []  # a hall of fame with top 10 all time best
        self.no_evolution = 0
        self.generation = 0

        self.max_score = pd.DataFrame(columns=['epoch', 'score'])

    ###########################
    #     general methods     #
    ###########################
    def arg_verification(self):
        if not 0 <= self.crov_w <= 1:
            raise Exception('CROV_W has to be in between 0 and 1. Right now is' + str(self.crov_w))
        if not 1 <= self.method_1pop <= 3:
            raise Exception('METHOD_1POP has to be 1,2 or 3. Right now is' + str(self.method_1pop))
        if not 1 <= self.method_ps <= 4:
            raise Exception('METHOD_PS has to be 1,2,3 or 4 Right now is' + str(self.method_ps))
        if not 1 <= self.method_crov <= 5:
            raise Exception('METHOD_CROV has to be 1,2,3,4 or 5. Right now is' + str(self.method_crov))
        if self.pop_size < 0:
            raise Exception('Population size has to be greater than 0. Right now is' + str(self.pop_size))
        if self.chromo_size < 0:
            raise Exception('Chromosome size has to be greater than 0. Right now is' + str(self.chromo_size))
        if not 0 <= self.mutation_rate <= 1:
            raise Exception('mutation rate has to be between 0 and 1. Right now is' + str(self.mutation_rate))
    def get_pop_size(self):
        return self.pop_size
    def get_chr_size(self):
        return self.chromo_size
    def get_chromo_list(self):
        return self.chromo_list
    def set_chromo_list(self, new_gen):
        self.chromo_list = new_gen
    def print_chromo_list(self):
        for i, chromo in zip(range(self.get_pop_size()), self.get_chromo_list()):
            ch_str = ''
            print('Gene:' + str(i) + ' | score:' + str(chromo.get_score()))
            for ge in chromo.get_gene_list():
                ch_str = ch_str + ' [' + str(ge.value) + ']'
            print(ch_str)
    def get_mutation_rate(self):
        return self.mutation_rate
    def get_crov_w(self):
        return self.crov_w
    def get_method_1pop(self):
        return self.method_1pop
    def get_method_ps(self):
        return self.method_ps
    def get_method_crov(self):
        return self.method_crov
    def get_parents(self):
        return self.parents
    def print_parents(self):
        for parent in self.get_parents():
            print(parent.get_score())
    def get_h_fame(self):
        return self.h_fame
    def set_h_fame(self, h_fame):
        self.h_fame = h_fame
    def print_h_fame(self):
        print('HALL OF FAME')
        for chromo in self.get_h_fame():
            print(chromo.get_score())
    def get_no_evol(self):
        return self.no_evolution
    def incr_no_evol(self):
        self.no_evolution = self.get_no_evol() + 1
    def reset_no_evol(self):
        self.no_evolution = 0
    def get_generation(self):
        return self.generation
    def increase_gen(self):
        gen = self.get_generation()
        self.generation = gen + 1
    def get_max_score(self):
        return self.max_score

    ###########################
    # pop generation methods  #
    ###########################
    def gen_random(self):
        chromo_list = []
        for i in range(self.get_pop_size()):
            gene_list = []
            for j in range(self.get_chr_size()):
                value = randrange(0, get_GENE_SIZE() + 1)
                ge = Gene(value)  # each gene value varies between 0 and 100 000
                gene_list.append(ge)
            chromo = Chromosome(gene_list)
            chromo_list.append(chromo)
        return chromo_list

    def gen_sequential(self):
        chromo_list = []
        for i in range(self.get_pop_size()):
            gene_list = []
            for j in range(self.get_chr_size()):
                value = floor((i + 1) * get_GENE_SIZE() / self.get_pop_size())
                ge = Gene(value)  # each gene value varies between 0 and 100 000
                gene_list.append(ge)
            chromo = Chromosome(gene_list)
            chromo_list.append(chromo)
        return chromo_list

    def gen_parallel(self):
        chromo_list = []
        step = floor(get_GENE_SIZE() / self.get_pop_size())
        for i in range(self.get_pop_size()):
            gene_list = []
            for j in range(self.get_chr_size()):
                smin = i * step
                smax = (i + 1) * step
                value = randrange(smin, smax)
                ge = Gene(value)  # each gene varies between 0 and 100 000
                gene_list.append(ge)
            chromo = Chromosome(gene_list)
            chromo_list.append(chromo)
        return chromo_list

    ############################
    # parent selection methods #
    ############################
    def ps_top(self):  # the parents are the chromosomes with best scores
        parents = []
        chromo_list = self.get_chromo_list().copy()

        for i in range(get_N_PARENTS()):
            index = 0
            score = 0
            for j in range(len(chromo_list)):
                if chromo_list[j].score > score:
                    score = chromo_list[j].score
                    index = j
            parents.append(chromo_list.pop(index))
        return parents

    def ps_roullete(self):  # the parents are selected by roullete method
        chromo_list = self.get_chromo_list().copy()
        scores = []
        sum_scores = 0
        parents = []

        for i in range(len(chromo_list)):
            sum_scores = sum_scores + chromo_list[i].get_score()
        for i in range(len(chromo_list)):
            scores.append(chromo_list[i].score / sum_scores)

        parents_i = np.random.choice(self.get_pop_size(), get_N_PARENTS(), replace=False, p=scores).tolist()
        for i in parents_i:
            parents.append(chromo_list[i])
        return parents

    def ps_roullete_top(self):  # ntop parents are the top scores and the others are chosen by roullete
        ntop = get_N_TOP()
        parents = []
        chromo_list = self.get_chromo_list().copy()
        scores = []
        sum_scores = 0

        # top method
        for i in range(ntop):
            index = 0
            score = 0
            for j in range(len(chromo_list)):
                if chromo_list[j].score > score:
                    score = chromo_list[j].score
                    index = j
            parents.append(chromo_list.pop(index))

        # roullete method
        if (get_N_PARENTS() - ntop > 0):
            for i in range(len(chromo_list)):
                sum_scores = sum_scores + chromo_list[i].get_score()
            try:
                for i in range(len(chromo_list)):
                    scores.append(chromo_list[i].score / sum_scores)
                parents_i = (np.random.choice(len(chromo_list), get_N_PARENTS() - ntop, replace=False, p=scores).tolist())
            except:
                print('Fewer non-zero entries in p than size (há demasiados poucos pais com score diferente de zero)')
            else:
                parents_i = (np.random.choice(len(chromo_list), get_N_PARENTS() - ntop, replace=False).tolist())
            for i in parents_i:
                parents.append(chromo_list[i])
        return parents

    def ps_tournament(self):
        chromo_list = self.get_chromo_list().copy()
        tournament_size = 10
        if tournament_size < self.get_pop_size():
            tournament_size = self.get_pop_size()
        tournament_list = []
        parents = []

        for i in range(tournament_size):
            tournament_list.append(chromo_list.pop(randrange(len(chromo_list))))
        for i in range(get_N_PARENTS()):
            index = 0
            score = 0
            for j in range(len(tournament_list)):
                if tournament_list[j].score > score:
                    score = tournament_list[j].score
                    index = j
            parents.append(tournament_list.pop(index))
        return parents

    ############################
    #   hall of fame methods   #
    ############################
    def update_h_fame(self):
        h_fame_size = H_FAME_SIZE
        chromo_list = self.get_chromo_list().copy()
        h_fame = self.get_h_fame().copy()
        if h_fame:
            old_best_score = h_fame[0].get_score()
        else:
            old_best_score = 0
        for chromo_h_fame in h_fame:
            for chromo in chromo_list:
                if check_same_chromo(chromo_h_fame, chromo):
                    print('same')
                    chromo_list.remove(chromo)
        #  TODO CONTINUAR AQUI
        # for chromo in chromo_list:
        #     print('score: ' + str(chromo.get_score()))
        # print('------------------------------------')
        chromo_list.sort(key=lambda x: x.score, reverse=True)
        # for chromo in chromo_list:
        #     print('score ordenado: ' + str(chromo.get_score()))
        # print('------------------------------------')
        h_fame.extend(chromo_list)
        # for chromo in h_fame:
        #     print('h_fame + list: ' + str(chromo.get_score()))
        # print('------------------------------------')
        h_fame.sort(key=lambda x: x.score, reverse=True)
        # for chromo in h_fame:
        #     print('h_fame + list ordenado: ' + str(chromo.get_score()))
        # print('------------------------------------')
        new_h_fame = deepcopy(h_fame[0:h_fame_size])
        for chromo in new_h_fame:
            print('H_FAME: ' + str(chromo.get_score()))
        print('------------------------------------')
        self.print_chromo_list()
        new_best_score = new_h_fame[0].get_score()

        if new_best_score > old_best_score:
            print(self.no_evolution)
            self.reset_no_evol()
            print('RESET')
            print(self.no_evolution)

        else:
            print(self.no_evolution)
            print('INCRESE')
            self.incr_no_evol()
            print(self.no_evolution)
        self.set_h_fame(new_h_fame)

    ############################
    #          phases          #
    ############################
    def mutation_phase(self):
        for chromo in self.get_chromo_list():
            for gene in chromo.get_gene_list():
                if get_MUTATION_RATE() > random.random():
                    gene.mutate()

    def pop_generation_phase(self):
        method_1pop = self.get_method_1pop()
        method = self.gen_random  # to eliminate warnings
        if method_1pop == 1:
            method = self.gen_random
        elif method_1pop == 2:
            method = self.gen_sequential
        elif method_1pop == 3:
            method = self.gen_parallel
        pop = method()
        return pop

    def parent_selection_phase(self):
        method_ps = self.get_method_ps()
        method = self.ps_tournament  # to eliminate warnings
        if method_ps == 1:
            method = self.ps_top
        elif method_ps == 2:
            method = self.ps_roullete
        elif method_ps == 3:
            method = self.ps_roullete_top
        elif method_ps == 4:
            method = self.ps_tournament
        self.parents = method()

    def crossover_phase(self):
        method_crov = self.get_method_crov()
        method = crov_2points  # to eliminate warnings
        parents = self.get_parents().copy()
        new_gen = []

        if method_crov == 1:
            method = crov_1point
        elif method_crov == 2:
            method = crov_2points
        elif method_crov == 3:
            method = crov_geometric
        elif method_crov == 4:
            method = crov_intermediate
        elif method_crov == 5:
            method = crov_random

        for i in range(get_N_CHILDREN()):
            if len(parents) < 2:
                parents = self.get_parents().copy()
            p1, p2 = np.random.choice(len(parents), 2, replace=False).tolist()
            child = method(parents[p1], parents[p2])
            new_gen.append(child)

        if len(new_gen) < self.get_pop_size():
            parents = self.get_parents()
            for i in range(self.get_pop_size() - len(new_gen)):
                index = 0
                score = 0
                for j in range(len(parents)):
                    if parents[j].score > score:
                        score = parents[j].score
                        index = j
                new_gen.append(parents.pop(index))

    # TODO INTRODUZIR A LISTA COMPLETA DE EMPRESAS
    def evaluation_phase(self, eval_start, eval_end, use_trading=False):

        tickers = data.open_sp500_tickers_to_list()
        # tickers = data.open_all_sp500_tickers_to_list()
        cnt = 1  # TODO tirar
        for chro in self.get_chromo_list():
            print('evaluating ' + 'chromossome ' + str(cnt) + ' (' + str(self.get_pop_size()) + ')') # TODO tirar
            [forecast, orders] = forecast_orders(chro.get_gene_list(), tickers, self.get_chr_size(), eval_start, eval_end)
            chro.forecast = forecast
            chro.orders = orders
            if use_trading:
                portfolio = ts.trade(eval_start, eval_end, orders)
                score = portfolio.get_ROI()['value'].iloc[-1]
            else:
                [score, nr_trading_days, nr_correct_days,
                nr_up_days, nr_stay_days, nr_down_days,
                nr_correct_ups, nr_correct_stays, nr_correct_downs] = forecast_check(forecast, tickers)
            chro.set_score(score)
            chro.nr_trading_days = nr_trading_days
            chro.nr_correct_days = nr_correct_days
            chro.nr_up_days = nr_up_days
            chro.nr_stay_days = nr_stay_days
            chro.nr_down_days = nr_down_days
            chro.nr_correct_ups = nr_correct_ups
            chro.nr_correct_stays = nr_correct_stays
            chro.nr_correct_downs = nr_correct_downs
            cnt += 1

    # TODO verificar se estão todas as condições
    def check_end_phase(self):  # 1 = end achieved, 0 = end not achieved
        score = 0
        best_chromo = deepcopy(self.h_fame[0])

        # if there is no evolution increase mutation_std 10 000, turns back to default otherwise
        if self.get_no_evol() >= get_NO_EVOL_STD_INCREASE():
            print('AUMENTOU')
            print(get_MUTATION_STD())
            global MUTATION_STD
            MUTATION_STD = get_MUTATION_STD() + 10000
            print(get_MUTATION_STD())
        else:
            print('MANTEVE')
            print(get_MUTATION_STD())
            MUTATION_STD = get_DEFAULT_MUTATION_STD()
            print(get_MUTATION_STD()    )
        if score > get_END_VALUE() or self.get_no_evol() > get_MAX_NO_EVOL() or \
                self.get_generation() >= get_MAX_N_GEN():
            return 1
        else:
            return 0


############################
#    crossover methods     #
############################
def crov_intermediate(chromo1, chromo2):  # w is the weight of the 1st parent (between 0 and 1)
    gene_list = []
    w = get_CROV_W()
    for p1_gene, p2_gene in zip(chromo1.get_gene_list(), chromo2.get_gene_list()):
        gene_list.append(Gene(w * p1_gene.value + (1 - w) * p2_gene.value))
    chromo3 = Chromosome(gene_list)
    return chromo3


def crov_geometric(chromo1, chromo2):
    gene_list = []
    for p1_gene, p2_gene in zip(chromo1.get_gene_list(), chromo2.get_gene_list()):
        gene_list.append(Gene(sqrt(p1_gene.value * p2_gene.value)))
    chromo3 = Chromosome(gene_list)
    return chromo3


def crov_1point(chromo1, chromo2):
    gene_list = []
    p1_genes = chromo1.get_gene_list().copy()
    p2_genes = chromo2.get_gene_list().copy()
    point = randrange(chromo1.get_size())

    for i in range(point):
        gene_list.append(Gene(p1_genes[i].value))
    for i in range(point, chromo1.get_size()):
        gene_list.append(Gene(p2_genes[i].value))
    chromo3 = Chromosome(gene_list)
    return chromo3


def crov_2points(chromo1, chromo2):
    gene_list = []
    p1_genes = chromo1.get_gene_list()
    p2_genes = chromo2.get_gene_list()
    points = np.random.choice(chromo1.get_size(), 2, replace=False).tolist()
    points.sort()

    for i in range(points[0]):
        gene_list.append(p1_genes[i])
    for i in range(points[0], points[1]):
        gene_list.append(p2_genes[i])
    for i in range(points[1], chromo1.get_size()):
        gene_list.append(p1_genes[i])

    chromo3 = Chromosome(gene_list)
    return chromo3


def crov_random(chromo1, chromo2):
    gene_list = []
    for p1_gene, p2_gene in zip(chromo1.get_gene_list(), chromo2.get_gene_list()):
        random_gene = np.random.choice([p1_gene, p2_gene])
        value = random_gene.get_value()
        gene_list.append(Gene(value))
    chromo3 = Chromosome(gene_list)
    return chromo3


############################
#        forecast          #
############################
# TODO METER O RESTO DOS INDICADORES...
def forecast_orders(genes, tickers, chr_size, eval_start, eval_end):
    forecast = pd.DataFrame()
    orders = pd.DataFrame()

    # fp_stock_rsi = 'data/technical_indicators/' + str(unnorm_ti(genes[-8].get_value())) + '_stock_rsi.csv'
    # fp_stock_roc = 'data/technical_indicators/' + str(unnorm_ti(genes[-7].get_value())) + '_stock_roc.csv'
    # fp_stock_sto = 'data/technical_indicators/' + str(unnorm_ti(genes[-6].get_value())) + '_stock_sto.csv'
    fp_ivol_rsi = 'data/technical_indicators/' + str(unnorm_ti(genes[-5].get_value())) + '_ivol_rsi.csv'
    fp_ivol_roc = 'data/technical_indicators/' + str(unnorm_ti(genes[-4].get_value())) + '_ivol_roc.csv'
    fp_ivol_sto = 'data/technical_indicators/' + str(unnorm_ti(genes[-3].get_value())) + '_ivol_sto.csv'
    fp_ivol_macd = 'data/technical_indicators/' + str(unnorm_ti(genes[-2].get_value())) + '_ivol_macd.csv'
    fp_ivol_xema = 'data/technical_indicators/' + str(unnorm_xema(genes[-1].get_value())[0])+'-'+str(unnorm_xema(genes[-1].get_value())[1]) + '_ivol_xema.csv'

    # stock_rsi = pd.read_csv(fp_stock_rsi, index_col='Date', parse_dates=True)
    # stock_roc = pd.read_csv(fp_stock_roc, index_col='Date', parse_dates=True)
    # stock_sto = pd.read_csv(fp_stock_sto, index_col='Date', parse_dates=True)
    ivol_rsi = pd.read_csv(fp_ivol_rsi, index_col='Date', parse_dates=True)
    ivol_roc = pd.read_csv(fp_ivol_roc, index_col='Date', parse_dates=True)
    ivol_sto = pd.read_csv(fp_ivol_sto, index_col='Date', parse_dates=True)
    ivol_macd = pd.read_csv(fp_ivol_macd, index_col='Date', parse_dates=True)
    ivol_xema = pd.read_csv(fp_ivol_xema, index_col='Date', parse_dates=True)

    gene_sum = 0
    for i in range(5):
        gene_sum += genes[i].get_value()
    for ticker in tickers:
        for date in ivol_rsi.index:
            if date < eval_start:
                continue
            elif date > eval_end:
                break
            else:
                fc = (ivol_rsi.loc[date, 'ivol_' + ticker + '_rsi'] * genes[0].get_value() +
                      ivol_roc.loc[date, 'ivol_' + ticker + '_roc'] * genes[1].get_value() +
                      ivol_sto.loc[date, 'ivol_' + ticker + '_sto'] * genes[2].get_value() +
                      ivol_macd.loc[date, 'ivol_' + ticker + '_macd'] * genes[3].get_value() +
                      ivol_xema.loc[date, 'ivol_' + ticker + '_xema'] * genes[4].get_value()) / gene_sum  # TODO ... AQUI

                # print('---------------------------')
                # print(ticker)
                # print('vix rsi: ' + str(vix_rsi.loc[date, 'vix_rsi']))
                # print('gene vix rsi: ' + str(genes[0].get_value()))
                # print('vix roc: ' + str(vix_roc.loc[date, 'vix_roc']))
                # print('gene vix roc: ' + str(genes[1].get_value()))
                # # print('stock rsi: ' + str(stock_rsi.loc[date, 'stock_' + ticker + '_rsi']))
                # # print('gene stock rsi: ' + str(genes[2].get_value()))
                # # print('stock roc: ' + str(stock_roc.loc[date, 'stock_' + ticker + '_roc']))
                # # print('gene stock roc: ' + str(genes[3].get_value()))
                # print('ivol rsi: ' + str(ivol_rsi.loc[date, 'ivol_' + ticker + '_rsi']))
                # print('gene ivols rsi: ' + str(genes[2].get_value()))
                # print('ivol roc: ' + str(ivol_roc.loc[date, 'ivol_' + ticker + '_roc']))
                # print('gene ivol roc: ' + str(genes[3].get_value()))
                # print('foretasted value: ' + str(fc))
                # print('genesum: ' + str(gene_sum))
                # print('---------------------------')
                # time.sleep(1)
                forecast.at[ticker, date] = fc
                if fc >= 60:  # TODO VERIFICAR O VALOR
                    orders.at[ticker, date] = 1
                elif fc <= 40:  # TODO VERIFICAR O VALOR
                    orders.at[ticker, date] = -1
                else:
                    orders.at[ticker, date] = 0

    return forecast, orders


def forecast_check(forecast, tickers):
    change_step = get_IVOL_CHANGE_STEP()
    nr_correct_days, nr_trading_days = 0,0
    nr_correct_ups, nr_correct_downs, nr_correct_stays = 0,0,0
    nr_up_days, nr_down_days, nr_stay_days = 0,0,0

    all_ivol = pd.read_csv('data/implied_volatility/all_tickers_ivol.csv')
    all_ivol['Date'] = pd.to_datetime(all_ivol['Date'])
    all_ivol = all_ivol.set_index('Date')
    for ticker in tickers:
        ivol = all_ivol[ticker]
        for date in forecast.columns:
            if date in ivol.index:
                ivol_pre_idx = ivol.index.get_loc(date)
                ivol_pre_value = ivol.iloc[ivol_pre_idx]
                ivol_fut_idx = ivol_pre_idx + FORECAST_DIST
                if ivol_fut_idx > ivol.size - 1 - FORECAST_DIST:
                    break
                ivol_fut_value = ivol.iloc[ivol_fut_idx]
                change = ivol_fut_value - ivol_pre_value
                if math.isnan(change):
                    continue
                # TODO VERIFICAR OS VALOR
                # print('--------')
                # print(forecast.at[ticker, date])
                # print(change)
                # print('--------')

                if change >= change_step:
                    nr_up_days += 1
                    if forecast.at[ticker, date] >= 60:
                        nr_correct_ups += 1
                        nr_correct_days += 1
                elif change <= -change_step:
                    nr_down_days += 1
                    if forecast.at[ticker, date] <= 40:
                        nr_correct_downs += 1
                        nr_correct_days += 1
                else:
                    nr_stay_days +=1
                    if 40 >= forecast.at[ticker, date] >= 60:
                        nr_correct_stays += 1
                        nr_correct_days += 1

                nr_trading_days += 1
    print('CORRECT DAYS: ' + str(nr_correct_days) +' ('+ str(nr_trading_days) +')')
    print('correct buy: ' + str(nr_correct_ups) +' ('+ str(nr_up_days) +')')
    print('correct sell: ' + str(nr_correct_downs) +' ('+ str(nr_down_days) +')')
    print('correct stay: ' + str(nr_correct_stays) +' ('+ str(nr_stay_days) +')')
    score = nr_correct_days / nr_trading_days
    return [score, nr_trading_days, nr_correct_days,
            nr_up_days, nr_stay_days, nr_down_days,
            nr_correct_ups, nr_correct_stays, nr_correct_downs]

############################
#          other           #
############################
def check_same_chromo(chromo1, chromo2):
    for i in range(chromo1.get_size()):
        if chromo1.get_gene_list()[i].get_value() != chromo2.get_gene_list()[i].get_value():
            return False
    return True
