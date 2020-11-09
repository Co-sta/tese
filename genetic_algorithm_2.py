import random
import numpy as np
from random import randrange
from math import floor
from math import sqrt
from scipy.stats import truncnorm
from functools import partial
from copy import deepcopy
from multiprocessing import Pool
# import statistics as stats
import genetic_algorithm_1 as ga1
import data
import technical_indicators as ti


############################
#       GA1 variables      #
############################
G1_POP_SIZE = 1.0  # initialization
G1_CHR_SIZE = 1.0  # initialization
G1_GENE_SIZE = 1.0  # initialization

def evaluate_multi(chromo, eval_start, eval_end):
    print('THREAD') # TODO tirar
    gene_list = chromo.get_gene_list()
    [ga1_n_parents,ga1_n_children,ga1_crov_w,ga1_mutation_rate,
    ga1_mutation_std,ga1_method_1pop,ga1_method_ps,ga1_method_crov] = unnorm(gene_list)

    # TODO VERIFICAR O QUE É QUE O GA1.SIMULATE RETORNA
    chromo.sub_pop = ga1.simulate(G1_POP_SIZE, G1_CHR_SIZE, G1_GENE_SIZE,
                                     ga1_n_parents, ga1_n_children, ga1_crov_w,
                                     ga1_mutation_rate, ga1_mutation_std,
                                     ga1_method_1pop, ga1_method_ps, ga1_method_crov,
                                     eval_start, eval_end)

    chromo.set_score(chromo.sub_pop.get_h_fame()[0].get_score())
    return chromo

def unnorm(gene_list, g1_pop_size=0, g1_gene_size=0):
    if not g1_pop_size: g1_pop_size = G1_POP_SIZE
    if not g1_gene_size: g1_gene_size = G1_GENE_SIZE
    ga1_n_parents = unnorm_n_parents(gene_list[0].get_value(), g1_pop_size, g1_gene_size)
    ga1_n_children = unnorm_n_children(gene_list[1].get_value(), ga1_n_parents, g1_pop_size, g1_gene_size)
    ga1_crov_w = unnorm_crov_w(gene_list[2].get_value())
    ga1_mutation_rate = unnorm_mutation_rate(gene_list[3].get_value())
    ga1_mutation_std = unnorm_mutation_std(gene_list[4].get_value(), g1_gene_size)
    ga1_method_1pop = unnorm_method_1pop(gene_list[5].get_value(), g1_gene_size)
    ga1_method_ps = unnorm_method_ps(gene_list[6].get_value(), g1_gene_size)
    ga1_method_crov = unnorm_method_crov(gene_list[7].get_value(), g1_gene_size)
    return [ga1_n_parents,ga1_n_children,ga1_crov_w,ga1_mutation_rate,
           ga1_mutation_std,ga1_method_1pop,ga1_method_ps,ga1_method_crov]

def unnorm_n_parents(n_parents_norm, g1_pop_size, g1_gene_size):
    pos_n_parents = np.arange(2, g1_pop_size + 1)
    step_n_parents = (g1_gene_size + 1) / len(pos_n_parents)
    i = int(np.floor(n_parents_norm / step_n_parents))
    return int(pos_n_parents[i])

def unnorm_n_children(n_children_norm, ga1_n_parents, g1_pop_size, g1_gene_size):
    pos_n_children = np.arange(g1_pop_size - ga1_n_parents, g1_pop_size + 1)
    step_n_children = (g1_gene_size + 1) / len(pos_n_children)
    i = int(np.floor(n_children_norm / step_n_children))
    return int(pos_n_children[i])

def unnorm_crov_w(crov_w_norm):
    return crov_w_norm / 100000

def unnorm_mutation_rate(mutation_rate_norm):
    return mutation_rate_norm / 100000

def unnorm_mutation_std(mutation_std_norm, g1_gene_size):
    pos_mutation_std = np.arange(0, 15000 + 1)
    step_mutation_std = (g1_gene_size + 1) / len(pos_mutation_std)
    i = int(np.floor(mutation_std_norm / step_mutation_std))
    return int(pos_mutation_std[i])

def unnorm_method_1pop(method_1pop_norm, g1_gene_size):
    pos_method_1pop = np.arange(1, 3 + 1)
    step_method_1pop = (g1_gene_size + 1) / len(pos_method_1pop)
    i = int(np.floor(method_1pop_norm / step_method_1pop))
    return int(pos_method_1pop[i])

def unnorm_method_ps(method_ps_norm, g1_gene_size):
    pos_method_ps = np.arange(1, 4 + 1)
    step_method_ps = (g1_gene_size + 1) / len(pos_method_ps)
    i = int(np.floor(method_ps_norm / step_method_ps))
    return int(pos_method_ps[i])

def unnorm_method_crov(method_crov_norm, g1_gene_size):
    pos_method_crov = np.arange(1, 5 + 1)
    step_method_crov = (g1_gene_size + 1) / len(pos_method_crov)
    i = int(np.floor(method_crov_norm / step_method_crov))
    return int(pos_method_crov[i])


############################
#     Global Variables     #
############################
END_VALUE = 0.9  # MEXER
N_TOP = 2   # MEXER
MAX_NO_EVOL = 2    # MEXER
MAX_N_GEN = 10  # max of generations per simulation # MEXER
H_FAME_SIZE = 5 # MEXER

GENE_SIZE = 1.0  # initialization

N_PARENTS = 1.0  # initialization
N_CHILDREN = 1.0  # initialization
CROV_W = 1.0  # initialization
MUTATION_RATE = 1.0  # initialization
MUTATION_STD = 1.0  # initialization

METHOD_1POP = 1.0  # initialization
METHOD_PS = 1.0  # initialization
METHOD_CROV = 1.0  # initialization




def set_global_var(ga2_gene_size, ga2_n_parents, ga2_n_children, ga2_crov_w, ga2_mutation_rate, ga2_mutation_std,
                   ga2_method_1pop, ga2_method_ps, ga2_method_crov,
                   g1_pop_size, g1_chr_size, g1_gene_size):

    global GENE_SIZE
    GENE_SIZE = ga2_gene_size
    global N_PARENTS
    N_PARENTS = ga2_n_parents
    global N_CHILDREN
    N_CHILDREN = ga2_n_children
    global CROV_W
    CROV_W = ga2_crov_w
    global MUTATION_RATE
    MUTATION_RATE = ga2_mutation_rate
    global MUTATION_STD
    MUTATION_STD = ga2_mutation_std
    global METHOD_1POP
    METHOD_1POP = ga2_method_1pop
    global METHOD_PS
    METHOD_PS = ga2_method_ps
    global METHOD_CROV
    METHOD_CROV = ga2_method_crov
    global G1_POP_SIZE
    G1_POP_SIZE = g1_pop_size
    global G1_CHR_SIZE
    G1_CHR_SIZE = g1_chr_size
    global G1_GENE_SIZE
    G1_GENE_SIZE = g1_gene_size



def simulate(ga2_pop_size, ga2_chromo_size, ga2_gene_size, ga2_n_parents, ga2_n_children, ga2_crow_w,
             ga2_mutation_rate, ga2_mutation_std, ga2_method_1pop, ga2_method_ps, ga2_method_crov,
             g1_pop_size, g1_chr_size, g1_gene_size, eval_start, eval_end, graph=False):
    set_global_var(ga2_gene_size, ga2_n_parents, ga2_n_children, ga2_crow_w, ga2_mutation_rate, ga2_mutation_std,
                   ga2_method_1pop, ga2_method_ps, ga2_method_crov,
                   g1_pop_size, g1_chr_size, g1_gene_size)

    pop = Population(ga2_pop_size, ga2_chromo_size)
    max_score = []

    while True:
        print('G2 generation nr: ' + str(pop.get_generation()))
        pop.evaluation_phase(eval_start, eval_end, 6)
        pop.update_h_fame()
        [end, best_chromo] = pop.check_end_phase()

        if end:
            return best_chromo   # TODO RETORNAR O MELHOR CHROMOSSOMA OU O HALL OF FAME?
        else:
            max_score.append(best_chromo.get_score())
            pop.parent_selection_phase()
            pop.crossover_phase()
            pop.mutation_phase()
            pop.increase_gen()


def get_GENE_SIZE():
    global GENE_SIZE
    return GENE_SIZE


def get_END_VALUE():
    global END_VALUE
    return END_VALUE


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


def get_MAX_NO_EVOL():
    global MAX_NO_EVOL
    return MAX_NO_EVOL


def get_MAX_N_GEN():
    global MAX_N_GEN
    return MAX_N_GEN


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
        self.sub_pop = None

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


    def get_sub_max_score(self):
        return self.sub_max_score

    def get_sub_pop(self):
        return self.sub_pop
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
        self.generation = 1

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
        for i in range(len(chromo_list)):
            sum_scores = sum_scores + chromo_list[i].get_score()
        for i in range(len(chromo_list)):
            scores.append(chromo_list[i].score / sum_scores)
        parents_i = (np.random.choice(len(chromo_list), get_N_PARENTS() - ntop, replace=False, p=scores).tolist())
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
                    chromo_list.remove(chromo)
        #  TODO CONTINUAR AQUI
        # for chromo in chromo_list:
        #     print('G2: score: ' + str(chromo.get_score()))
        # print('------------------------------------')
        chromo_list.sort(key=lambda x: x.score, reverse=True)
        # for chromo in chromo_list:
        #     print('G2: score ordenado: ' + str(chromo.get_score()))
        # print('------------------------------------')
        h_fame.extend(chromo_list)
        # for chromo in h_fame:
        #     print('G2: h_fame + list: ' + str(chromo.get_score()))
        # print('------------------------------------')
        h_fame.sort(key=lambda x: x.score, reverse=True)
        # for chromo in h_fame:
        #     print('G2: h_fame + list ordenado: ' + str(chromo.get_score()))
        # print('------------------------------------')
        new_h_fame = deepcopy(h_fame[0:h_fame_size])
        # for chromo in new_h_fame:
        #     print('G2: H_FAME: ' + str(chromo.get_score()))
        # print('------------------------------------')
        new_best_score = new_h_fame[0].get_score()

        if new_best_score > old_best_score:
            self.reset_no_evol()
        else:
            self.incr_no_evol()
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

        self.set_chromo_list(new_gen)

    def evaluation_phase(self, eval_start, eval_end, n_threads=5):
        with Pool(n_threads) as p:
            self.print_chromo_list()
            eval_multi=partial(evaluate_multi, eval_start=eval_start, eval_end=eval_end)
            self.chromo_list = p.map(eval_multi, self.get_chromo_list())


    # def evaluation_phase(self, eval_start, eval_end):
    #     cnt = 1  # TODO tirar
    #     for chromo in self.get_chromo_list():
    #         print('G2 generation nr: ' + str(self.get_generation())) # TODO tirar
    #         print('G2: evaluating ' + 'chromossome ' + str(cnt) + ' (' + str(self.get_pop_size()) + ')')  # TODO tirar
    #         gene_list = chromo.get_gene_list()
    #         [ga1_n_parents,ga1_n_children,ga1_crov_w,ga1_mutation_rate,
    #         ga1_mutation_std,ga1_method_1pop,ga1_method_ps,ga1_method_crov] = unnorm(gene_list)
    #
    #         # TODO VERIFICAR O QUE É QUE O GA1.SIMULATE RETORNA
    #         chromo.sub_pop = ga1.simulate(G1_POP_SIZE, G1_CHR_SIZE, G1_GENE_SIZE,
    #                                          ga1_n_parents, ga1_n_children, ga1_crov_w,
    #                                          ga1_mutation_rate, ga1_mutation_std,
    #                                          ga1_method_1pop, ga1_method_ps, ga1_method_crov,
    #                                          eval_start, eval_end)
    #
    #         chromo.set_score(chromo.sub_pop.get_h_fame()[0].get_score())
    #         cnt += 1

    # TODO verificar se estão todas as condições
    def check_end_phase(self):  # 1 = end achieved, 0 = end not achieved
        best_chromo = deepcopy(self.h_fame[0])
        score = best_chromo.get_score()

        if score > get_END_VALUE() or self.get_no_evol() > get_MAX_NO_EVOL() or \
                self.get_generation() >= get_MAX_N_GEN():
            return 1, best_chromo
        else:
            return 0, best_chromo


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
#          other           #
############################
def check_same_chromo(chromo1, chromo2):
    for i in range(chromo1.get_size()):
        if chromo1.get_gene_list()[i].get_value() != chromo2.get_gene_list()[i].get_value():
            return False
    return True
