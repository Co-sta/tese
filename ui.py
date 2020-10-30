import genetic_algorithm_2 as ga2
import genetic_algorithm_1 as ga1
import data as data
import technical_indicators as ti
import pickle
import plotly.express as px


def print_result(filename, ga1_pop_size, ga1_gene_size):
    filepath = 'data/results/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    genes2 = best_chromo.get_gene_list()
    print('G2-Chromossome :' + '(score=' +str(best_chromo.get_score())+ ')')


    [ga1_n_parents,ga1_n_children,ga1_crov_w,ga1_mutation_rate,
    ga1_mutation_std,ga1_method_1pop,ga1_method_ps,ga1_method_crov] = \
    ga2.unnorm(genes2, ga1_pop_size, ga1_gene_size)

    print('N parents: ' + str(ga1_n_parents) + ' (' + str(genes2[0].get_value()) + ')')
    print('N children: ' + str(ga1_n_children) + ' (' + str(genes2[1].get_value()) + ')')
    print('Crov w: ' + str(ga1_crov_w) + ' (' + str(genes2[2].get_value()) + ')')
    print('Mutaion rate: ' + str(ga1_mutation_rate) + ' (' + str(genes2[3].get_value()) + ')')
    print('Mutation std: ' + str(ga1_mutation_std) + ' (' + str(genes2[4].get_value()) + ')')
    print('First pop: ' + str(ga1_method_1pop) + ' (' + str(genes2[5].get_value()) + ')')
    print('Parent selct: ' + str(ga1_method_ps) + ' (' + str(genes2[6].get_value()) + ')')
    print('Crossover: ' + str(ga1_method_crov) + ' (' + str(genes2[7].get_value()) + ')')
    for i in range(len(best_chromo.get_sub_pop().get_h_fame()[0].get_gene_list())):
        chr1 = best_chromo.get_sub_pop().get_h_fame()[0].get_gene_list()[i]
        if i == 0: print('     vix_rsi_weight: ' + str(chr1.get_value()))
        if i == 1: print('     vix_roc_weight: ' + str(chr1.get_value()))
        if i == 2: print('     stock_rsi_weight: ' + str(chr1.get_value()))
        if i == 3: print('     stock_roc_weight: ' + str(chr1.get_value()))
        if i == 4: print('     ivol_rsi_weight: ' + str(chr1.get_value()))
        if i == 5: print('     ivol_roc_weight: ' + str(chr1.get_value()))

        if i == 6: print('     n_vix_rsi: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 7: print('     n_vix_roc: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 8: print('     n_stock_rsi: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 9: print('     n_stock_roc: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 10: print('     n_ivol_rsi: ' + str(ga1.unnorm_ti(chr1.get_value())))
        if i == 11: print('     n_ivol_roc: ' + str(ga1.unnorm_ti(chr1.get_value())))


def graph_score(filename):
    filepath = 'data/results/' + filename
    best_chromo = pickle.load( open( filepath, "rb" ))
    score_evol =  best_chromo.get_sub_pop().get_max_score()
    print(score_evol)
    fig = px.line(score_evol, x="epoch", y="score")
    fig.show()

# def graph_ROI(filename):
#     filepath = 'data/results/' + filename
#     best_chromo = pickle.load( open( filepath, "rb" ))
#
#     roi_evol =  best_chromo.get_sub_pop().get_max_score()
#     print(score_evol)
#     fig = px.line(score_evol, x="epoch", y="score")
#     fig.show()
