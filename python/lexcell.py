
from operator import itemgetter
import itertools
from utils import *
import random


def nb_coalitions_ei_belongs_to (el, coals):
    # counts the number of occurences of the considered individual (el) in a list of coalitions
    return (len([1 for gi in coals[1] if el in gi]))    
def nb_total_sorted_coalitions_ei_belongs_to (el, grouped_coals):
    #print (el, list(grouped_coals))
    return list(map (lambda gi: nb_coalitions_ei_belongs_to(el,gi), grouped_coals))

    

def lex_cell (individuals, coalitions, scores, eps=0):
    #print (individuals)

    #sorted_coals = sorted (zip (scores, coalitions), reverse=True)
    #grouped_coals = itertools.groupby(sorted_coals, key=itemgetter(0))
    #grouped_coals = map (lambda x: (x[0], list(map (itemgetter(1), x[1]))), grouped_coals) #group by list
    #print (sorted_coals)

    grouped_coals = list(reversed(my_groupby (zip (scores, coalitions), eps) ))



    lex_scores = map (lambda ei, gcoals: nb_total_sorted_coalitions_ei_belongs_to(ei, gcoals), individuals, itertools.tee(grouped_coals, len(individuals)))
    lex_scores = sorted (zip(lex_scores, individuals), reverse=True)

    #print (list (lex_scores))
    lex_order = map (itemgetter(1), lex_scores)
    
    return list(lex_order)


################## extended version

def is_in_coal (el, coal): # Int -> ([{0,1}], [Int]) -> Bool
    """ true if el belongs to the coalition AND is selected (ie ==1) """
    pos, inds = coal
    return el in inds and  pos[el] == 1

def nb_coals_equal_1(el, coals):
    return len([ci for ci in coals if is_in_coal(el, ci)])

def compute_eq_classes (el, grouped_coals):
    return list(map (lambda gi: nb_coals_equal_1(el,gi), grouped_coals))
 
def adv_lex_cell (individuals, coalitions, scores, eps=0):
    #print (individuals)
    #sorted_coals = sorted (zip (scores, coalitions), reverse=True)
    #grouped_coals = itertools.groupby(sorted_coals, key=itemgetter(0))
    #grouped_coals = map (lambda x: (list(map (itemgetter(1), x[1]))), grouped_coals) #group by list
    #print (coalitions)
    #print (list(grouped_coals))
    #print (sorted_coals)
    grouped_coals = list(reversed(my_groupby (zip (scores, coalitions), eps) ))
    grouped_coals = map (itemgetter(1), grouped_coals)



    lex_scores = map (lambda ei, gcoals: compute_eq_classes(ei, gcoals), individuals, itertools.tee(grouped_coals, len(individuals)))
    lex_scores = sorted (zip(lex_scores, individuals), reverse=True)

    #print (list (lex_scores))
    lex_order = map (itemgetter(1), lex_scores)
    
    return list(lex_order)

   
    
    


#ret = lex_cell ([1,2,3,4], [{1,2},{3,2},{1,3}, {1,4}], [3,1,2,1])
#print(ret)

#inds = {1,2,3,4,5,6,7,8,9,10}
#ret = random_coalition ({1,2,3,4,5}, 3)
#ret = random_coalitions(inds, 3, 5)
#print (ret)
