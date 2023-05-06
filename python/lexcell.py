
from operator import itemgetter
import itertools
import random


def nb_coalitions_ei_belongs_to (el, coals):
    # counts the number of occurences of the considered individual (el) in a list of coalitions
    return (len([1 for gi in coals[1] if el in gi]))    
def nb_total_sorted_coalitions_ei_belongs_to (el, grouped_coals):
    #print (el, list(grouped_coals))
    _,coals = itertools.tee(grouped_coals, 2)
    return list(map (lambda gi: nb_coalitions_ei_belongs_to(el,gi), coals))

    

def lex_cell (individuals, coalitions, scores):
    #print (individuals)
    sorted_coals = sorted (zip (scores, coalitions))

    grouped_coals = itertools.groupby(sorted_coals, key=itemgetter(0))
    grouped_coals = map (lambda x: (x[0], list(map (itemgetter(1), x[1]))), grouped_coals) #group by list
    #print (sorted_coals)



    lex_scores = map (lambda ei, gcoals: nb_total_sorted_coalitions_ei_belongs_to(ei, gcoals), individuals, itertools.tee(grouped_coals, len(individuals)))
    lex_scores = sorted (zip(lex_scores, individuals), reverse=True)

    #print (list (lex_scores))
    lex_order = map (itemgetter(1), lex_scores)
    
    return list(lex_order)


def random_coalition (individuals, l):
    inds = list(individuals)
    random.shuffle(inds)
    return set(inds[:l])

def random_coalitions (individuals, l, n): 
    """generates at most n coalitions of size n"""
    ret = [random_coalition(individuals,l) for i in range (n)]
    return [k for k, v in itertools.groupby(sorted(ret)) ]
    
    


#ret = lex_cell ([1,2,3,4], [{1,2},{3,2},{1,3}, {1,4}], [3,1,2,1])
#print(ret)

#inds = {1,2,3,4,5,6,7,8,9,10}
#ret = random_coalition ({1,2,3,4,5}, 3)
#ret = random_coalitions(inds, 3, 5)
#print (ret)
