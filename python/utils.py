
import numpy as np
from operator import itemgetter


def unzip (l):
    return list(zip(*l))
def my_groupby(scorelist, eps):
    l = sorted(scorelist)

    packs = []
    cur = []
    cur_score = l[0][0]

    for score, coal in l:
        if score <= cur_score*(1+eps):
            cur.append(coal)
        else:
            packs.append((cur_score,cur))
            cur_score = score
            cur = [coal]
            #print (cur_score, " target=", cur_score*(1+eps))
    packs.append((cur_score,cur))
    
    print ("nb eq classes: ", len(packs))
    return packs
        
def my_groupby_window(scorelist, eps, initval=None):
    l = sorted(scorelist)

    packs = []
    cur = []

    if initval == None:
        initval = l[0][0]
    cur_score = initval

    for score, coal in l:
        if score <= cur_score*(1+eps):
            cur.append(coal)
        else:
            if cur != []:
                packs.append((cur_score,cur))
            cur_score = cur_score*(1+eps)
            cur = [coal]
            #print (cur_score, " target=", cur_score*(1+eps))
    packs.append((cur_score,cur))
    
    print ("nb eq classes: ", len(packs))
    return packs
        

def pop_nfirst (l, n):
    if l == []:
        return []
    else:
        return (l[:n], l[n:])
def group_by_chunks(l,n):
    cur = l
    ret = []
    i = 0
    while (cur != []):
        (tmp, cur) = pop_nfirst(cur, n)
        tmp = map (itemgetter(1), tmp)
        ret.append((i,list(tmp)))
        i += 1
    return ret
        

    
def linear_scale_groupby (scorelist, nclasses):
    l = sorted (scorelist)
    #return group_by_chunks(l, nclasses)
    return group_by_chunks(l,int(len(l)/nclasses))



   
