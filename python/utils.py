


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
        
def linear_scale_groupby (scorelist, nclasses):
    l = sorted (scorelist)
    groups = np.split(np.array(scorelist), nclasses)
    groups = map (list, groups)
    return list (groups)



   
