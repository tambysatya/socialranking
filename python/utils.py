


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
        

   
