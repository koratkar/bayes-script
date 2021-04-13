import operator
from functools import reduce
import re

# predict the likelyhood that some text is "bad"
bad_corp      =  ""
good_corp     =  ""

with open('bad.txt', 'r', encoding="utf8") as file:
    bad_corp = file.read().replace('\n', '')

with open('good.txt', 'r', encoding="utf8") as file:
    good_corp = file.read().replace('\n', '')

def tokenize(str):
    clean = str.lower().split()
    p = r'[^a-z]'
    return [re.sub(p, '', s) for s in clean]

def create_token_lib(str):
    s = tokenize(str)
    def count_of(goal, inp):
        def citer(cdr, c):
            if (len(cdr) == 0):
                return c
            elif (cdr[0] == goal):
                return citer(cdr[1:], c+1)
            else:
                return citer(cdr[1:], c)
        return citer(inp, 0)
    def dicter(inp):
        def citer(cdr, d):
            if (len(cdr) == 0):
                return d
            elif cdr[0] in d:
                return citer(cdr[1:], d)
            else:
                d[cdr[0]] = count_of(cdr[0], cdr)
                return citer(cdr[1:], d)
        return citer(inp, {})
    return dicter(s)

good_dict    = create_token_lib(good_corp)
bad_dict     = create_token_lib(bad_corp)

def word_prob(word, good, bad):
    def in_dict_or_0(n, dict):
        if n in dict:
            return dict[n]
        else:
            return 0
    g = 2 * in_dict_or_0(word, good)
    b = in_dict_or_0(word, bad)
    gs = len(good_dict) * 2
    bs = len(bad_dict)
    if g > 0 and b <= 0:
        return 0.01
    elif g <= 0 and b > 0:
        return 0.99
    elif g == 0 and b == 0:
        return 0
    else:
        return min([round((min([(g / gs), .99])), 2) + round(min([(b / bs), .99]), 2), .99])

def naive_bayes_prob(str):
    s = tokenize(str)
    p = [word_prob(x, good_dict, bad_dict) for x in s]
    mp = reduce(operator.mul, [x for x in p if x > 0])
    one_minus = reduce(operator.mul, [1 - x for x in p])
    return round(mp / (mp + one_minus), 2)

## RUN naive_bayes_prob from here