import pandas as pd 
import numpy as np 

def string_to_int(x):
    if x == 'A':
        return 0
    elif x =='C':
        return 1
    elif x == 'G':
        return 2
    elif x == 'T':
        return 3
    else:
        print('WARNING: Unknown char')
        return None

def convert_data(str):
    int_s = []
    for s in str:
        int_s.append(string_to_int(s))
    return int_s

