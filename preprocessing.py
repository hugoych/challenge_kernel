import pandas as pd 
import numpy as np 

def string_to_int(x):
    if x == 'A':
        return 1
    elif x =='C':
        return 2
    elif x == 'G':
        return 3
    elif x == 'T':
        return 4
    else:
        print('WARNING: Unknown char')
        return None

def convert_data(str):
    int_s = []
    for s in str:
        int_s.append(string_to_int(s))
    return int_s

