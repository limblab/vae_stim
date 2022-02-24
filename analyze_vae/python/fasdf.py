# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:07:09 2021

@author: Joseph Sombeck
"""


import numpy as np



max_count = 100

for i in range(1,max_count+1):
    to_print = ""
    
    if(np.mod(i,5)==0):
        to_print = to_print + "Fizz"
    if(np.mod(i,3)==0):
        to_print = to_print + "Buzz"

    if(to_print==""):
        to_print = i   
    
    print(to_print)