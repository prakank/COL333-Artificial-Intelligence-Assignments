import pandas as pd
import csv
import os
import time

N = 40
D = 21
m = N
a = N
e = N

L = []

for i in range(20,N+1):
    for j in range(5, D+1):
        if j%7 != 6 and j%7 != 0:
            continue
        for a in range(m):
            for b in range(a):
                for c in range(e):
                    
                    if (a+b+c) > i:
                        continue
                    
                    if j >= 7 and (a+b+c) >= i:
                        continue
                    
                    r = i - (a+b+c)
                    
                    if r + b < a:
                        continue
                    
                    if 7*r < i and j >= 7:
                        continue
                    
                    # if (7*r-i) > 4:
                    #     continue

                    d = {'N':i, 'D':j, 'm':a, 'a':b, 'e':c}                    
                    
                    with open('input.csv', 'w') as csv_file:
                        writer = csv.writer(csv_file)
                        L1 = []
                        L2 = []
                        for key, value in d.items():
                            L1.append(key)
                            L2.append(str(value))

                    
                    os.system("python3 testing.py input.csv a {} {} {} {} {}".format(i,j,a,b,c))