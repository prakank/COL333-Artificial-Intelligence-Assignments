import pandas as pd
import csv
import os
import time

N = 30
D = 14
m = N
a = N
e = N

L = []

for i in range(1,N+1):
    for j in range(1, D+1):
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

                    d = {'N':i, 'D':j, 'm':a, 'a':b, 'e':c}                    
                    
                    with open('input.csv', 'w') as csv_file:
                        writer = csv.writer(csv_file)
                        L1 = []
                        L2 = []
                        for key, value in d.items():
                            L1.append(key)
                            L2.append(str(value))
                        
                        # print(L1)
                        # print(L2)
                        # s1 = ",".join(L1)
                        # # s1 = s1[1:len(s1)-2]
                        # s2 = ",".join(L2)
                        # # s2 = s2[1:len(s2)-2]
                        # print(s1)
                        # print(s2)
                        
                        # # writer.writerow(s1)
                        # # writer.writerow(s2)
                        # writer.writerows(d)
                    
                    os.system("python3 testing.py input.csv a {} {} {} {} {}".format(i,j,a,b,c))