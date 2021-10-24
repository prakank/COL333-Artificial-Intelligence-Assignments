import numpy as np
import pandas as pd
import sys
import os
import time
import json
import copy
import random

def read_input(path):
    df = pd.read_csv(path)
    param = {
        'N' : int(df.N),
        'D' : int(df.D),
        'm' : int(df.m),
        'a' : int(df.a),
        'e' : int(df.e),
        'r' : int(df.N) - (int(df.m) + int(df.a) + int(df.e))
    }
    return param

def dump_output(X):
    if len(X) == 0:
        print('No solution')
        ans = dict()
    else:
        N = len(X)
        M = len(X[0])
        ans = dict()
        for i in range(N):
            print('N-'+str(i)+': ',end='')
            for j in range(M):
                print(X[i][j].value + ' ',end=' ')
                ans['N{}_{}'.format(str(i),str(j))] = X[i][j].value
            print()
            
    with open('solution.json' , 'w') as file:
        json.dump(ans,file)
        file.write('\n')

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.out_nodes = []
        self.in_nodes = []
        self.value = ''
        
    # def add_out(self, out_node):
    #     np.append(self.out_nodes, out_node)

    # def add_in(self, in_node):
    #     np.append(self.in_nodes, in_node)
    #     in_node.add_out(self)

    def assign_node(self, value):
        self.value = value
        
    def is_equal(self, node):
        return (self.row == node.row and self.col == node.col)

def initialize_param(N, D):
    #domain = [[[] for i in range(D)] for j in range(N)]
    nodes = [[None for i in range(D)] for j in range(N)]
    
    for r in range(N):
        for c in range(D):
            nodes[r][c] = Node(r,c)
            if (r >= 1):
                nodes[r][c].in_nodes.append(nodes[r-1][c])
                nodes[r-1][c].out_nodes.append(nodes[r][c])
            if (c >= 1):
                nodes[r][c].in_nodes.append(nodes[r][c-1])
                nodes[r][c-1].out_nodes.append(nodes[r][c])
    return nodes

def arc_valid(nodei, nodej, nodes, param):
    # i -> j
    if nodei.row == nodej.row:
        if (nodej.value == 'M' and (nodei.value == 'M' or nodei.value == 'E')):
            return False
        if (nodej.col + 1) % 7 == 0:
            R_found = False
            for day in range(7):
                if (nodes[nodej.row][nodej.col-day].value == 'R'):
                    R_found = True
                    break
            if not R_found:
                return False
        # return True
    
    count = {'a':0, 'm':0, 'e':0, 'r':0}
    
    for row in range(nodej.row+1):
        if nodes[row][nodej.col].value != '':
            count[(nodes[row][nodej.col].value).lower()] += 1

    available = param['N'] - sum(count.values())
    
    if  count['m'] > param['m'] or count['a'] > param['a'] or count['e'] > param['e'] \
        or (param['m'] - count['m']) > available \
        or (param['a'] - count['a']) > available \
        or (param['e'] - count['e']) > available \
        or ((param['m'] - count['m']) + (param['a'] - count['a']) + (param['e'] - count['e'])) > available:
            return False
    return True
    
def arc_consistency(arcs, nodes, domain, param):
    while len(arcs) > 0:
        nodei, nodej = arcs.pop(0)
        init_i, init_j = nodei.value, nodej.value
        revised = False
        
        # print('Rowi:{}, Coli:{}, Rowj:{}, Colj:{}'.format(nodei.row, nodei.col, nodej.row, nodej.col))
        
        # xi, xj
        # xk, xi
        
        # xi, xj
        # xj, xk

        j=0
        while j<len(domain[nodej.row][nodej.col]):
            valj = domain[nodej.row][nodej.col][j]
            nodes[nodej.row][nodej.col].assign_node(valj)
            valid = False
            
            i=0
            while i < len(domain[nodei.row][nodei.col]):
                vali = domain[nodei.row][nodei.col][i]
                nodes[nodei.row][nodei.col].assign_node(vali)
                if arc_valid(nodei, nodej, nodes, param):
                    valid = True
                    break
                i+=1
                
            if not valid:
                domain[nodej.row][nodej.col].remove(valj)
                revised = True
            else:
                j+=1

        nodei.value, nodej.value = init_i, init_j
        
        if len(domain[nodej.row][nodej.col]) == 0 or len(domain[nodei.row][nodei.col]) == 0:
            return False, domain
        
        if len(domain[nodej.row][nodej.col]) == 1:
            nodej.value = domain[nodej.row][nodej.col][0]
            
        if len(domain[nodei.row][nodei.col]) == 1:
            nodei.value = domain[nodei.row][nodei.col][0]
        
        if revised:
            for out_node in nodej.out_nodes:
                arcs.append((nodej, out_node))
    return True, domain

def ordered(arr):
    s = 0
    for i in arr:
        if i.value == 'M' or i.value == 'E':
            s+=1
    return s

def find_domain(row, col, X, param):
    count = {'a':0, 'm':0, 'e':0, 'r':0}
    
    for r in range(row+1):
        if X[r][col].value != '':
            count[(X[r][col].value).lower()] += 1
            
    available = param['N'] - sum(count.values())
    
    d = []
    
    if (col + 1) % 7 == 0:
        R_found = False
        for day in range(1,7):
            if (X[row][col-day].value == 'R'):
                R_found = True
                break
        if not R_found and count['r'] < param['r']:
            return ['R']
        elif not R_found and count['r'] == param['r']:
            return []
    
    r_found = False
    r_prior = None
    min_val = (col%7)

    for day in range(1,min_val+1):
        if X[row][col-day].value == 'R':
            r_found = True
            break

    # 21,7,10,7,1    
    if ((X[row][col-1].value == 'R' or X[row][col-1].value == 'A') and count['m'] < param['m']):
        d.append('M')
        if (param['a'] + param['r'] == param['m']):
            return d
  
    if r_found == False and count['r'] < param['r']:
        r_prior = True
        d.append('R')
    else:
        r_prior = False        
        
    # if X[row][col-1].value == 'A':
    #     if count['a'] < param['a']:
    #         d.append('A')
    #     elif count['r'] < param['r']:
    #         d.append('R')
    
    # if X[row][col-1].value == 'A':
    #     if count['a'] < param['a']:
    #         d.append('A')
    #     elif count['r'] < param['r']:
    #         d.append('R')

    if count['a'] < param['a']:
        d.append('A')
        
    if count['e'] < param['e']:
        d.append('E')
        
    if (count['r'] < param['r']) and (r_prior == False):
        d.append('R')
        
    return d


    # return random.sample(d, len(d))

count = 0
def back_track(row, col, nodes, param):
    if row == param['N']:
        row = 0
        col += 1    
    if col == param['D']:
        return nodes
    
    curr_node = nodes[row][col]
    
    global count
    count+=1
    if count % 10000000 == 0:
        # print("Printing ....")
        print('Count: {}, Row: {}, Col: {}'.format(count, row, col))
        # for r in range(row+1):
        #     for c in range(col+1):
        #         print(nodes[r][c].value,end=" ")
        #     print()
        # print("\n\n")

    # if col >= 2:
    #     print('2nd column, Count: {}, Row: {}, Col: {}'.format(count, row, col))
    
    curr_domain = find_domain(row, col, nodes, param)
    
    for i in range(len(curr_domain)):        
        val = curr_domain[i]
        curr_node.assign_node(val)
        consistent = True
        
        if consistent:
            result = back_track(row+1, col, nodes, param)
            if len(result) > 0:
                return result
            curr_node.assign_node('')
    return []

def solve_csp(param):
    
    print(param)
    
    if param['m'] + param['a'] + param['e'] > param['N']:
        print('Invalid arguments for m, a, e, N')
        print( '(m + a + e <= N) should hold')
        return []
    
    if param['D'] >= 7 and param['m'] + param['a'] + param['e'] == param['N']:
        print('Invalid arguments for m, a, e, N')
        print( '(m + a + e < N) should hold {Strictly less cause each nurse needs to have atleast 1 R shift in a week}')
        return []
    
    # if param['r'] + param['a'] < param['m']:
    #     print('Invalid arguments for m, a, e, N')
    #     print("Sum of param r and a should be greater than param m")
    #     return []
    
    if (param['r'] * 7 < param['N'] and param['D'] >= 7):
        print('Invalid arguments for m, a, e, N')
        print("param r should be high enough to assign Rest day to each nurse")
        return []
    
    # r, a, e, m
    nodes = initialize_param(param['N'], param['D'])    
    for i in range(param['r']):
        nodes[i][0].value = 'R'
        #domain[i][0] = ['M']
    
    for i in range(param['a']):
        nodes[param['r']+i][0].value = 'A'
        #domain[param['m']+i][0] = ['A']
    
    for i in range(param['e']):
        nodes[param['r']+param['a']+i][0].value = 'E'
        #domain[param['m']+param['a']][0] = ['E']
    
    for i in range(param['m']):
        nodes[param['r'] + param['a'] + param['e'] + i][0].value = 'M'
        #domain[i][param['m'] + param['a'] + param['e']] = ['R']
    
    return back_track(0, 1, nodes, param)

def check(X, param):
    if not len(X):
        print("\nERROR\n")
        return False
    correct = True
    for r in range(len(X)):
        for c in range(len(X[0])):
            if (r >= 1):
                correct = arc_valid(X[r-1][c], X[r][c], X, param)
                if not correct:
                    return False
            if (c >= 1):
                correct = arc_valid(X[r][c-1], X[r][c], X, param)
                if not correct:
                    return False
            if (c+1)%7==0:
                r_found = False
                for day in range(7):
                    if X[r][c-day].value == 'R':
                        r_found = True
                        break
                if not r_found:
                    return False
    return True

# def combine(DAYS, X, param):
#     X_new = initialize_param(param['N'], DAYS)
#     start = 0
#     end = 6
    
#     for i in range(param['N']):        
#         assign = [-1 for j in range(param['N'])]
        
#         dict_start = {'R':[], 'A':[], 'E':[], 'M':[]}
#         dict_end = {'R':[], 'A':[], 'E':[], 'M':[]}
        
#         dict_start[X[i][start].value].append(i)
#         dict_end[X[i][end].value].append(i)
    
#     r = 0
#     a = 0
#     start_index = [1 for j in range(param['N'])]
#     end_index   = [1 for j in range(param['N'])]
    
#     for i in range(len(dict_start['M'])):
#         if r < len(dict_end['R'][r]):
#             start_index[dict_start['M'][i]] = 0
#             end_index[dict_end['R'][r]] = 0
            
#             assign[dict_end['R'][r]] = dict_start['M'][i]
#             r+=1
#         elif a < len(dict_end['A'][a]):
#             start_index[dict_start['M'][i]] = 0
#             end_index[dict_end['A'][a]] = 0
            
#             assign[dict_end['A'][a]] = dict_start['M'][i]
#             a+=1
    
#     i,j=0,0
#     while
    
    
if __name__ == '__main__':
    input_path = sys.argv[1]
    
    start = time.time()
    
    param = read_input(input_path)
    
    # DAYS = param['D']
    # param['D'] = min(param['D'], 7)
    
    sys.setrecursionlimit(4000)
    print(sys.getrecursionlimit())
    
    result = solve_csp(param)
    result = sorted(result, key=lambda x: ordered(x), reverse=True)
    
    # if DAYS > 7:
    #     result = combine(DAYS, result, param)
    
    correct = check(result, param)

    dump_output(result)

    if correct:
        print("\nVOILA\n")
    else:
        print("\nError\n")
        
    end = time.time()
    
    print('Time: ' + str(round(end-start,5)) + 's')
    