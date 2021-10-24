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

def find_domain(row, col, X, param):
    count = {'a':0, 'm':0, 'e':0, 'r':0}

    d = []
    
    for r in range(row+1):
        if X[r][col].value != '':
            count[(X[r][col].value).lower()] += 1
    available = param['N'] - sum(count.values())
    
    if ((X[row][col-1].value == 'R' or X[row][col-1].value == 'A') and count['m'] < param['m']):
        d.append('M')
    if  count['m'] > param['m'] or count['a'] > param['a'] or count['e'] > param['e'] \
        or (param['r'] - count['r']) > available \
        or (param['m'] - count['m']) > available \
        or (param['a'] - count['a']) > available \
        or (param['e'] - count['e']) > available \
        or ((param['m'] - count['m']) + (param['a'] - count['a']) + (param['e'] - count['e'])) > available:
            return []
    if count['r'] < param['r']:
        d.append('R')
    # if count['m'] < param['m'] and (X[row][col-1].value != 'E' and X[row][col-1].value != 'M'):
    #     d.append('M')
    if count['e'] < param['e']:
        d.append('E')
    if count['a'] < param['a']:
        d.append('A')
    
    #return d
    return random.sample(d, len(d))
    

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
    if count % 20000 == 0:
        print('Count: {}, Row: {}, Col: {}'.format(count, row, col))
    
    # if len(domain[row][col]) == 0:
    #     return []        
    # if col == 0:            
    # domain[row][col] = find_domain(row, col, nodes, domain)    
    
    curr_domain = find_domain(row, col, nodes, param)
    for i in range(len(curr_domain)):
        # val = domain[row][col][(col+i)%len(domain[row][col])]
        val = curr_domain[i]
        curr_node.assign_node(val)
        consistent = True
        
        for in_node in curr_node.in_nodes:
            if not arc_valid(in_node, curr_node, nodes, param):
                consistent = False
                curr_node.assign_node('')
                break
        
        if consistent:
            # domain_new = copy.deepcopy(domain)
            # domain_new[row][col] = [val]
            # arcs = []
            # for out_node in curr_node.out_nodes:
            #     arcs.append((curr_node, out_node))
            
            # if row == (param['N']-1) or True:
            #     valid, domain_updated = arc_consistency(arcs, nodes, domain_new, param)
            # else:
            #     valid, domain_updated = True, domain_new
            
            # if valid:
            result = back_track(row+1, col, nodes, param)
            if len(result) > 0:
                return result
            # else:
            #     print(domain)
            #     print('Row:{}, Col:{}'.format(row,col))
            #     print(domain_updated)
                
            curr_node.assign_node('')
    return []

def solve_csp(input_path):
    param = read_input(input_path)
    print(param)
    if param['m'] + param['a'] + param['e'] > param['N']:
        print('Invalid arguments for m, a, e, N')
        print( '(m + a + e <= N) should hold')
        return []
    
    if param['D'] >= 7 and param['m'] + param['a'] + param['e'] == param['N']:
        print('Invalid arguments for m, a, e, N')
        print( '(m + a + e < N) should hold {Strictly less cause each nurse needs to have atleast 1 R shift in a week}')
        return []
    if param['r'] + param['a'] < param['m']:
        print('Invalid arguments for m, a, e, N')
        return []
    if (param['r'] * 7 < param['N'] and param['D'] >= 7):
        return []
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

# def check_solution(nodes):
    

if __name__ == '__main__':
    input_path = sys.argv[1]
    
    start = time.time()
    
    result = solve_csp(input_path)
    dump_output(result)
    
    end = time.time()
    
    print('Time: ' + str(round(end-start,5)) + 's')
    

# Look out for these settings

# 18,7,9,3,1

# N,D,m,a,e
# 18,7,9,3,1

# {'N': 10, 'D': 14, 'm': 4, 'a': 2, 'e': 2}

# R, M, A, E

# ['R', 'M', 'A', 'M', 'A', 'M', 'R', 'M', 'A', 'M', 'R', 'M', 'A', 'M']
# ['R', 'M', 'A', 'M', 'A', 'M', 'R', 'M', 'A', 'M', 'R', 'M', 'A', 'M']
# ['M', 'R', 'M', 'A', 'M', 'A', 'M', 'R', 'M', 'A', 'M', 'R', 'M', 'A']
# ['M', 'R', 'M', 'A', 'M', 'A', 'M', 'R', 'M', 'A', 'M', 'R', 'M', 'A']
# ['M', 'A', 'M', 'R', 'M', 'E', 'A', 'M', 'R', 'M', 'A', 'M', 'R', 'M']
# ['M', 'A', 'M', 'R', 'M', 'E', 'A', 'M', 'R', 'M', 'A', 'M', 'R', 'M']
# ['A', 'M', 'E', 'E', 'R', 'M', 'E', 'A', 'M', 'R', 'M', 'A', 'M', 'E']
# ['A', 'M', 'E', 'E', 'R', 'M', 'E', 'A', 'M', 'R', 'M', 'A', 'M', 'E']
# ['E', 'E', 'R', 'M', 'E', 'R', 'M', 'E', 'E', 'E', 'E', 'E', 'E', 'R']
# ['E', 'E', 'R', 'M', 'E', 'R', 'M', 'E', 'E', 'E', 'E', 'E', 'E', 'R']

# 5R
# 6A
# 5E
# 10M
