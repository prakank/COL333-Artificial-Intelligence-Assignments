import numpy as np
import pandas as pd
import sys
import time
import json

def read_input(path):
    df = pd.read_csv(path)
    param = {
        "N" : int(df.N),
        "D" : int(df.D),
        "m" : int(df.m),
        "a" : int(df.a),
        "e" : int(df.e)
    }
    return param

class Node: 
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.out_nodes = np.array([])
        self.in_nodes = np.array([])
        self.value = ""
    def add_out(self, out_node):
        np.append(self.out_nodes, out_node)

    def add_in(self, in_node):
        np.append(self.in_nodes, in_node)
        in_node.add_out(self)

    def assign_node(self, value):
        self.value = value
    def is_equal(self, node):
        return (self.row == node.row and self.col == node.col)

def initialize_param(N, D):
    Dx = np.array([np.array([np.array(["M", "A", "E", "R"]) for _ in range(D)]) for __ in range(N)])
    X = np.array([np.array([None for _ in range(D)]) for __ in range(N)])
    for r in range(N):
        for c in range(D):
            X[r][c] = Node(r,c)
            if (r >= 1): X[r][c].add_in(X[r-1][c])
            if (c >= 1): X[r][c].add_in(X[r][c-1])
    return Dx, X

def arc_valid(X, Xi, Xj, param):
    '''
    Xi, Xj = Nodes
    return if Xi -> Xj is valid
    '''
    if Xi.row == Xj.row:
        if (Xj.value == "M" and (Xi.value == "M" or Xi.value == "E")):
            return False
        if ((Xj.col+1) % 7 == 0):
            for dist in range(7):
                if (X[Xj.row][Xj.col-dist] == "R"):
                    return True
            return False
        return True
    else:
        count = {
                "m" : 0, "a" : 0, "e" : 0
            }
        for r in range(Xj.row+1):
            count[X[r][Xj.col].value] += 1
        if (Xj.row == param["N"]-1):
            '''Xj in last row'''
            if (count["m"] == param["m"] and count["a"] == param["a"] and count["e"] == param["e"]):
                return True
            return False
        else:
            if (count["m"] > param["m"] or count["a"] > param["a"] or count["e"] > param["e"]):
                return False
            return True
def revise(Xi, Xj, Dx):
    revised = False
    for x_dom in Dx[Xi.row][Xi.col]:
        satisfied = False
        for y_dom in Dx[Xj.row][Xj.col]:
            (xi_init, xj_init) = (Xi.value, Xj.value)
            Xi.value = x_dom
            Xj.value = y_dom
            if (arc_valid(Xi, Xj)):
                satisfied = True
            Xi.value = xi_init
            Xj.value = xj_init
        if not satisfied:
            Dx[Xi.row][Xj.row] = np.delete(Dx[Xi.row][Xj.row], np.argwhere(Dx[Xi.row][Xj.row] == x_dom))
            revised = True
    return revised, Dx

def AC_3(arcs, Dx):
    '''
    arcs <= Queue
    '''
    while(len(arcs) > 0):
        (Xi, Xj) = arcs.pop(0)
        revised, Dx = revise(Xi, Xj, Dx)
        if (revised == True):
            if (len(Dx[Xi.row][Xi.col]) == 0):
                return (False, Dx)
            for in_nbr in Xi.in_nodes:
                if not in_nbr.is_equal(Xj):
                    arcs.append((in_nbr, Xi))
    return (True, Dx)

def back_track(row, col, param, Dx, X):
    if (row == param["N"] or col == param["D"]):
        return X
    new_r, new_c = row, col
    curr_node = X[row][col]
    if (col == param["D"]-1):
        new_r = row+1
        new_c = 0
    else:
        new_c += 1
    for x_dom in Dx[row][col]:
        consistent = True
        for in_node in curr_node.in_nodes:
            curr_node.assign_node(x_dom)
            if not arc_valid(X, in_node, curr_node, param):
                consistent = False
            curr_node.value = ""
        if (consistent):
            curr_node.assign_node(x_dom)
            arcs = []
            for in_node in curr_node.in_nodes:
                arcs.append((in_node, curr_node))
            Dx_new = np.copy(Dx)
            ac_val, Dx_updated = AC_3(arcs, Dx_new)
            if (ac_val):
                result = back_track(new_r, new_c, param, Dx_updated, X)
                #print(result)
                if (len(result) > 0):
                    return result
            curr_node.value = ""
    return []

def solve_csp(input_path):
    param = read_input(input_path)
    Dx, X = initialize_param(param["N"], param["D"])
    return back_track(0, 0, param, Dx, X)

def get_result(X):
    if (X == []):
        ans = dict()
    else:
        N = len(X)
        M = len(X[0])
        ans = dict()
        for i in range(N):
            for j in range(M):
                ans["N{}_{}".format(str(i),str(j))] = X[i][j].value
    with open("solution.json" , 'w') as file:
        json.dump(ans,file)
        file.write("\n")
if __name__ == "__main__":
    input_path = sys.argv[1]
    result = solve_csp(input_path)
    get_result(result)