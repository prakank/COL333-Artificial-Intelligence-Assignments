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
        self.out_nodes = []
        self.in_nodes = []
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
    # Dx = np.array([np.array([np.array(["M", "A", "E", "R"]) for _ in range(D)]) for __ in range(N)])
    Dx = np.array([np.array([np.array(["A", "M", "E", "R"]) for _ in range(D)]) for __ in range(N)])
    X = np.array([np.array([None for _ in range(D)]) for __ in range(N)])
    for r in range(N):
        for c in range(D):
            X[r][c] = Node(r,c)
            if (r >= 1):
                X[r][c].in_nodes.append(X[r-1][c])
                X[r-1][c].out_nodes.append(X[r][c])
            if (c >= 1):
                X[r][c].in_nodes.append(X[r][c-1])
                X[r][c-1].out_nodes.append(X[r][c])
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
                if (X[Xj.row][Xj.col-dist].value == "R"):
                    return True
            return False
        return True
    else:
        count = {"m" : 0, "a" : 0, "e" : 0, "r" : 0}
        for r in range(Xj.row+1):
            count[(X[r][Xj.col].value).lower()] += 1
            
        if (Xj.row == param["N"]-1):
            '''Xj in last row'''
            if (count["m"] == param["m"] and count["a"] == param["a"] and count["e"] == param["e"]):
                return True
            return False
        else:
            if (count["m"] > param["m"] or count["a"] > param["a"] or count["e"] > param["e"]):
                return False
            return True
        
def revise(X, Xi, Xj, Dx, param):
    revised = False
    for x_dom in Dx[Xi.row][Xi.col]:
        if x_dom == '.':
            continue
        satisfied = False
        
        for y_dom in Dx[Xj.row][Xj.col]:
            if y_dom == '.':
                continue            
            (xi_init, xj_init) = (Xi.value, Xj.value)
            Xi.value = x_dom
            Xj.value = y_dom
            if (arc_valid(X, Xi, Xj, param)):
                satisfied = True
            Xi.value = xi_init
            Xj.value = xj_init
        
        if not satisfied:            
            index = np.argwhere(Dx[Xi.row][Xi.col] == x_dom)[0][0]
            Dx[Xi.row][Xi.col][index] = '.'            
            revised = True
            
    return revised, Dx

def AC_3(X, arcs, Dx, param):
    '''
    arcs <= Queue
    '''
    while(len(arcs) > 0):
        (Xi, Xj) = arcs.pop(0)
        revised, Dx = revise(X, Xi, Xj, Dx, param)
        if (revised == True):
            if (np.count_nonzero(Dx[Xi.row][Xi.col] == '.') == 4):
                return (False, Dx)
            for in_nbr in Xi.in_nodes:
                if not in_nbr.is_equal(Xj):
                    arcs.append((in_nbr, Xi))
    return (True, Dx)

count = 0
def back_track(row, col, param, Dx, X):    
    if (col == param["D"]):
        row = row+1
        col = 0    
    if row >= param["N"] or col >= param["D"]:
        return X    
    curr_node = X[row][col]
    
    global count
    print("Count: {}, Row: {}, Col: {}".format(count, row, col))
    count+=1
    
    # if col > 5:
    #     print("Start: Row:{}, Col:{}".format(row,col))
    #     for i in range(row+1):
    #         p=False
    #         for j in range(col+1):
    #             if i==row and j==col:
    #                 p = True
    #                 break
    #             print(X[i][j].value,end=" ")        
    #         print("")
    #         if p:
    #             break
    
    for x_dom in Dx[row][col]:
        if x_dom == '.':
            continue        
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
            ac_val, Dx_updated = AC_3(X, arcs, Dx_new, param)
            if (ac_val):
                result = back_track(row, col+1, param, Dx_updated, X)
                if (len(result) > 0):
                    return result
            curr_node.value = ""
    return []

def solve_csp(input_path):
    param = read_input(input_path)
    print(param)
    if param["m"] + param["a"] + param["e"] > param["N"]:
        print("Invalid arguments for m, a, e, N")
        print( "(m + a + e <= N) should hold")
        sys.exit(1)
    
    if param["D"] >= 7 and param["m"] + param["a"] + param["e"] == param["N"]:
        print("Invalid arguments for m, a, e, N")
        print( "(m + a + e < N) should hold {Strictly less cause each nurse needs to have atleast 1 R shift in a week}")
        sys.exit(1)
            
    Dx, X = initialize_param(param["N"], param["D"])
    return back_track(0, 0, param, Dx, X)

def get_result(X):
    if len(X) == 0:
        ans = dict()
    else:
        N = len(X)
        M = len(X[0])
        ans = dict()
        for i in range(N):
            print("N-"+str(i)+": ",end="")
            for j in range(M):
                print(X[i][j].value + " ",end=" ")
                ans["N{}_{}".format(str(i),str(j))] = X[i][j].value
            print()
            
    with open("solution.json" , 'w') as file:
        json.dump(ans,file)
        file.write("\n")
        
if __name__ == "__main__":
    input_path = sys.argv[1]
    result = solve_csp(input_path)
    get_result(result)
    

# Look out for these settings
