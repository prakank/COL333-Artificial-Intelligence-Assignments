import json
from math import inf
import random
import copy
import random
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

inf = 1000000000

map = {
    "N": 0,
    "S": 1,
    "W": 2,
    "E": 3,
    "U": 4,
    "D": 5
}


class MarkovDecisionProblem:

    class Node:
        def __init__(self, ps, tr, tc, cr, cc):
            self.taxi_row = tr
            self.taxi_col = tc
            self.client_row = cr
            self.client_col = cc
            self.pick_state = ps
            self.transitions = {
                "N": [{"state": (ps, tr, tc, cr, cc), "p": 0, "r": -1}, {"state": (ps, tr - 1, tc, cr - ps, cc), "p": 0.85, "r": -1}, {"state": (ps, tr + 1, tc, cr + ps, cc), "p": 0.05, "r": -1}, {"state": (ps, tr, tc-1, cr, cc - ps), "p": 0.05, "r": -1}, {"state": (ps, tr, tc + 1, cr, cc + ps), "p": 0.05, "r": -1}],
                "S": [{"state": (ps, tr, tc, cr, cc), "p": 0, "r": -1}, {"state": (ps, tr - 1, tc, cr - ps, cc), "p": 0.05, "r": -1}, {"state": (ps, tr + 1, tc, cr + ps, cc), "p": 0.85, "r": -1}, {"state": (ps, tr, tc-1, cr, cc - ps), "p": 0.05, "r": -1}, {"state": (ps, tr, tc + 1, cr, cc + ps), "p": 0.05, "r": -1}],
                "E": [{"state": (ps, tr, tc, cr, cc), "p": 0, "r": -1}, {"state": (ps, tr - 1, tc, cr - ps, cc), "p": 0.05, "r": -1}, {"state": (ps, tr + 1, tc, cr + ps, cc), "p": 0.05, "r": -1}, {"state": (ps, tr, tc-1, cr, cc - ps), "p": 0.05, "r": -1}, {"state": (ps, tr, tc + 1, cr, cc + ps), "p": 0.85, "r": -1}],
                "W": [{"state": (ps, tr, tc, cr, cc), "p": 0, "r": -1}, {"state": (ps, tr - 1, tc, cr - ps, cc), "p": 0.05, "r": -1}, {"state": (ps, tr + 1, tc, cr + ps, cc), "p": 0.05, "r": -1}, {"state": (ps, tr, tc-1, cr, cc - ps), "p": 0.85, "r": -1}, {"state": (ps, tr, tc + 1, cr, cc + ps), "p": 0.05, "r": -1}],
                "U": [],
                "D": []
            }
            if tr == cr and tc == cc:
                self.transitions["U"].append(
                    {"state": (1, tr, tc, cr, cc), "p": 1, "r": -1})
                self.transitions["D"].append(
                    {"state": (0, tr, tc, cr, cc), "p": 1, "r": -1})
            elif ps == 0:
                self.transitions["U"].append(
                    {"state": (ps, tr, tc, cr, cc), "p": 1, "r": -10})
                self.transitions["D"].append(
                    {"state": (ps, tr, tc, cr, cc), "p": 1, "r": -10})

        def print(self):
            print(self.pick_state, self.taxi_row, self.taxi_col,
                  self.client_row, self.client_col)
            print(json.dumps(self.transitions))

        def change_weights(self, row, col):
            for action in self.transitions:
                for i in range(len(self.transitions[action])):
                    st = self.transitions[action][i]["state"]
                    if st[1] == row and st[2] == col:
                        self.transitions[action][0]["p"] += self.transitions[action][i]["p"]
                        self.transitions[action][i]["p"] = 0

    def __init__(self, params):
        self.possibleActions = ["N", "S", "W", "E", "U", "D"]
        self.passengerPicked = False
        self.gridType = params['gridType']
        self.generate(params)

    def check_terminal(self, ps, tr, tc, cr, cc):
        if ps == 0 and tr == cr and tc == cc and tr == self.dest[0] and tc == self.dest[1]:
            return True
        else:
            return False

    def generate(self, params):
        self.rows = rows = params['rows']
        self.cols = cols = params['cols']
        grid = [[[[[None for i1 in range(cols)]
                 for j1 in range(rows)] for i2 in range(cols)] for j2 in range(rows)]for k in range(2)]

        self.V = [[[[[0.0 for i1 in range(cols)]
                     for j1 in range(rows)] for i2 in range(cols)] for j2 in range(rows)]for k in range(2)]

        self.policy = [[[[[0.0 for i1 in range(cols)]
                          for j1 in range(rows)] for i2 in range(cols)] for j2 in range(rows)]for k in range(2)]

        self.V_temp = [[[[[0.0 for i1 in range(cols)]
                          for j1 in range(rows)] for i2 in range(cols)] for j2 in range(rows)]for k in range(2)]

        for ps in range(2):
            for tr in range(rows):
                for tc in range(cols):
                    for cr in range(rows):
                        for cc in range(cols):
                            node = self.Node(ps, tr, tc, cr, cc)
                            if tr == 0:
                                node.change_weights(tr-1, tc)
                            if tr == (rows-1):
                                node.change_weights(tr+1, tc)
                            if tc == 0:
                                node.change_weights(tr, tc-1)
                            if tc == (cols-1):
                                node.change_weights(tr, tc+1)
                            grid[ps][tr][tc][cr][cc] = node
        self.grid = grid

        if self.gridType == 'easy':
            self.generate_easy()
        else:
            self.generate_hard()

        for tr in range(rows):
            for tc in range(cols):
                for cr in range(rows):
                    for cc in range(cols):
                        if tr != cr or tc != cc:
                            self.grid[1][tr][tc][cr][cc] = None

        if 'passenger' in params:
            self.passenger = params['passenger']
            self.dest = params['dest']
            self.taxi = params['taxi']
        else:
            passenger = random.randrange(1, self.num_depots + 1)
            dest = passenger
            while dest == passenger:
                dest = random.randrange(1, self.num_depots + 1)

            taxi = (random.randrange(self.rows), random.randrange(self.cols))

            self.taxi = taxi

            def depot(x):
                if x == 1:
                    loc = self.Rloc
                elif x == 2:
                    loc = self.Gloc
                elif x == 3:
                    loc = self.Yloc
                elif x == 4:
                    loc = self.Bloc
                elif x == 5:
                    loc = self.Cloc
                elif x == 6:
                    loc = self.Wloc
                elif x == 7:
                    loc = self.Mloc
                elif x == 8:
                    loc = self.Ploc
                return loc

            self.passenger = depot(passenger)
            self.dest = depot(dest)

        self.grid[1][self.dest[0]][self.dest[1]][self.dest[0]
                                                 ][self.dest[1]].transitions["D"][0]["r"] = 20
        self.grid[0][self.dest[0]][self.dest[1]][self.dest[0]
                                                 ][self.dest[1]].transitions = {}

    def generate_easy(self):
        # Walls
        self.num_depots = 4
        rows, cols = self.rows, self.cols

        for ps in range(2):
            for cr in range(rows):
                for cc in range(cols):
                    self.grid[ps][0][1][cr][cc].change_weights(0, 2)
                    self.grid[ps][0][2][cr][cc].change_weights(0, 1)

                    self.grid[ps][1][1][cr][cc].change_weights(1, 2)
                    self.grid[ps][1][2][cr][cc].change_weights(1, 1)

                    self.grid[ps][3][0][cr][cc].change_weights(3, 1)
                    self.grid[ps][3][1][cr][cc].change_weights(3, 0)

                    self.grid[ps][3][2][cr][cc].change_weights(3, 3)
                    self.grid[ps][3][3][cr][cc].change_weights(3, 2)

                    self.grid[ps][4][0][cr][cc].change_weights(4, 1)
                    self.grid[ps][4][1][cr][cc].change_weights(4, 0)

                    self.grid[ps][4][2][cr][cc].change_weights(4, 3)
                    self.grid[ps][4][3][cr][cc].change_weights(4, 2)

        self.Rloc = (0, 0)
        self.Gloc = (0, 4)
        self.Yloc = (4, 0)
        self.Bloc = (4, 3)

    def generate_hard(self):
        self.num_depots = 8
        rows, cols = self.rows, self.cols

        wall_start = [[0, 2], [2, 5], [0, 7], [6, 0], [6, 3], [6, 7]]

        for ps in range(2):
            for cr in range(rows):
                for cc in range(cols):
                    for w in wall_start:
                        for i in range(4):
                            x = w[0]+i
                            y = w[1]
                            self.grid[ps][x][y][cr][cc].change_weights(x, y+1)
                            self.grid[ps][x][y+1][cr][cc].change_weights(x, y)

        self.Rloc = (0, 0)
        self.Gloc = (0, 5)
        self.Cloc = (0, 8)
        self.Wloc = (3, 3)
        self.Mloc = (4, 6)
        self.Yloc = (8, 0)
        self.Bloc = (9, 4)
        self.Ploc = (9, 9)

    def simulate(self, ps, tr, tc, cr, cc, action):
        r = random.random()
        for trans in self.grid[ps][tr][tc][cr][cc].transitions[action]:
            if trans["p"] == 0:
                continue
            else:
                r -= trans["p"]
            if r < 0:
                return trans
        return None

def utilityValue(MDP, params, ps, tr, tc, cr, cc, policy=False):
    if policy == False:
        val = -float('inf')
        bestAction = ''
        for i in MDP.grid[ps][tr][tc][cr][cc].transitions:  # Number of actions
            temp = Q_Value(MDP, params, ps, tr, tc, cr, cc, i)
            if temp > val:
                val = temp
                bestAction = i

        return val, bestAction
    else:
        val = Q_Value(MDP, params, ps, tr, tc, cr, cc, MDP.policy[ps][tr][tc][cr][cc])
        return val

def Q_Value(MDP, params, ps, tr, tc, cr, cc, action):
    node = MDP.grid[ps][tr][tc][cr][cc]
    q = 0
    for trans in node.transitions[action]:
        nps, ntr, ntc, ncr, ncc = trans["state"]
        if trans["p"] != 0:
            q += (trans["p"] * (trans["r"] + params['discount'] *
                                MDP.V[nps][ntr][ntc][ncr][ncc]))    
    return q

def value_iteration(MDP_params, params, simulate_policy=False, maxStep=23, printOpt=False):
    
    MDP = MarkovDecisionProblem(MDP_params)
    data = []    
    
    iteration = 0
    params['discountedEpsilon'] = params['epsilon'] * \
        (1-params['discount'])/params['discount']

    print('\n\n')
    while True:
        iteration += 1
        delta = 0.0
        for ps in range(2):
            for tr in range(MDP.rows):
                for tc in range(MDP.cols):
                    for cr in range(MDP.rows):
                        for cc in range(MDP.cols):
                            if not MDP.grid[ps][tr][tc][cr][cc]:
                                continue
                            if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                                continue
                            MDP.V_temp[ps][tr][tc][cr][cc], action = utilityValue(
                                MDP, params, ps, tr, tc, cr, cc)
                            MDP.policy[ps][tr][tc][cr][cc] = action

                            delta = max(delta, abs(
                                MDP.V_temp[ps][tr][tc][cr][cc] - MDP.V[ps][tr][tc][cr][cc]))

        MDP.V = copy.deepcopy(MDP.V_temp)

        print("Iteration:{}, Delta:{}".format(iteration, delta))        
        data.append(delta)

        if delta <= params['discountedEpsilon']:
            break

    print("\nConverged\nDiscount:{}, Iterations:{}, Max-Norm:{}".format(params['discount'], iteration, delta))
    print("Passenger:({},{})".format(
        MDP.passenger[0], MDP.passenger[1]))
    print("Destination:({},{})".format(MDP.dest[0], MDP.dest[1]))
    print("Taxi:({},{})".format(MDP.taxi[0], MDP.taxi[1]))

    tr, tc = MDP.taxi[0], MDP.taxi[1]
    cr, cc = MDP.passenger[0], MDP.passenger[1]

    action = ''
    picked = False
    step   = 0
    
    state_action_seq = [(0,tr, tc, cr, cc)]
    
    if printOpt == True:
        line_new = '{:>14}  {:>6}  {:>14}  {:>10}'.format('Current State', 'Action', 'Next State', 'Reward')
        print(line_new)

    while simulate_policy == True and not (tr == MDP.dest[0] and tc == MDP.dest[1] and picked == False and cr == tr and cc == tc) and step <= maxStep:
        if not picked:
            action = MDP.policy[0][tr][tc][cr][cc]
        else:
            action = MDP.policy[1][tr][tc][tr][tc]
        
        state_action_seq.append(action)
        
        ret = MDP.simulate(picked, tr, tc, cr, cc, action)
        (picked, tr, tc, cr, cc) = ret["state"]
        
        state_action_seq.append(ret['r'])
        
        state_action_seq.append((picked, tr, tc, cr, cc))
        
        reward = ret["r"]
        step += 1
        if printOpt == True:
            action_print = ''
            if str(action) == 'N': action_print = 'North'
            elif str(action) == 'S': action_print = 'South'
            elif str(action) == 'E': action_print = 'East'
            elif str(action) == 'W': action_print = 'West'
            elif str(action) == 'U': action_print = 'Pickup'
            elif str(action) == 'D': action_print = 'Putdown'
            print('{:>14}  {:>6}  {:>12}  {:>10}'.
                  format(str(state_action_seq[-4]),action_print,str(state_action_seq[-1]), 
                         str(state_action_seq[-2])))
            # print(action, ret)
            time.sleep(0.04)
    
    if simulate_policy == False:
        return data
    elif simulate_policy == True and maxStep > 21 and printOpt == True:
        return data

    return state_action_seq

def discount_vs_iteration(MDP_params, params):
    discount_list = {0.01:[], 0.1:[], 0.5:[], 0.8:[], 0.99:[]}
    for discount in discount_list:
        params["discount"] = discount
        discount_list[discount] = value_iteration(MDP_params, params)
    
    plt.figure(figsize=(10, 6))
    plt.title('Max-norm vs Iterations (varying discount)',fontsize=12)
    plt.xlabel('Iterations',fontsize=12)
    plt.ylabel('Max-norm',fontsize=12)
    
    for discount in discount_list:
        if discount_list[discount] == None or len(discount_list[discount]) == 0: continue
        x = np.arange(1,len(discount_list[discount])+1)
        plt.plot(x, discount_list[discount], label='discount='+str(discount))

    plt.legend()
    plt.savefig('output/max_norm_vs_iterations.jpg')
    plt.show()
    
def value_iter_parta(MDP_params, params):
    data = value_iteration(MDP_params, params, simulate_policy=True, maxStep=150, printOpt=True)
    plt.figure(figsize=(10, 6))
    plt.title('Value Iteration (discount:{}, epsilon:{})'.format(params['discount'], params['epsilon']),fontsize=12)
    plt.xlabel('Iterations',fontsize=12)
    plt.ylabel('Max-norm',fontsize=12)
    
    plt.plot(np.arange(1,len(data)+1), data)
    plt.savefig('output/value_iteration_parta.jpg')
    plt.show()

def value_iter_partb(MDP_params, params):
    discount_vs_iteration(MDP_params, params)

def value_iter_partc(MDP_params, params, multipleRun=False):
    discount_list = {0.1:[], 0.99:[]}
    
    if multipleRun == False:
        for discount in discount_list:
            params['discount'] = discount
            discount_list[discount] = value_iteration(MDP_params, params, simulate_policy=True)
        
        for discount in discount_list:
            print('\nTaxi:({},{}), Passenger:({},{}), Dest:({},{}), Discount={}'.
                    format(MDP_params['taxi'][0],MDP_params['taxi'][1],MDP_params['passenger'][0],MDP_params['passenger'][1],
                           MDP_params['dest'][0],MDP_params['dest'][1],discount))
            line_new = '{:>14}  {:>6}  {:>14}  {:>10}'.format('Current State', 'Action', 'Next State', 'Reward')
            print(line_new)
            for i in range(0,len(discount_list[discount])-3,3):
                action = ''
                if str(discount_list[discount][i+1]) == 'N': action = 'North'
                elif str(discount_list[discount][i+1]) == 'S': action = 'South'
                elif str(discount_list[discount][i+1]) == 'E': action = 'East'
                elif str(discount_list[discount][i+1]) == 'W': action = 'West'
                elif str(discount_list[discount][i+1]) == 'U': action = 'Pickup'
                elif str(discount_list[discount][i+1]) == 'D': action = 'Putdown'
                
                print('{:>14}  {:>6}  {:>12}  {:>10}'.format(str(discount_list[discount][i]),action,str(discount_list[discount][i+3]), str(discount_list[discount][i+2])))
            print()
    else:
        start_states = {(0,0):{}, (0,4):{}, (4,3):{}}
        
        for i in start_states:
            discount_list = {0.1:[], 0.99:[]}
            MDP_params['passenger'] = i
            MDP_params['taxi'] = (random.randrange(5), random.randrange(5))
            
            for discount in discount_list:
                params['discount'] = discount
                discount_list[discount] = value_iteration(MDP_params, params, simulate_policy=True)
            
            start_states[i]['list'] = discount_list
            start_states[i]['taxi'] = MDP_params['taxi']
            start_states[i]['passenger'] = MDP_params['passenger']
            start_states[i]['dest'] = MDP_params['dest']
            
        for i in start_states:
            discount_list = start_states[i]['list']
            MDP_params['taxi'] = start_states[i]['taxi']
            MDP_params['passenger'] = start_states[i]['passenger']
            MDP_params['dest'] = start_states[i]['dest']
            
            for discount in discount_list:
                print('\nTaxi:({},{}), Passenger:({},{}), Dest:({},{}), Discount={}'.
                        format(MDP_params['taxi'][0],MDP_params['taxi'][1],MDP_params['passenger'][0],MDP_params['passenger'][1],
                            MDP_params['dest'][0],MDP_params['dest'][1],discount))
                line_new = '{:>14}  {:>6}  {:>14}  {:>10}'.format('Current State', 'Action', 'Next State', 'Reward')
                print(line_new)
                for i in range(0,len(discount_list[discount])-3,3):
                    action = ''
                    if str(discount_list[discount][i+1]) == 'N': action = 'North'
                    elif str(discount_list[discount][i+1]) == 'S': action = 'South'
                    elif str(discount_list[discount][i+1]) == 'E': action = 'East'
                    elif str(discount_list[discount][i+1]) == 'W': action = 'West'
                    elif str(discount_list[discount][i+1]) == 'U': action = 'Pickup'
                    elif str(discount_list[discount][i+1]) == 'D': action = 'Putdown'
                    
                    print('{:>14}  {:>6}  {:>12}  {:>10}'.format(str(discount_list[discount][i]),action,str(discount_list[discount][i+3]), str(discount_list[discount][i+2])))
            print('\n\n')

def new_policy(MDP, params, ps, tr, tc, cr, cc):
    _, action = utilityValue(MDP, params, ps, tr, tc, cr, cc)
    return action

def policy_iteration_iterative(MDP_params, params, plotting=False, V_opt=None):
    MDP = MarkovDecisionProblem(MDP_params)
    data = []
    
    if V_opt != None:
        data.append(max(abs(np.min(np.asarray(V_opt))), np.max(np.asarray(V_opt))))
    
    iteration = 0
    
    params['discountedEpsilon'] = params['epsilon'] * \
        (1-params['discount'])/params['discount']
        
    for ps in range(2):
        for tr in range(MDP.rows):
            for tc in range(MDP.cols):
                for cr in range(MDP.rows):
                    for cc in range(MDP.cols):
                        if not MDP.grid[ps][tr][tc][cr][cc]:
                            continue
                        if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                            continue
                        MDP.policy[ps][tr][tc][cr][cc] = MDP.possibleActions[random.randrange(6)]
    while True:
        
        # Policy Evaluation        
        iteration += 1
        delta = 0.0
        converged = True
        max_norm = -inf
        
        for ps in range(2):
            for tr in range(MDP.rows):
                for tc in range(MDP.cols):
                    for cr in range(MDP.rows):
                        for cc in range(MDP.cols):
                            if not MDP.grid[ps][tr][tc][cr][cc]:
                                continue
                            if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                                continue
                            MDP.V_temp[ps][tr][tc][cr][cc] = utilityValue(MDP, params, ps, tr, tc, cr, cc, policy=True)

                            delta = max(delta, abs(
                                MDP.V_temp[ps][tr][tc][cr][cc] - MDP.V[ps][tr][tc][cr][cc]))
                            
                            if V_opt != None:
                                max_norm = max(max_norm, abs(MDP.V_temp[ps][tr][tc][cr][cc] - V_opt[ps][tr][tc][cr][cc]))
        
        if V_opt != None:
            data.append(max_norm)
    
        MDP.V = copy.deepcopy(MDP.V_temp)
        
        # Policy Improvement
        for ps in range(2):
            for tr in range(MDP.rows):
                for tc in range(MDP.cols):
                    for cr in range(MDP.rows):
                        for cc in range(MDP.cols):
                            if not MDP.grid[ps][tr][tc][cr][cc]:
                                continue
                            if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                                continue
                            best_action = new_policy(MDP, params, ps, tr, tc, cr, cc)
                            if best_action != MDP.policy[ps][tr][tc][cr][cc]:
                                MDP.policy[ps][tr][tc][cr][cc] = best_action
                                converged = False
        
        
        print("Iteration:{}, Delta:{}".format(iteration, delta))
        
        if plotting == True and iteration == 50:
            break        
        elif plotting == False and (delta <= params['discountedEpsilon'] or converged):
            break
    
    print("\nConverged\nDiscount:{}, Iterations:{}, Max-Norm:{}".format(params['discount'], iteration, delta))
    print("Passenger:({},{})".format(
        MDP.passenger[0], MDP.passenger[1]))
    print("Destination:({},{})".format(MDP.dest[0], MDP.dest[1]))
    print("Taxi:({},{})".format(MDP.taxi[0], MDP.taxi[1]))

    tr, tc = MDP.taxi[0], MDP.taxi[1]
    cr, cc = MDP.passenger[0], MDP.passenger[1]

    action = ''
    picked = False
    step   = 0

    while not (tr == MDP.dest[0] and tc == MDP.dest[1] and picked == False and cr == tr and cc == tc) and step < 500:
        if not picked:
            action = MDP.policy[0][tr][tc][cr][cc]
        else:
            action = MDP.policy[1][tr][tc][tr][tc]
        ret = MDP.simulate(picked, tr, tc, cr, cc, action)
        (picked, tr, tc, cr, cc) = ret["state"]
        reward = ret["r"]
        step += 1
        # print(action, ret)
        # time.sleep(0.1)
    
    if V_opt != None:
        return data
    
    return MDP.V

def policy_iteration_linear_algebra(MDP_params, params, plotting=False, V_opt=None):
    MDP = MarkovDecisionProblem(MDP_params)
    
    params['discountedEpsilon'] = params['epsilon'] * \
        (1-params['discount'])/params['discount']
        
    state_encoder = {}
    state_decoder = {}
    
    idx = 0
    
    data = []
    
    if V_opt != None:
        data.append(max(abs(np.min(np.asarray(V_opt))), np.max(np.asarray(V_opt))))
    
    for ps in range(2):
        for tr in range(MDP.rows):
            for tc in range(MDP.cols):
                for cr in range(MDP.rows):
                    for cc in range(MDP.cols):                        
                        if not MDP.grid[ps][tr][tc][cr][cc]:
                            continue
                        # if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                        #     continue
                        # MDP.policy[ps][tr][tc][cr][cc] = MDP.possibleActions[random.randrange(6)]
                        MDP.policy[ps][tr][tc][cr][cc] = 'N'
                        val = (ps, tr, tc, cr, cc)
                        state_encoder[val] = idx
                        state_decoder[idx] = val    
                        idx+=1
                        
    iteration = 0
    while True:
        iteration += 1
        delta = 0.0
        converged = True
        
        A = np.zeros((idx, idx))
        B = np.zeros(idx)
        cnt = -1
                            
        for ps in range(2):
            for tr in range(MDP.rows):
                for tc in range(MDP.cols):
                    for cr in range(MDP.rows):
                        for cc in range(MDP.cols):                            
                            if not MDP.grid[ps][tr][tc][cr][cc]:
                                continue
                            cnt+=1
                            # if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                                # continue
                            
                            temp = np.zeros(idx)
                            temp[state_encoder[(ps,tr,tc,cr,cc)]] += 1.0
                            # temp[cnt] += 1.0
                            const = 0.0

                            if MDP.grid[ps][tr][tc][cr][cc].transitions != {}:
                                for s in MDP.grid[ps][tr][tc][cr][cc].transitions[MDP.policy[ps][tr][tc][cr][cc]]:
                                    if s['p'] > 0:
                                        state = s['state']
                                        temp[state_encoder[state]] += (-s['p']*params['discount'])
                                        const += s['p']*s['r']
                                
                            # temp[state_encoder[(ps,tr,tc,cr,cc)]] 
                            
                            A[cnt,:] = temp
                            B[cnt] = const
                            
        # C = np.dot(np.linalg.pinv(A),B)
        C = np.linalg.solve(A,B)
        
        max_norm = -inf
        
        for i in range(len(C)):
            ps, tr, tc, cr, cc = state_decoder[i]
            delta = max(delta, abs(MDP.V[ps][tr][tc][cr][cc] - C[i]))
            MDP.V[ps][tr][tc][cr][cc] = C[i]
            if V_opt != None:
                max_norm = max(max_norm, abs(MDP.V[ps][tr][tc][cr][cc] - V_opt[ps][tr][tc][cr][cc]))
        
        if V_opt != None:
            data.append(max_norm)
            
        
        # Policy Improvement
        for ps in range(2):
            for tr in range(MDP.rows):
                for tc in range(MDP.cols):
                    for cr in range(MDP.rows):
                        for cc in range(MDP.cols):
                            if not MDP.grid[ps][tr][tc][cr][cc]:
                                continue
                            if MDP.grid[ps][tr][tc][cr][cc].transitions == {}:
                                continue
                            best_action = new_policy(MDP, params, ps, tr, tc, cr, cc)
                            if best_action != MDP.policy[ps][tr][tc][cr][cc]:
                                if iteration > 1:
                                    dwn=1
                                    # print(ps, tr, tc, cr, cc, MDP.policy[ps][tr][tc][cr][cc], best_action, MDP.V[ps][tr][tc][cr][cc])
                                MDP.policy[ps][tr][tc][cr][cc] = best_action
                                converged = False
                                
        
        print("Iteration:{}, Delta:{}".format(iteration, delta))

        if plotting == True and iteration >= 10 and (converged or delta <= params['discountedEpsilon']):
            break
        elif plotting == False and (delta <= params['discountedEpsilon'] or converged):
            break
    
    print("\nConverged\nDiscount:{}, Iterations:{}, Max-Norm:{}".format(params['discount'], iteration, delta))
    print("Passenger:({},{})".format(
        MDP.passenger[0], MDP.passenger[1]))
    print("Destination:({},{})".format(MDP.dest[0], MDP.dest[1]))
    print("Taxi:({},{})".format(MDP.taxi[0], MDP.taxi[1]))

    tr, tc = MDP.taxi[0], MDP.taxi[1]
    cr, cc = MDP.passenger[0], MDP.passenger[1]

    action = ''
    picked = False
    step   = 0

    while not (tr == MDP.dest[0] and tc == MDP.dest[1] and picked == False and cr == tr and cc == tc) and step < 500:
        if not picked:
            action = MDP.policy[0][tr][tc][cr][cc]
        else:
            action = MDP.policy[1][tr][tc][tr][tc]
        ret = MDP.simulate(picked, tr, tc, cr, cc, action)
        (picked, tr, tc, cr, cc) = ret["state"]
        reward = ret["r"]
        step += 1
        # print(action, ret)
        # time.sleep(0.1)
    
    if V_opt != None:
        return data
    
    return MDP.V

def policy_loss(MDP_params, params):
    discount_list = {0.01:[], 0.1:[], 0.5:[], 0.8:[], 0.99:[]}
    for discount in discount_list:
        params["discount"] = discount
        V_opt = policy_iteration_iterative(MDP_params, params, plotting=True)
        discount_list[discount] = np.asarray(policy_iteration_iterative(MDP_params, params, plotting=True, V_opt=V_opt))        
        discount_list[discount] = discount_list[discount] / max(discount_list[discount])
    
    plt.figure(figsize=(10, 6))
    plt.title('Policy Loss vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Policy Loss')
    
    for discount in discount_list:
        if len(discount_list[discount]) == 0: continue
        x = np.arange(1,len(discount_list[discount])+1)
        plt.plot(x, discount_list[discount], label='discount='+str(discount))
        
    plt.legend()
    plt.savefig('output/policy_loss_iterative.jpg')
    plt.show()
    
    discount_list = {0.01:[], 0.1:[], 0.5:[], 0.8:[], 0.99:[]}
    for discount in discount_list:
        params["discount"] = discount
        V_opt = policy_iteration_linear_algebra(MDP_params, params, plotting=True)
        discount_list[discount] = np.asarray(policy_iteration_linear_algebra(MDP_params, params, plotting=True, V_opt=V_opt))
        
                
        discount_list[discount] = discount_list[discount] / max(discount_list[discount])
    
    plt.figure(figsize=(10, 6))
    plt.title('Policy Loss vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Policy Loss')
    
    for discount in discount_list:
        if len(discount_list[discount]) == 0: continue
        x = np.arange(1,len(discount_list[discount])+1)
        plt.plot(x, discount_list[discount], label='discount='+str(discount))
        
    plt.legend()
    plt.savefig('output/policy_loss_linear_algebra.jpg')
    plt.show()
    
def evaluate_policy(Q, MDP, discount):
    passenger = MDP.dest
    while(MDP.dest == passenger):
        def depot(x):
            if x == 1:
                loc = MDP.Rloc
            elif x == 2:
                loc = MDP.Gloc
            elif x == 3:
                loc = MDP.Yloc
            elif x == 4:
                loc = MDP.Bloc
            elif x == 5:
                loc = MDP.Cloc
            elif x == 6:
                loc = MDP.Wloc
            elif x == 7:
                loc = MDP.Mloc
            elif x == 8:
                loc = MDP.Ploc
            return loc
        passenger = random.randrange(1, MDP.num_depots+1)
        passenger = depot(passenger)

    ps, tr, tc, cr, cc = 0, random.randrange(
        0, MDP.rows), random.randrange(0, MDP.cols), passenger[0], passenger[1]
    action = ''

    cumulative_discount = 1
    reward = 0.0
    step = 0
    while (not MDP.check_terminal(ps, tr, tc, cr, cc)) and step <= 500:
        step += 1
        best = -inf
        best_action = ""
        # select best action
        for action in MDP.grid[ps][tr][tc][cr][cc].transitions:
            if Q[ps][tr][tc][cr][cc][map[action]] > best or best_action == "":
                best = Q[ps][tr][tc][cr][cc][map[action]]
                best_action = action

        ret = MDP.simulate(ps, tr, tc, cr, cc, best_action)
        (ps, tr, tc, cr, cc) = ret["state"]
        reward += ret["r"] * cumulative_discount
        cumulative_discount *= discount
    return reward


def simulator(Q, MDP, discount):
    passenger = MDP.dest
    while(MDP.dest == passenger):
        def depot(x):
            if x == 1:
                loc = MDP.Rloc
            elif x == 2:
                loc = MDP.Gloc
            elif x == 3:
                loc = MDP.Yloc
            elif x == 4:
                loc = MDP.Bloc
            elif x == 5:
                loc = MDP.Cloc
            elif x == 6:
                loc = MDP.Wloc
            elif x == 7:
                loc = MDP.Mloc
            elif x == 8:
                loc = MDP.Ploc
            return loc
        passenger = random.randrange(1, MDP.num_depots+1)
        passenger = depot(passenger)

    ps, tr, tc, cr, cc = 0, random.randrange(
        0, MDP.rows), random.randrange(0, MDP.cols), passenger[0], passenger[1]
    action = ''

    cumulative_discount = 1
    reward = 0.0
    step = 0
    print(f"Start state: ({ps}, {tr}, {tc}, {cr}, {cc})")

    line_new = '{:>16}  {:>7}  {:>16}  {:>7}'.format(
        'Current State', 'Action', 'Next State', 'Reward')
    print(line_new)
    while (not MDP.check_terminal(ps, tr, tc, cr, cc)) and step <= 500:
        # print(f"State: ({ps}, {tr}, {tc}, {cr}, {cc})")
        step += 1
        best = -inf
        best_action = ""
        # select best action
        for action in MDP.grid[ps][tr][tc][cr][cc].transitions:
            if Q[ps][tr][tc][cr][cc][map[action]] > best or best_action == "":
                best = Q[ps][tr][tc][cr][cc][map[action]]
                best_action = action
        # print(f"Policy prescribes: {best_action}")
        ret = MDP.simulate(ps, tr, tc, cr, cc, best_action)

        print('{:>16}  {:>7}  {:>16}  {:>7}'.format(
            str((ps, tr, tc, cr, cc)), best_action, str(ret["state"]), ret["r"]))

        # print(f"Reached state: {ret['state']}, reward: {ret['r']}")
        (ps, tr, tc, cr, cc) = ret["state"]
        reward += ret["r"] * cumulative_discount
        cumulative_discount *= discount
        # print()
    print(f"Reward: {reward}")


def q_learning(params, episodes, learning_rate, discount, epsilon_exploration=0.1, decay=False):
    MDP = MarkovDecisionProblem(params)
    rewards = []
    Q = [[[[[[0.0 for x in MDP.possibleActions]for i1 in range(MDP.cols)]
            for j1 in range(MDP.rows)] for i2 in range(MDP.cols)] for j2 in range(MDP.rows)]for k in range(2)]
    learning_updates = 0
    for iter in range(1, episodes+1):
        passenger = MDP.dest
        while(MDP.dest == passenger):
            def depot(x):
                if x == 1:
                    loc = MDP.Rloc
                elif x == 2:
                    loc = MDP.Gloc
                elif x == 3:
                    loc = MDP.Yloc
                elif x == 4:
                    loc = MDP.Bloc
                elif x == 5:
                    loc = MDP.Cloc
                elif x == 6:
                    loc = MDP.Wloc
                elif x == 7:
                    loc = MDP.Mloc
                elif x == 8:
                    loc = MDP.Ploc
                return loc
            passenger = random.randrange(1, MDP.num_depots+1)
            passenger = depot(passenger)

        ps, tr, tc, cr, cc = 0, random.randrange(
            0, MDP.rows), random.randrange(0, MDP.cols), passenger[0], passenger[1]
        print("Starting new episode", iter, ps, tr, tc, cr, cc)

        for step in range(1, 501):
            learning_updates += 1
            best = -inf
            best_action = ""

            # find best action for greedy
            for action in MDP.grid[ps][tr][tc][cr][cc].transitions:
                if Q[ps][tr][tc][cr][cc][map[action]] > best or best_action == "":
                    best = Q[ps][tr][tc][cr][cc][map[action]]
                    best_action = action

            r = random.random()
            if decay:
                epsilon_exploration_corrected = epsilon_exploration/learning_updates
            else:
                epsilon_exploration_corrected = epsilon_exploration

            if r < epsilon_exploration_corrected:
                # EXPLORE
                random_action_idx = random.randint(
                    0, len(MDP.grid[ps][tr][tc][cr][cc].transitions) - 1)
                selected_action = list(MDP.grid[ps][tr][tc][cr][cc].transitions.keys())[
                    random_action_idx]
            else:
                # GREEDY
                selected_action = best_action

            # perform selected action
            ret = MDP.simulate(ps, tr, tc, cr, cc, selected_action)

            # learn
            nps, ntr, ntc, ncr, ncc = ret["state"]
            next_state_max_Q = -inf
            best_action = ""
            for action in MDP.grid[nps][ntr][ntc][ncr][ncc].transitions:
                if Q[nps][ntr][ntc][ncr][ncc][map[action]] > next_state_max_Q or best_action == "":
                    next_state_max_Q = Q[nps][ntr][ntc][ncr][ncc][map[action]]
                    best_action = action
            if MDP.check_terminal(nps, ntr, ntc, ncr, ncc):
                next_state_max_Q = 0

            Q[ps][tr][tc][cr][cc][map[selected_action]] = (
                1-learning_rate)*Q[ps][tr][tc][cr][cc][map[selected_action]] + learning_rate * (ret["r"] + discount * next_state_max_Q)

            ps, tr, tc, cr, cc = ret["state"]

            if MDP.check_terminal(ps, tr, tc, cr, cc):
                # print(step)
                break

        if iter % 50 == 0:
            reward = 0
            for i in range(50):
                reward += evaluate_policy(Q, MDP, discount)
            reward /= 50
            rewards.append((iter, reward))
    return Q, rewards


def sarsa(params, episodes, learning_rate, discount, epsilon_exploration=0.1, decay=False):
    MDP = MarkovDecisionProblem(params)
    eval_after_episodes = episodes/40
    rewards = []
    Q = [[[[[[0.0 for x in MDP.possibleActions]for i1 in range(MDP.cols)]
            for j1 in range(MDP.rows)] for i2 in range(MDP.cols)] for j2 in range(MDP.rows)]for k in range(2)]
    learning_updates = 0
    for iter in range(1, episodes+1):
        passenger = MDP.dest
        while(MDP.dest == passenger):
            def depot(x):
                if x == 1:
                    loc = MDP.Rloc
                elif x == 2:
                    loc = MDP.Gloc
                elif x == 3:
                    loc = MDP.Yloc
                elif x == 4:
                    loc = MDP.Bloc
                elif x == 5:
                    loc = MDP.Cloc
                elif x == 6:
                    loc = MDP.Wloc
                elif x == 7:
                    loc = MDP.Mloc
                elif x == 8:
                    loc = MDP.Ploc
                return loc
            passenger = random.randrange(1, MDP.num_depots+1)
            passenger = depot(passenger)

        ps, tr, tc, cr, cc = 0, random.randrange(
            0, MDP.rows), random.randrange(0, MDP.cols), passenger[0], passenger[1]
        print("Starting new episode", iter, ps, tr, tc, cr, cc)
        best = -inf
        best_action = ""

        # find best action for greedy
        for action in MDP.grid[ps][tr][tc][cr][cc].transitions:
            if Q[ps][tr][tc][cr][cc][map[action]] > best or best_action == "":
                best = Q[ps][tr][tc][cr][cc][map[action]]
                best_action = action
        r = random.random()

        if r < epsilon_exploration:
            # EXPLORE
            random_action_idx = random.randint(
                0, len(MDP.grid[ps][tr][tc][cr][cc].transitions) - 1)
            selected_action = list(MDP.grid[ps][tr][tc][cr][cc].transitions.keys())[
                random_action_idx]
        else:
            # GREEDY
            selected_action = best_action

        for step in range(1, 501):
            learning_updates += 1
            if decay:
                epsilon_exploration_corrected = epsilon_exploration/learning_updates
            else:
                epsilon_exploration_corrected = epsilon_exploration
            # perform selected action
            ret = MDP.simulate(ps, tr, tc, cr, cc, selected_action)

            # learn
            nps, ntr, ntc, ncr, ncc = ret["state"]
            next_state_max_Q = -inf
            best_action = ""
            for action in MDP.grid[nps][ntr][ntc][ncr][ncc].transitions:
                if Q[nps][ntr][ntc][ncr][ncc][map[action]] > next_state_max_Q or best_action == "":
                    next_state_max_Q = Q[nps][ntr][ntc][ncr][ncc][map[action]]
                    best_action = action

            r = random.random()
            next_state_Q = 0
            if MDP.check_terminal(nps, ntr, ntc, ncr, ncc):
                next_state_Q = 0
            elif r < epsilon_exploration_corrected:
                # EXPLORE
                random_action_idx = random.randint(
                    0, len(MDP.grid[ps][tr][tc][cr][cc].transitions) - 1)
                selected_action_next_state = list(MDP.grid[ps][tr][tc][cr][cc].transitions.keys())[
                    random_action_idx]
                next_state_Q = Q[nps][ntr][ntc][ncr][ncc][map[selected_action_next_state]]
            else:
                # GREEDY
                selected_action_next_state = best_action
                next_state_Q = next_state_max_Q

            Q[ps][tr][tc][cr][cc][map[selected_action]] = (
                1-learning_rate)*Q[ps][tr][tc][cr][cc][map[selected_action]] + learning_rate * (ret["r"] + discount * next_state_Q)

            ps, tr, tc, cr, cc = ret["state"]
            selected_action = selected_action_next_state
            if MDP.check_terminal(ps, tr, tc, cr, cc):
                # print(step)
                break

        if iter % eval_after_episodes == 0:
            reward = 0
            for i in range(50):
                reward += evaluate_policy(Q, MDP, discount)
            reward /= 50
            rewards.append((iter, reward))
    return Q, rewards


def make_plot_1(rewards_Q, rewards_Q_decay, rewards_sarsa, rewards_sarsa_decay):
    plt.figure(figsize=(10, 6))
    plt.title('Rewards vs Iterations for different algorithms')
    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Rewards')

    x, y = zip(*rewards_Q)
    plt.plot(x, y, label='Q-learning')
    x, y = zip(*rewards_Q_decay)
    plt.plot(x, y, label='Q-learning with decaying exploration')
    x, y = zip(*rewards_sarsa)
    plt.plot(x, y, label='SARSA')
    x, y = zip(*rewards_sarsa_decay)
    plt.plot(x, y, label='SARSA with decaying exploration')
    plt.legend()
    plt.savefig('plot_B2.jpg')
    print(rewards_Q[-1])
    print(rewards_Q_decay[-1])
    print(rewards_sarsa[-1])
    print(rewards_sarsa_decay[-1])
    # plt.show()


def make_plot_5(rewards, dest_arr):
    plt.figure(figsize=(10, 6))
    plt.title('Rewards vs Iterations for different algorithms')
    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Rewards')
    for i in range(len(dest_arr)):
        x, y = zip(*rewards[i])
        plt.plot(
            x, y, label="dest=("+str(dest_arr[i][0]) + "," + str(dest_arr[i][1])+")")
        print(dest_arr[i], rewards[i][-1])
    plt.legend()
    plt.savefig('plot_B5.jpg')
    plt.show()


def make_plot_3_eps(epsilons, rewards):
    plt.figure(figsize=(10, 6))
    plt.title('Rewards vs Iterations(varying exploration)')
    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Rewards')
    for i in range(len(rewards)):
        x, y = zip(*rewards[i])
        plt.plot(x, y, label='epsilon='+str(epsilons[i]))
    plt.legend()
    plt.savefig('plot_B4e.jpg')
    plt.show()


def make_plot_3_alpha(alphas, rewards):
    plt.figure(figsize=(10, 6))
    plt.title('Rewards vs Iterations(varying learning rate)')
    plt.xlabel('Number of Training Episodes')
    plt.ylabel('Rewards')
    for i in range(len(rewards)):
        x, y = zip(*rewards[i])
        plt.plot(x, y, label='alpha='+str(alphas[i]))
    plt.legend()
    plt.savefig('plot_B4a.jpg')
    plt.show()

if __name__ == '__main__':

    params_easy = {
        'gridType': 'easy',
        'rows': 5,
        'cols': 5,
        'passenger': (4, 3),
        'dest': (4, 0),
        'taxi': (1, 4)
    }
    params_hard = {
        'gridType': 'hard',
        'rows': 10,
        'cols': 10
    }

    value_iter_params = {
        'discount': 0.9,
        'epsilon': 1e-6,
        'success_prob': 0.85
    }
    
    # Value Iteration with plot (max-norm vs iteration)
    value_iter_parta(params_easy, value_iter_params)
    
    # Loss vs Iteration (varying discount)
    value_iter_partb(params_easy, value_iter_params)
    
    # State Action Seq for gamma=0.1, gamma=0.99
    value_iter_partc(params_easy, value_iter_params, True)    

    # policy_iteration_iterative(params_easy, value_iter_params)
    # policy_iteration_linear_algebra(params_easy, value_iter_params)
    
    # Policy Loss (with linear algebra, varying discount)
    policy_loss(params_easy, value_iter_params)

    MDP = MarkovDecisionProblem(params=params_easy)
    MDP10 = MarkovDecisionProblem(params=params_hard)
    Q_q, rewards_Q = q_learning(params_easy, 2000, 0.25, 0.99, 0.1, False)
    Q_q_decay, rewards_Q_decay = q_learning(
        params_easy, 2000, 0.25, 0.99, 0.1, True)
    Q_sarsa, rewards_sarsa = sarsa(params_easy, 2000, 0.25, 0.99, 0.1, False)
    Q_sarsa_decay, rewards_sarsa_decay = sarsa(
        params_easy, 2000, 0.25, 0.99, 0.1, True)
    eps_vals = [0, 0.05, 0.1, 0.5, 0.9]
    alpha_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    rewards_eps = []
    rewards_alpha = []
    for eps in eps_vals:
        rewards_eps.append(q_learning(
            params_easy, 2000, 0.1, 0.99, eps, False)[1])

    for alpha in alpha_vals:
        rewards_alpha.append(q_learning(
            params_easy, 2000, alpha, 0.99, 0.1, False)[1])

    make_plot_1(rewards_Q, rewards_Q_decay, rewards_sarsa, rewards_sarsa_decay)
    make_plot_3_eps(eps_vals, rewards_eps)
    make_plot_3_alpha(alpha_vals, rewards_alpha)

    def B5():
        depot_arr = [1, 2, 3, 4, 5, 6, 7, 8]
        np.random.shuffle(depot_arr)
        reward = 0.0
        rewards_10 = []
        dest_arr = []
        for i in range(5):
            def depot(x):
                if x == 1:
                    loc = MDP10.Rloc
                elif x == 2:
                    loc = MDP10.Gloc
                elif x == 3:
                    loc = MDP10.Yloc
                elif x == 4:
                    loc = MDP10.Bloc
                elif x == 5:
                    loc = MDP10.Cloc
                elif x == 6:
                    loc = MDP10.Wloc
                elif x == 7:
                    loc = MDP10.Mloc
                elif x == 8:
                    loc = MDP10.Ploc
                return loc
            params_10 = {
                'gridType': 'hard',
                'rows': 10,
                'cols': 10,
                'passenger': depot(depot_arr[MDP10.num_depots - 1]),
                'dest': depot(depot_arr[i]),
                'taxi': (random.randrange(0, MDP10.rows), random.randrange(0, MDP10.cols))
            }
            dest_arr.append(depot(depot_arr[i]))
            Q_q_decay_10, rewards_Q_decay_10 = q_learning(
                params_10, 10000, 0.25, 0.99, 0.1, True)
            rewards_10.append(rewards_Q_decay_10)
            MDP_temp = MarkovDecisionProblem(params=params_10)
            # reward = 0
            # for i in range(50):
            #     reward += evaluate_policy(Q_q_decay_10, MDP_temp, 0.99)
            # rewards_arr.append(reward/50)

        make_plot_5(rewards_10, dest_arr)
    B5()

    for i in range(5):
        print()
        simulator(Q_q_decay, MDP, 0.99)    
