import json
from math import inf
import random
import copy
import random
import time
inf = 1000000000


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
            passenger = random.randrange(1, 5)
            dest = passenger
            while dest == passenger:
                dest = random.randrange(1, 5)

            taxi = (random.randrange(5), random.randrange(5))

            self.taxi = taxi

            def depot(x):
                if x == 1:
                    loc = self.Rloc
                elif x == 2:
                    loc = self.Gloc
                elif x == 3:
                    loc = self.Yloc
                else:
                    loc = self.Bloc
                return loc

            self.passenger = depot(passenger)
            self.dest = depot(dest)

        self.grid[1][self.dest[0]][self.dest[1]][self.dest[0]
                                                 ][self.dest[1]].transitions["D"][0]["r"] = 20
        self.grid[0][self.dest[0]][self.dest[1]][self.dest[0]
                                                 ][self.dest[1]].transitions = {}

    def generate_easy(self):
        # Walls
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


def utilityValue(MDP, params, ps, tr, tc, cr, cc):
    val = -float('inf')
    bestAction = -1

    for i in MDP.grid[ps][tr][tc][cr][cc].transitions:  # Number of actions
        temp = Q_Value(MDP, params, ps, tr, tc, cr, cc, i)
        if temp > val:
            val = temp
            bestAction = i

    return val, bestAction


def Q_Value(MDP, params, ps, tr, tc, cr, cc, action):
    node = MDP.grid[ps][tr][tc][cr][cc]
    q = 0
    for trans in node.transitions[action]:
        nps, ntr, ntc, ncr, ncc = trans["state"]
        if trans["p"] != 0:
            q += (trans["p"] * (trans["r"] + params['discount'] *
                                MDP.V[nps][ntr][ntc][ncr][ncc]))
    return q


def value_iteration(MDP, params):
    iteration = 0
    params['discountedEpsilon'] = params['epsilon'] * \
        (1-params['discount'])/params['discount']

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

        if delta <= params['discountedEpsilon']:
            break

    print("\nConverged\nIterations:{}, Max-Norm:{}".format(iteration, delta))
    print("Passenger:({},{})".format(
        MDP.passenger[0], MDP.passenger[1]))
    print("Destination:({},{})".format(MDP.dest[0], MDP.dest[1]))
    print("Taxi:({},{})".format(MDP.taxi[0], MDP.taxi[1]))

    tr, tc = MDP.taxi[0], MDP.taxi[1]
    cr, cc = MDP.passenger[0], MDP.passenger[1]

    action = ''
    picked = False

    while not (tr == MDP.dest[0] and tc == MDP.dest[1] and picked == False and cr == tr and cc == tc):
        if not picked:
            action = MDP.policy[0][tr][tc][cr][cc]
        else:
            action = MDP.policy[1][tr][tc][tr][tc]
        ret = MDP.simulate(picked, tr, tc, cr, cc, action)
        (picked, tr, tc, cr, cc) = ret["state"]
        reward = ret["r"]
        print(action, ret)


def q_learning(params, episodes, learning_rate, discount, epsilon_exploration=0.1, decay=False):
    MDP = MarkovDecisionProblem(params)

    Q = [[[[[[0.0 for x in MDP.possibleActions]for i1 in range(MDP.cols)]
            for j1 in range(MDP.rows)] for i2 in range(MDP.cols)] for j2 in range(MDP.rows)]for k in range(2)]
    # for ps in 2:
    #     for tr in range(MDP.rows):
    #         for tc in range(MDP.cols):
    #             for cr in range(MDP.rows):
    #                 for cc in range(MDP.cols):
    #                     for action in MDP.actions:
    map = {
        "N": 0,
        "S": 1,
        "W": 2,
        "E": 3,
        "U": 4,
        "D": 5
    }

    for iter in range(episodes):
        passenger = MDP.dest
        while(MDP.dest == passenger):
            def depot(x):
                if x == 1:
                    loc = MDP.Rloc
                elif x == 2:
                    loc = MDP.Gloc
                elif x == 3:
                    loc = MDP.Yloc
                else:
                    loc = MDP.Bloc
                return loc
            passenger = random.randrange(1, 5)
            passenger = depot(passenger)

        ps, tr, tc, cr, cc = 0, random.randrange(
            0, 5), random.randrange(0, 5), passenger[0], passenger[1]
        print("Starting new episode", iter, ps, tr, tc, cr, cc)

        for step in range(500):
            best = -inf
            best_action = ""

            # select best action
            for action in MDP.grid[ps][tr][tc][cr][cc].transitions:
                if Q[ps][tr][tc][cr][cc][map[action]] > best or best_action == "":
                    best = Q[ps][tr][tc][cr][cc][map[action]]
                    best_action = action
            r = random.random()
            if r < epsilon_exploration:
                random_action_idx = random.randint(
                    0, len(MDP.grid[ps][tr][tc][cr][cc].transitions) - 1)
                selected_action = list(MDP.grid[ps][tr][tc][cr][cc].transitions.keys())[
                    random_action_idx]
            else:
                selected_action = best_action

            # perform selected action
            # print("Step:", step, ps, tr, tc, cr, cc, selected_action)
            ret = MDP.simulate(ps, tr, tc, cr, cc, selected_action)

            # learn
            nps, ntr, ntc, ncr, ncc = ret["state"]
            next_state_max_Q = -inf
            best_action = ""
            for action in MDP.grid[nps][ntr][ntc][ncr][ncc].transitions:
                if Q[nps][ntr][ntc][ncr][ncc][map[action]] > next_state_max_Q or best_action == "":
                    next_state_max_Q = Q[ps][tr][tc][cr][cc][map[action]]
                    best_action = action
            if MDP.check_terminal(nps, ntr, ntc, ncr, ncc):
                next_state_max_Q = 0

            Q[ps][tr][tc][cr][cc][map[selected_action]] = (
                1-learning_rate)*Q[ps][tr][tc][cr][cc][map[selected_action]] + learning_rate * (ret["r"] + discount * next_state_max_Q)
            ps, tr, tc, cr, cc = ret["state"]

            if MDP.check_terminal(ps, tr, tc, cr, cc):
                # print(step)
                break
            # print(Q[1][4][0][4][0])
            # input()
        if step != 499:
            print(step)
            time.sleep(0.1)


if __name__ == '__main__':

    params = {
        'gridType': 'easy',
        'rows': 5,
        'cols': 5,
        'passenger': (4, 3),
        'dest': (4, 0),
        'taxi': (1, 4)
    }

    value_iter_params = {
        'discount': 0.9,
        'epsilon': 1e-6,
        'success_prob': 0.85
    }

    MDP = MarkovDecisionProblem(params=params)
    # q_learning(params, 2000, 0.25, 0.99, 0.1, False)

    value_iteration(MDP, value_iter_params)
