import json
import random
import copy
import random


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

    def __init__(self, discount, epsilon, success_prob, gridType):
        self.possibleActions = ["N", "S", "W", "E", "U", "D"]
        self.discount = discount
        self.epsilon = epsilon
        self.discountedEpsilon = epsilon*(1-discount)/discount
        self.success_prob = success_prob
        self.passengerPicked = False

        if gridType == 'easy':
            self.generate_easy()
        elif gridType == 'hard':
            self.generate_hard()

    def generate_easy(self):
        # 5*5 matrix
        self.rows = rows = 5
        self.cols = cols = 5
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
                            if tr == 4:
                                node.change_weights(tr+1, tc)
                            if tc == 0:
                                node.change_weights(tr, tc-1)
                            if tc == 4:
                                node.change_weights(tr, tc+1)
                            grid[ps][tr][tc][cr][cc] = node

        for ps in range(2):
            for cr in range(rows):
                for cc in range(cols):
                    grid[ps][0][1][cr][cc].change_weights(0, 2)
                    grid[ps][0][2][cr][cc].change_weights(0, 1)

                    grid[ps][1][1][cr][cc].change_weights(1, 2)
                    grid[ps][1][2][cr][cc].change_weights(1, 1)

                    grid[ps][3][0][cr][cc].change_weights(3, 1)
                    grid[ps][3][1][cr][cc].change_weights(3, 0)

                    grid[ps][3][2][cr][cc].change_weights(3, 3)
                    grid[ps][3][3][cr][cc].change_weights(3, 2)

                    grid[ps][4][0][cr][cc].change_weights(4, 1)
                    grid[ps][4][1][cr][cc].change_weights(4, 0)

                    grid[ps][4][2][cr][cc].change_weights(4, 3)
                    grid[ps][4][3][cr][cc].change_weights(4, 2)

        for tr in range(rows):
            for tc in range(cols):
                for cr in range(rows):
                    for cc in range(cols):
                        if tr != cr or tc != cc:
                            grid[1][tr][tc][cr][cc] = None

        self.grid = grid

        self.Rloc = (0, 0)
        self.Gloc = (0, 4)
        self.Yloc = (4, 0)
        self.Bloc = (4, 3)

        # self.taxi = self.Node(random.randrange(0, 5), random.randrange(0, 5))

        # 2 points for client out of 4 depots cr,cc start.     dr,dc end
        # 1 point for taxi in 5x5 tr,tc start
        # initial state: 0,tr,tc,cr,cc

        # 1-R, 2-G, 3-Y, 4-B
        passenger = random.randrange(1, 5)
        dest = passenger
        while dest == passenger:
            dest = random.randrange(1, 5)

        taxi = (random.randrange(5), random.randrange(5))

        self.taxi = taxi

        def depot(x):
            if x == 1:
                loc = (0, 0)
            elif x == 2:
                loc = (0, 4)
            elif x == 3:
                loc = (4, 0)
            else:
                loc = (4, 3)
            return loc

        self.passenger = depot(passenger)
        self.dest = depot(dest)
        # self.grid[0][self.dest[0]][self.dest[1]
        #                            ][self.dest[0]][self.dest[1]].transitions = {}
        # self.grid[0][self.dest[0]][self.dest[1]][self.dest[0]][self.dest[1]]
        self.grid[1][self.dest[0]][self.dest[1]][self.dest[0]
                                                 ][self.dest[1]].transitions["D"][0]["r"] = 20

    def utilityValue(self, ps, tr, tc, cr, cc):
        val = -float('inf')
        bestAction = -1

        for i in self.grid[ps][tr][tc][cr][cc].transitions:  # Number of actions
            temp = self.Q_Value(ps, tr, tc, cr, cc, i)
            if temp > val:
                val = temp
                bestAction = i

        return val, bestAction

    def Q_Value(self, ps, tr, tc, cr, cc, action):
        node = self.grid[ps][tr][tc][cr][cc]
        q = 0
        for trans in node.transitions[action]:
            nps, ntr, ntc, ncr, ncc = trans["state"]
            if trans["p"] != 0:
                # print(trans)
                q += (trans["p"] * (trans["r"] + self.discount *
                      self.V[nps][ntr][ntc][ncr][ncc]))
        return q

    def value_iteration(self):
        iteration = 0

        while True:
            iteration += 1
            delta = 0.0

            for ps in range(2):
                for tr in range(self.rows):
                    for tc in range(self.cols):
                        for cr in range(self.rows):
                            for cc in range(self.cols):
                                if not self.grid[ps][tr][tc][cr][cc]:
                                    continue
                                self.V_temp[ps][tr][tc][cr][cc], action = self.utilityValue(
                                    ps, tr, tc, cr, cc)
                                self.policy[ps][tr][tc][cr][cc] = action

                                delta = max(delta, abs(
                                    self.V_temp[ps][tr][tc][cr][cc] - self.V[ps][tr][tc][cr][cc]))

            self.V = copy.deepcopy(self.V_temp)

            print("Iteration:{}, Delta:{}".format(iteration, delta))

            if delta <= self.discountedEpsilon:
                break

        print("\nConverged\nIterations:{}, Max-Norm:{}".format(iteration, delta))
        print("Passenger:({},{})".format(
            self.passenger[0], self.passenger[1]))
        print("Destination:({},{})".format(self.dest[0], self.dest[1]))
        print("Taxi:({},{})".format(self.taxi[0], self.taxi[1]))

        tr, tc = self.taxi[0], self.taxi[1]
        cr, cc = self.passenger[0], self.passenger[1]

        action = ''
        picked = False

        while not (tr == self.dest[0] and tc == self.dest[1] and picked == False):
            if not picked:
                action = self.policy[0][tr][tc][cr][cc]
            else:
                action = self.policy[1][tr][tc][tr][tc]
            ret = self.simulate(picked, tr, tc, cr, cc, action)
            (picked, tr, tc, cr, cc) = ret["state"]
            reward = ret["r"]
            print(action, ret)

            # for r in range(self.rows):
            #     for c in range(self.cols):
            #         # print(round(self.utility[r][c], 3), end=" ")
            #         action = self.policy[r][c]
            #         direction = ""
            #         if action == 0:
            #             direction = "↑"
            #         elif action == 1:
            #             direction = "→"
            #         elif action == 2:
            #             direction = "↓"
            #         elif action == 3:
            #             direction = "←"
            #         elif action == 4:
            #             direction = "Pick"
            #         elif action == 5:
            #             direction = "Putdown"
            #         else:
            #             direction = "Error"
            #         print(direction, end=" ")

    def simulate(self, ps, tr, tc, cr, cc, action):
        r = random.random()
        # print(r)
        for trans in self.grid[ps][tr][tc][cr][cc].transitions[action]:
            if trans["p"] == 0:
                continue
            else:
                r -= trans["p"]
            if r < 0:
                return trans


if __name__ == '__main__':
    MDP = MarkovDecisionProblem(0.9, 1e-2, 0.85, 'easy')
    MDP.value_iteration()
