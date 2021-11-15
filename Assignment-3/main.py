import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import math
import warnings
import random
import copy

NORTH   = 0
UP      = 0
SOUTH   = 2
DOWN    = 2
EAST    = 1
RIGHT   = 1
WEST    = 3
LEFT    = 3

PICKUP  = 4 
PUTDOWN = 5

class MarkovDecisionProblem:
    
    class Node:
        def __init__(self, row, col):
            self.row = row
            self.col = col
            self.possibleActions = [i for i in range(6)]
            self.forbiddenActions = []
    
    def __init__(self, discount, epsilon, success_prob, gridType):
        self.discount = discount
        self.epsilon  = epsilon
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
        grid = [[None for i in range(cols)] for j in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                grid[r][c] = self.Node(r, c)
                if r == 0:
                    grid[r][c].forbiddenActions.append(UP)
                if r == 4:
                    grid[r][c].forbiddenActions.append(DOWN)
                if c == 0:
                    grid[r][c].forbiddenActions.append(LEFT)
                if c == 4:
                    grid[r][c].forbiddenActions.append(RIGHT)
        
        grid[0][1].forbiddenActions.append(RIGHT)
        grid[0][2].forbiddenActions.append(LEFT)
        
        grid[1][1].forbiddenActions.append(RIGHT)
        grid[1][2].forbiddenActions.append(LEFT)
        
        grid[3][0].forbiddenActions.append(RIGHT)
        grid[3][1].forbiddenActions.append(LEFT)
        
        grid[4][0].forbiddenActions.append(RIGHT)
        grid[4][1].forbiddenActions.append(LEFT)
        
        grid[3][2].forbiddenActions.append(RIGHT)
        grid[3][3].forbiddenActions.append(LEFT)
        
        grid[4][2].forbiddenActions.append(RIGHT)
        grid[4][3].forbiddenActions.append(LEFT)
        
        self.grid = grid
        
        self.Rloc = self.Node(0,0)
        self.Gloc = self.Node(0,4)
        self.Yloc = self.Node(4,0)
        self.Bloc = self.Node(4,3)

        self.taxi = self.Node(random.randrange(0,5), random.randrange(0,5))
        
        src = random.randrange(1,5)
        dest = src
        while dest == src:
            dest = random.randrange(1,5)
        
        if src == 1:
            self.src = self.Node(self.Rloc.row, self.Rloc.col)
        elif src == 2:
            self.src = self.Node(self.Gloc.row, self.Gloc.col)
        elif src == 3:
            self.src = self.Node(self.Yloc.row, self.Yloc.col)
        else:
            self.src = self.Node(self.Bloc.row, self.Bloc.col)
        
        self.passenger = self.Node(self.src.row, self.src.col)
            
        if dest == 1:
            self.dest = self.Node(self.Rloc.row, self.Rloc.col)
        elif dest == 2:
            self.src = self.Node(self.Gloc.row, self.Gloc.col)
        elif dest == 3:
            self.dest = self.Node(self.Yloc.row, self.Yloc.col)
        else:
            self.dest = self.Node(self.Bloc.row, self.Bloc.col)
    
    def utilityValue(self, r, c):
        val = -float('inf')
        bestAction = -1
        
        for i in range(6): # Number of actions
            temp = self.Q_Value(r, c, i)
            if temp > val:
                val = temp
                bestAction = i
        
        if bestAction == 4 and self.passenger.row == r \
            and self.passenger.col == c and self.passengerPicked == False: # Pickup
                self.passengerPicked = True
        
        if self.passengerPicked == True and bestAction <= 3:
            if bestAction not in self.grid[r][c].forbiddenActions:                
                if bestAction == 0: self.passenger.row = r-1
                if bestAction == 1: self.passenger.col = c+1
                if bestAction == 2: self.passenger.row = r+1
                if bestAction == 3: self.passenger.row = c-1
        
        if bestAction == 5 and self.passengerPicked == True: # Putdown
            self.passengerPicked = False
            
            
        return val, bestAction

    def Q_Value(self, r, c, action):
        if action <= 3: # Nav action
            val = 0
            for i in range(4):
                prob   = self.transitionProbability(r, c, action, i)
                reward = self.rewardValue(r, c, i)
                util_reach_state = self.maxUtilsValue(r, c, i)
                val += prob*(reward + self.discount*util_reach_state)
            return val
        else:
            reward = self.rewardValue(r, c, action)
            util_reach_state = self.maxUtilsValue(r, c, action)            
            
            val = 1.0*(reward + self.discount*util_reach_state)
            return val
    
    def maxUtilsValue(self, r, c, reach_state_action):
        if reach_state_action in self.grid[r][c].forbiddenActions:
            return self.utility[r][c]
        
        elif reach_state_action  == 4 or reach_state_action == 5:
            return self.utility[r][c]
        
        elif reach_state_action == 0:
            return self.utility[r-1][c]
        
        elif reach_state_action == 1:
            return self.utility[r][c+1]
        
        elif reach_state_action == 2:
            return self.utility[r+1][c]
        
        elif reach_state_action == 3:
            return self.utility[r][c-1]

    # Check this function again
    def transitionProbability(self, r, c, action, reach_state):
        if action == reach_state:
            return self.success_prob
        else:
            return (1.0-self.success_prob)/3.0
    
    def rewardValue(self, r, c, reach_state):
        if reach_state <= 3:
            return -1
        
        elif reach_state == 4: # Pickup
            if r == self.passenger.row and c == self.passenger.col:
                return -1
            else:
                return -10
            
        elif reach_state == 5: # Putdown
            if self.passengerPicked == True and r == self.dest.row and c == self.dest.col:
                return 20
            elif self.passengerPicked == True:
                return -1
            else:
                return -10

    def run(self):
        self.utility  = [[0.0 for i in range(self.cols)] for j in range(self.rows)]
        utility_temp  = [[0.0 for i in range(self.cols)] for j in range(self.rows)]
        iteration = 0
        
        print("Passenger:({},{})".format(self.passenger.row, self.passenger.col))
        print("Destination:({},{})".format(self.dest.row, self.dest.col))
        print("Taxi:({},{})".format(self.taxi.row,self.taxi.col))

        while True:
            iteration+=1
            delta = 0.0
            self.passengerPicked = False

            for r in range(self.rows):
                for c in range(self.cols):
                    utility_temp[r][c], action = self.utilityValue(r,c)
                    delta = max(delta, abs(utility_temp[r][c] - self.utility[r][c] ))
            self.utility = copy.deepcopy(utility_temp)

            print("Iteration:{}, Delta:{}".format(iteration, delta))
            
            if delta <= self.discountedEpsilon:
                break
        
        print("\nConverged\nIterations:{}, Max-Norm:{}".format(iteration, delta))
        for r in range(self.rows):
            for c in range(self.cols):
                print(round(self.utility[r][c], 3), end=" ")
            print()
                
if __name__ == '__main__':
    MDP = MarkovDecisionProblem(0.9, 0.01, 0.85, 'easy')
    MDP.run()