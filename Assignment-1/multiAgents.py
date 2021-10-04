# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        "*** YOUR CODE HERE ***"
        
        score = successorGameState.getScore()
        Feasible_Distance = 5
        alpha_feasible = 500
        width, height = newFood.width, newFood.height
        ghost_vicinity = False
        
        
        # Taking the account the ghost's position
        for i in range(len(newGhostStates)):
            dist = manhattanDistance(newPos,newGhostStates[i].getPosition())
            if dist <= Feasible_Distance:
                ghost_vicinity = True
                if(dist > 0):                    
                    score -= alpha_feasible*(1/dist)
        
        # Taking into account the location of the food        
        
        if newFood.data[newPos[0]][newPos[1]] == True:
            return score
        else:
            min_dist = width + height
            for i in range(width):
                for j in range(height):
                    if newFood.data[i][j] == True:
                        min_dist = min(min_dist, manhattanDistance(newPos,(i,j)))

            return score + 1.0/float(min_dist)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        action = None
        score  = None
        legalMoves = gameState.getLegalActions(0)
        
        def min_value(agentIndex, depth, state): # agentIndex is required for Arbitrary number of ghosts
            
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0: # Pacman                
                return max_value(depth+1,state)
            
            # if depth == self.depth or state.isLose() or state.isWin():
                
            min_val = None
            for move in state.getLegalActions(agentIndex):
                temp_val = min_value( (agentIndex+1)%state.getNumAgents(), depth, state.generateSuccessor(agentIndex,move))
                if min_val == None or min_val > temp_val:
                    min_val = temp_val
            return min_val

        def max_value(depth, state):
            if depth == self.depth:
                return self.evaluationFunction(state)
            
            max_val = None
            legalActions = state.getLegalActions(0)
            
            for move in legalActions:
                temp_val = min_value( 1, depth, state.generateSuccessor(0,move))
                if max_val == None or max_val < temp_val:
                    max_val = temp_val
            return max_val
            
        #     -     Layer 0
        #   /   \
        #  -     -  Layer 1
        
        # On iterating over all possible actions, we generate a min state
        # And of all such min states, we have to choose the max one
        
        for move in legalMoves:
            successorGameState = gameState.generateSuccessor(0,move)
            temp_score = min_value(1,0,successorGameState)
            
            if score == None or temp_score > score:
                score = temp_score
                action = move
        
        # print("Optimal Value: {}".format(score))
        return action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
              
        def min_value(alpha, beta, agentIndex, depth, state): # agentIndex is required for Arbitrary number of ghosts            
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0: # Pacman                
                return max_value(alpha, beta, depth+1,state)
            min_val = None
            
            for move in state.getLegalActions(agentIndex):
                temp_val = min_value(alpha, beta, (agentIndex+1)%state.getNumAgents(), depth, state.generateSuccessor(agentIndex,move))
                
                if min_val == None or min_val > temp_val:
                    min_val = temp_val                
                if min_val < alpha:
                    return min_val                
                
                beta = min(beta, min_val)
                
            return min_val

        def max_value(alpha, beta, depth, state):
            if depth == self.depth:
                return self.evaluationFunction(state)
            
            max_val = None
            legalActions = state.getLegalActions(0)
            
            for move in legalActions:
                temp_val = min_value(alpha, beta, 1, depth, state.generateSuccessor(0,move))
                
                if max_val == None or max_val < temp_val:
                    max_val = temp_val
                if max_val > beta:
                    return max_val
                
                alpha = max(alpha, max_val)
                
            return max_val
            
        #     -     Layer 0
        #   /   \
        #  -     -  Layer 1
        
        # On iterating over all possible actions, we generate a min state
        # And of all such min states, we have to choose the max one
                     
        alpha, beta = -float("inf"), float("inf")   
        action = None
        score  = None
        legalMoves = gameState.getLegalActions(0)
        
        for move in legalMoves:
            successorGameState = gameState.generateSuccessor(0,move)
            temp_score = min_value(alpha, beta, 1,0,successorGameState)            
            if score == None or temp_score > score:
                score = temp_score
                action = move
            if score > beta:
                return action
            alpha = max(alpha, score)
        
        # print("Optimal Value: {}".format(score))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
              
        def expecti_val(alpha, beta, agentIndex, depth, state): # agentIndex is required for Arbitrary number of ghosts            
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0: # Pacman                
                return max_value(alpha, beta, depth+1,state)
            min_val = 0.0
            
            legalActions = state.getLegalActions(agentIndex)
            
            for move in legalActions:
                temp_val = expecti_val(alpha, beta, (agentIndex+1)%state.getNumAgents(), depth, state.generateSuccessor(agentIndex,move))                
                min_val += temp_val
                
                # if min_val == None or min_val > temp_val:
                #     min_val = temp_val                
                # if min_val < alpha:
                #     return min_val                
                
                # beta = min(beta, min_val)
            
            return float(min_val)/float(len(legalActions))

        def max_value(alpha, beta, depth, state):
            if depth == self.depth:
                return self.evaluationFunction(state)
            
            max_val = None
            legalActions = state.getLegalActions(0)
            
            for move in legalActions:
                temp_val = expecti_val(alpha, beta, 1, depth, state.generateSuccessor(0,move))
                
                if max_val == None or max_val < temp_val:
                    max_val = temp_val
                # if max_val > beta:
                #     return max_val
                
                # alpha = max(alpha, max_val)
                
            return max_val
            
        #     -     Layer 0
        #   /   \
        #  -     -  Layer 1
        
        # On iterating over all possible actions, we generate a min state
        # And of all such min states, we have to choose the max one
                     
        alpha, beta = -float("inf"), float("inf")   
        action = None
        score  = None
        legalMoves = gameState.getLegalActions(0)
        
        for move in legalMoves:
            successorGameState = gameState.generateSuccessor(0,move)
            temp_score = expecti_val(alpha, beta, 1,0,successorGameState)
            if score == None or temp_score > score:
                score = temp_score
                action = move
            if score > beta:
                return action
            alpha = max(alpha, score)
        
        # print("Optimal Value: {}".format(score))
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    # if current state is lose or win, simply return the state score
    if currentGameState.isWin():
        return 1e7
    if currentGameState.isLose():
        return -1e7
    
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]    
    
    
    # Features to include
    # 1.) Distance to closest ghost
    # 2.) Distance to closest food
    # 3.) Distance to pellet
    # 4.) Count of food

    pos    = currentGameState.getPacmanPosition()
    food   = currentGameState.getFood().asList()
    pellet = currentGameState.getCapsules()

    if len(food) > 0:
        closestFood = min(map(lambda x: manhattanDistance(pos, x), food))
    else:
        closestFood = float("inf")
    
    if len(pellet) > 0:
        closestPellet = min(map(lambda x: manhattanDistance(pos, x), pellet))
    else:
        closestPellet = float("inf")

    activeGhosts = []
    scaredGhosts = []

    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer > 0: # scared
            scaredGhosts.append(ghost)
        else:
            activeGhosts.append(ghost)

    # Active
    if(len(activeGhosts) > 0):
        closestActiveGhostDistance = min(map(lambda x: manhattanDistance(pos,x.getPosition()), activeGhosts))
    else:
        closestActiveGhostDistance = float("inf")
        
    # Scared    
    if(len(scaredGhosts) > 0):
        closestScaredGhostDistance = min(map(lambda x: manhattanDistance(pos,x.getPosition()), scaredGhosts))
    else:
        closestScaredGhostDistance = float("inf")
    
    parameters = {
        "activeGhost"  : -20,
        "scaredGhost"  :  10,
        "closestFood"  :   1.2,
        "closestPellet":   5
    }
            
    # parameters = {
    #     "activeGhost"  : -10,
    #     "scaredGhost"  :  10,
    #     "closestFood"  :   2,
    #     "closestPellet":   5
    # }
    
    Finalscore = currentGameState.getScore() \
                + parameters["activeGhost"]*(1/closestActiveGhostDistance) \
                + parameters["scaredGhost"]*(1/closestScaredGhostDistance) \
                + parameters["closestFood"]*(1/closestFood) \
                + parameters["closestPellet"]*(1/closestPellet)

    return Finalscore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
