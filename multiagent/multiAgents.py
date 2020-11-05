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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        # below is my code
        #about food
        foodPos = newFood.asList() #list of food pos
        remainFood = [i for i in foodPos if newFood[i[0]][i[1]]==True] #remaining food pos
        if len(remainFood)!= 0:
            fooddist = [manhattanDistance(food, newPos) for food in remainFood] #dist to remaining food
            nearFoodDist = min(fooddist)
        else:
            nearFoodDist = 0
        numFood = successorGameState.getNumFood()

        #about the large dots
        capsules = currentGameState.getCapsules()

        numActions = len(successorGameState.getLegalActions())

        ghostposlist = [ghostState.getPosition() for ghostState in newGhostStates]
        nonscared = 0
        scared = 0
        for i in range(len(ghostposlist)):
            if newGhostStates[i].scaredTimer != 0:
                scared += manhattanDistance(ghostposlist[i], newPos)
            elif manhattanDistance(ghostposlist[i], newPos)==0:
                nonscared -= 100
            else:
                nonscared += manhattanDistance(ghostposlist[i], newPos)

        return successorGameState.getScore()-3*numFood+2/(nearFoodDist+0.1)-1.5/(nonscared+0.1)+1.5/(scared+0.1) + sum(newScaredTimes)


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

        numGhost = gameState.getNumAgents() - 1

        def value(gameState, cur_depth, turn):
            if cur_depth > self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if turn == 0:
                return maxvalue(gameState, cur_depth, turn)

            if turn != 0:
                return minvalue(gameState, cur_depth, turn)

        def maxvalue(gameState, cur_depth, turn):
            v = float('-inf')
            pacmanActions = gameState.getLegalActions(0)
            bestAct = pacmanActions[0]
            #successors = [gameState.generateSuccessor(0, action) for action in pacmanActions]
            for action in pacmanActions:
                suc = gameState.generateSuccessor(0, action)
                temp = value(suc, cur_depth, turn+1)
                if temp > v:
                    v = temp
                    bestAct = action

            if cur_depth > 1:
                return v
            return bestAct

        def minvalue(gameState, cur_depth, turn):
            v = float('inf')
            ghostActions = gameState.getLegalActions(turn)
            #bestAct = ghostActions[0]

            if turn < numGhost:
                for action in ghostActions:
                    suc = gameState.generateSuccessor(turn, action)
                    if value(suc, cur_depth, turn+1) < v:
                        v = value(suc, cur_depth, turn+1)
            elif turn == numGhost:
                for action in ghostActions:
                    suc = gameState.generateSuccessor(turn, action)
                    if value(suc, cur_depth+1, 0) < v:
                        v = value(suc, cur_depth+1, 0)

            return v

        return value(gameState, 1, 0)

    #     numAgents = gameState.getNumAgents()
    #     numghost = numAgents - 1 #number of ghsots
    #
    #     if gameState.isWin() or gameState.isLose():
    #         return self.evaluationFunction(gameState)
    #
    #     return self.maxValueAct(gameState, 1)
    #
    #
    # def maxValueAct(self, gameState, cur_depth):
    #     if gameState.isWin() or gameState.isLose():
    #         return self.evaluationFunction(gameState)
    #
    #     v = float('-inf')
    #     pacmanActions = gameState.getLegalActions(0)  #legal action of pacman
    #     bestAct = pacmanActions[0]
    #
    #     for action in pacmanActions:
    #         suc = gameState.generateSuccessor(0, action)
    #         next = self.minValueAct(suc, cur_depth, 1)
    #         if v < next:
    #             v = next
    #             bestAct = action
    #
    #     if cur_depth >= self.depth:
    #         return v
    #
    #     return bestAct
    #
    # def minValueAct(self, gameState, cur_depth, cur_ghost):
    #     if gameState.isWin() or gameState.isLose():
    #         return self.evaluationFunction(gameState)
    #
    #     v = float('inf')
    #     ghostActions = gameState.getLegalActions(cur_ghost)
    #     successors = [gameState.generateSuccessor(cur_ghost, action) for action in ghostActions]
    #
    #     if cur_ghost < (gameState.getNumAgents()-1):
    #         for suc in successors:
    #             v = min(v, self.minValueAct(suc, cur_depth, cur_ghost+1))
    #     elif cur_ghost == (gameState.getNumAgents()-1):
    #         if cur_depth < self.depth:
    #             for suc in successors:
    #                 v = min(v, self.maxValueAct(suc, cur_depth+1))
    #         else:
    #             for suc in successors:
    #                 v = min(v, self.evaluationFunction(suc))
    #     else:
    #         return 'wrong num of ghost'
    #
    #     return v




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numGhost = gameState.getNumAgents() - 1

        def value(gameState, cur_depth, turn, alpha, beta):
            if cur_depth > self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if turn == 0:
                return maxvalue(gameState, cur_depth, turn, alpha, beta)

            if turn != 0:
                return minvalue(gameState, cur_depth, turn, alpha, beta)

        def maxvalue(gameState, cur_depth, turn, alpha, beta):
            v = float('-inf')
            pacmanActions = gameState.getLegalActions(0)
            bestAct = pacmanActions[0]
            for action in pacmanActions:
                suc = gameState.generateSuccessor(0, action)
                temp = value(suc, cur_depth, turn+1, alpha, beta)
                if temp > v:
                    v = temp
                    bestAct = action
                if v > beta:
                    return v
                alpha = max(alpha, v)

            if cur_depth > 1:
                return v
            return bestAct

        def minvalue(gameState, cur_depth, turn, alpha, beta):
            v = float('inf')
            ghostActions = gameState.getLegalActions(turn)
            #bestAct = ghostActions[0]

            if turn < numGhost:
                for action in ghostActions:
                    suc = gameState.generateSuccessor(turn, action)
                    if value(suc, cur_depth, turn+1, alpha, beta) < v:
                        v = value(suc, cur_depth, turn+1, alpha, beta)
                    if v < alpha:
                        return v
                    beta = min(beta, v)

            elif turn == numGhost:
                for action in ghostActions:
                    suc = gameState.generateSuccessor(turn, action)
                    if value(suc, cur_depth+1, 0, alpha, beta) < v:
                        v = value(suc, cur_depth+1, 0, alpha, beta)
                    if v < alpha:
                        return v
                    beta = min(beta, v)

            return v

        return value(gameState, 1, 0, float('-inf'), float('inf'))

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
        numGhost = gameState.getNumAgents()-1
        def value(gameState, cur_depth, turn):
            if cur_depth > self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if turn == 0:
                return maxvalue(gameState, cur_depth, turn)

            if turn != 0:
                return expectimax(gameState, cur_depth, turn)

        def maxvalue(gameState, cur_depth, turn):
            v = float('-inf')
            pacmanActions = gameState.getLegalActions(0)
            bestAct = pacmanActions[0]
            for action in pacmanActions:
                suc = gameState.generateSuccessor(0, action)
                temp = value(suc, cur_depth, turn+1)
                if temp > v:
                    v = temp
                    bestAct = action

            if cur_depth > 1:
                return v
            return bestAct

        def expectimax(gameState, cur_depth, turn):
            v = 0
            ghostActions = gameState.getLegalActions(turn)
            #bestAct = ghostActions[0]
            probdist = 1/len(ghostActions)
            if turn < numGhost:
                for action in ghostActions:
                    suc = gameState.generateSuccessor(turn, action)
                    v += probdist * value(suc, cur_depth, turn+1)
            elif turn == numGhost:
                for action in ghostActions:
                    suc = gameState.generateSuccessor(turn, action)
                    v += probdist * value(suc, cur_depth+1, 0)

            return v

        return value(gameState, 1, 0)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # below is my code
    #about food
    foodPos = newFood.asList() #list of food pos
    remainFood = [i for i in foodPos if newFood[i[0]][i[1]]==True] #remaining food pos
    if len(remainFood)!= 0:
        fooddist = [manhattanDistance(food, newPos) for food in remainFood] #dist to remaining food
        nearFoodDist = min(fooddist)
    else:
        nearFoodDist = 0
    numFood = currentGameState.getNumFood()

    #about the large dots
    capsules = currentGameState.getCapsules()

    numActions = len(currentGameState.getLegalActions())

    ghostposlist = [ghostState.getPosition() for ghostState in newGhostStates]
    nonscared = 0
    scared = 0
    for i in range(len(ghostposlist)):
        if newGhostStates[i].scaredTimer != 0:
            scared += manhattanDistance(ghostposlist[i], newPos)
        elif manhattanDistance(ghostposlist[i], newPos)==0:
            nonscared -= 100
        else:
            nonscared += manhattanDistance(ghostposlist[i], newPos)

    return currentGameState.getScore()-3*numFood+2/(nearFoodDist+0.1)-1.5/(nonscared+0.1)+1.5/(scared+0.1) + 2*sum(newScaredTimes)

# Abbreviation
better = betterEvaluationFunction
