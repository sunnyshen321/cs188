# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        allStates = self.mdp.getStates()
        num_i = 0
        while num_i < self.iterations:
            old_values = self.values.copy()
            for s in allStates:
                if self.mdp.isTerminal(s):
                    continue
                allActions = self.mdp.getPossibleActions(s)
                q_values = []
                for action in allActions:
                    q_values.append(self.getQValue(s, action))
                old_values[s] = max(q_values)

            self.values = old_values
            num_i += 1

            # bestact = self.computeActionFromValues(s)
            # if bestact is None:
            #     continue

            # transition = self.mdp.getTransitionStatesAndProbs(s, bestact)
            # v_Value = 0
            # for i in range(len(transition)):
            #     nextState = transition[i][0]
            #     nextStateValue = old_values[nextState]
            #     prob = transition[i][1]
            #     reward = self.mdp.getReward(s, bestact, nextState)
            #     v_Value += prob * (reward + self.discount * nextStateValue)
            # self.values[s] = v_Value


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for i in range(len(transition)):
            nextState = transition[i][0]
            prob = transition[i][1]
            nextStateValue = self.values[nextState]
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob*(reward + self.discount * nextStateValue)

        return qValue



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        legalActions = self.mdp.getPossibleActions(state)
        if len(legalActions) == 0:
            return None
        #normal case
        action_value = {}
        for a in legalActions:
            action_value[a] = self.getQValue(state, a)
        all = list(action_value.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        allStates = self.mdp.getStates()
        num_i = 0
        while num_i < self.iterations:
            #old_values = self.values.copy()
            s = allStates[num_i % len(allStates)]

            if not self.mdp.isTerminal(s):
                allActions = self.mdp.getPossibleActions(s)
                q_values = []
                for action in allActions:
                    q_values.append(self.computeQValueFromValues(s, action))

                #old_values[s] = max(q_values)

                self.values[s] = max(q_values)
            num_i += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        queue = util.PriorityQueue()
        pred = {}

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                q_values = []

                for a in actions:
                    transition = self.mdp.getTransitionStatesAndProbs(s, a)
                    for i in transition:
                        nextState = i[0]
                        if nextState in pred:
                            pred[nextState].add(s)
                        else:
                            pred[nextState] = {s}
                    q_values.append(self.getQValue(s, a))

                best_q = max(q_values)
                diff = abs(best_q - self.values[s])
                queue.update(s, -diff)

        # for s in self.mdp.getStates():
        #     if not self.mdp.isTerminal(s):
        #         allActions = self.mdp.getPossibleActions(s)
        #         q_values = []
        #         for action in allActions:
        #             q_values.append(self.getQValue(s, action))
        #         best_q = max(q_values)
        #         diff = abs(best_q - self.values[s])
        #
        #         queue.update(s, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break

            temp_s = queue.pop()
            if not self.mdp.isTerminal(temp_s):
                allActions = self.mdp.getPossibleActions(temp_s)
                q_values = []
                for action in allActions:
                    q_values.append(self.getQValue(temp_s, action))
                best_q = max(q_values)
                self.values[temp_s] = best_q

            preds = pred[temp_s]
            for p in preds:
                if not self.mdp.isTerminal(p):
                    allActions = self.mdp.getPossibleActions(p)
                    q_values = []
                    for action in allActions:
                        q_values.append(self.getQValue(p, action))
                    best_q = max(q_values)
                    diff = abs(best_q - self.values[p])

                    if diff > self.theta:
                        queue.update(p, -diff)