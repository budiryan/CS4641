# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter()  # A Counter is a dict with default 0

    "*** YOUR CODE HERE ***"
    states = mdp.getStates()
    self.good_policy = util.Counter()
    for i in range(iterations):
      # Compute the bellman equation
      temp_dict = self.values.copy()
      for state in states:
        possible_actions = mdp.getPossibleActions(state)
        temp_array = []
        # choose the action that yields the maximum value
        for action in possible_actions:
          transition_states = mdp.getTransitionStatesAndProbs(state, action)
          temp = 0
          # transition_states returns: (nextState, prob)
          # the sigma part
          temp_array.append(self.getQValue(state, action))
          # Now have to choose the largest value
        if temp_array:
          maximum_value = max(temp_array)
          temp_dict[state] = maximum_value
          self.good_policy[state] = possible_actions[temp_array.index(maximum_value)]
        else:
          temp_dict[state] = 0
          self.good_policy[state] = None
      self.values = temp_dict.copy()

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    transition_states = self.mdp.getTransitionStatesAndProbs(state, action)
    temp = 0
    # transition_states returns: (nextState, prob)
    # the sigma part
    for next_state_and_prob in transition_states:
      next_state, prob = next_state_and_prob
      temp = temp + (prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]))
    return temp

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return str(self.good_policy[state])

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
