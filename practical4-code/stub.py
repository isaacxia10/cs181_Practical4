# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

monkey.top.max = 600
monkey.top.min = 50
monkey.vel.max 
monkey.vel.min
tree.dist.max
tree.dist.min
tree.top.max
tree.top.min
tree.bot.max
tree.bot.min

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.total_reward = None
        self.gravity = None



    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.total_reward = None
        self.gravity = None


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.05
        new_state  = state
        print x

        # Infer gravity
        if self.gravity is None:
            if self.last_state:
                grav_est = new_state['monkey']['vel'] - self.last_state['monkey']['vel']
                if grav_est < 0:
                    self.gravity = -grav_est



        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=True,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        print 'hi'
        while swing.game_loop():
            # print learner.last_state
            # print learner.last_reward
            # print learner.last_action
            # print swing.gravity, learner.gravity
            # print (learner.last_state['monkey']['top'] - learner.last_state['monkey']['bot']) != 56
            # print learner.last_state['tree']['dist']
            print (learner.last_state['tree']['top'] - learner.last_state['tree']['bot'])
            pass
        
        # print "out %i: %i"%(ii,learner.last_reward)
        # print swing.score
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 100)

	# Save history. 
	np.save('hist',np.array(hist))


