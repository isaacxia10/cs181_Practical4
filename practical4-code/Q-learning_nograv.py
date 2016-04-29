# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey





''' Bin sizes '''
d_top_szbin = 47
monkey_vel_szbin = 28
tree_dist_szbin = 95

d_top_nbin = 10
monkey_vel_nbin = 3
tree_dist_nbin = 10

class Learner(object):

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.curr_state  = None
        self.curr_action = None
        # gravity of 1 is state 0, gravity of 4 is state 1s
        # self.gravity = None
        self.epoch = 0

        # Discount rate
        self.gamma = 0.4

        ''' State/Actions consist of S = (d_top, monkey_vel,
            tree_dist) and A = {0,1}
            0 = no jump, 1 = jump
        '''
        state_dims = (d_top_nbin, monkey_vel_nbin, tree_dist_nbin, 2)
        self.Q = np.zeros(state_dims)
        self.state_counts = np.zeros(state_dims)


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.curr_state  = None
        self.curr_action = None
        # self.gravity = None

        # Don't reset epoch!
        self.epoch += 1


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # # You might do some learning here based on the current state and the last state.


        # Infer gravity
        # if self.gravity is None:
        #   if self.curr_state:
        #       grav_est = state['monkey']['vel'] - self.curr_state['monkey']['vel']
        #       if grav_est < 0:
        #           if grav_est == -1:
        #               self.gravity = 0
        #           else:
        #               self.gravity = 1


        # Epsilon-greedy learning

        epsilon = 1.0/(self.epoch + 1.0)

        # If no check for gravity, argmax could return > 1 
        if npr.random() > epsilon:
            new_action = np.argmax(self.Q[self.state_hash(state)])
        else:
            # Random choice; jump with lower probability
            if npr.random() > 0.1:
                new_action = 0
            else:
                new_action = 1


        self.last_action = self.curr_action
        self.curr_action = new_action
        self.last_state  = self.curr_state
        self.curr_state = state


        return self.curr_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        # If we have s', s, and a, we can start learning!
        if (self.curr_state is not None) and (self.last_state is not None) and (self.last_action is not None):
            s1 = self.state_hash(self.curr_state)
            s = self.state_hash(self.last_state)
            a = self.last_action

            self.state_counts[s][a] += 1
            alpha = 1.0/self.state_counts[s][a] 

            change = ((reward + self.gamma*np.max(self.Q[s1])) - self.Q[s][a])
            self.Q[s][a] += alpha*change

        self.last_reward = reward


    def state_hash(self, state):
        ''' Returns a hash of the given state for use in Q:
        (d_top, monkey_vel, tree_dist, gravity)'''

        d_top_hash = (70 + (state['monkey']['bot'] - state['tree']['bot']))/d_top_szbin
        monkey_vel_hash = (state['monkey']['vel']+39)/monkey_vel_szbin
        tree_dist_hash = (state['tree']['dist'])/tree_dist_szbin

        return (d_top_hash, monkey_vel_hash, tree_dist_hash)





def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
        
    
    high = 0
    avg = 0
    for ii in range(iters):



        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        new_score = swing.score

        # Save score history.
        if new_score > high:
            high = new_score

        avg = (new_score + ii*avg)/(ii+1.0)
        print "%i\t%i\t%i\t%s:\t%s"%(ii,new_score,high,avg,np.mean(learner.Q))
        hist.append(swing.score)
        # Reset the state of the learner.
        learner.reset()
        
    print learner.Q
    print learner.state_counts
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. 
    run_games(agent, hist, 1000, 1)


    # Save history. 
    print hist
    np.save('hist',np.array(hist))


