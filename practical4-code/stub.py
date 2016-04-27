# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey

''' Ranges for states '''
# d_top is monkey_top - tree_top
d_top_max = 500
d_top_min = -500
monkey_vel_max = 50
monkey_vel_min = -50
tree_dist_max = 500
tree_dist_min = 0


''' Bin sizes '''
# 1000/20 = 50
d_top_nbin = 10
# 100/10 = 10
monkey_vel_nbin = 3
# First few distances are 485, 460, 435...
tree_dist_nbin = 10


""" Computed number of bins (don't touch!)"""
d_top_szbin = (d_top_max - d_top_min)/d_top_nbin
monkey_vel_szbin = (monkey_vel_max - monkey_vel_min)/monkey_vel_nbin
tree_dist_szbin = (tree_dist_max - tree_dist_min)/tree_dist_nbin


class Learner(object):

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.curr_state  = None
		self.curr_action = None
		self.total_reward = None
		# gravity of 1 is state 0, gravity of 4 is state 1s
		self.gravity = None
		self.epoch = 0

		# Discount rate
		self.gamma = 0.4

		''' State/Actions consist of S = (d_top, monkey_vel,
			tree_dist, gravity) and A = {0,1}
			0 = no jump, 1 = jump
		'''
		state_dims = (d_top_nbin, monkey_vel_nbin, tree_dist_nbin, 2, 2)
		self.Q = np.zeros(state_dims)
		self.state_counts = np.zeros(state_dims)


	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.curr_state  = None
		self.curr_action = None
		self.total_reward = None
		self.gravity = None

		# Don't reset epoch!
		self.epoch += 1


	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''

		# # You might do some learning here based on the current state and the last state.

		# # You'll need to select and action and return it.
		# # Return 0 to swing and 1 to jump.

		# new_action = npr.rand() < 0.05
		# new_state  = state

		# Infer gravity
		if self.gravity is None:
			if self.last_state:
				grav_est = state['monkey']['vel'] - self.last_state['monkey']['vel']
				if grav_est < 0:
					if grav_est == -1:
						self.gravity = 0
					else:
						self.gravity = 1


		# Epsilon-greedy learning

		if npr.random() > self.epsilon():
			self.last_action = np.argmax(self.Q[self.state_hash(state)])
		else:
			# Random choice
			if npr.random() > 0.1:
				self.last_action = 0
			else:
				self.last_action = 1


		self.last_state = self.curr_state
		self.curr_state = state

		# print self.last_action


		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''

		# If we have s', s, and a, we can start learning!
		if (self.curr_state is not None) and (self.last_state is not None) and (self.last_action is not None) and (self.gravity is not None):
				s1 = self.state_hash(self.curr_state)
				s = self.state_hash(self.last_state)
				a = self.last_action
				alpha = self.alpha(self.last_state, self.last_action)


				newQ = self.Q[s + (a,)] + alpha*((reward + self.gamma*np.max(self.Q[s1])) - self.Q[s + (a,)])
				self.Q[s + (a,)] = newQ
				# print newQ

		# print reward
		self.last_reward = reward


	def alpha(self, state, action):
		''' Returns adaptive alpha for state; alpha(state) = 1/count(state)(action),
		and updates count.'''

		self.state_counts[self.state_hash(state)][action] += 1
		alpha = 1.0/self.state_counts[self.state_hash(state)][action]
		return alpha

	def epsilon(self):
		''' Returns eps for epsilon-greedy policy under eps(t) = 1/(t+1).'''
		return 1.0/(self.epoch + 1.0)

	def state_hash(self, state):
		''' Returns a hash of the given state for use in Q:
		(d_top, monkey_vel, tree_dist, gravity)'''

		d_top_value = state['monkey']['top'] - state['tree']['top']
		monkey_vel_value = state['monkey']['vel']
		tree_dist_value  = state['tree']['dist']

		d_top_hash = np.floor((d_top_value - d_top_min)/d_top_szbin)
		monkey_vel_hash = np.floor((monkey_vel_value - monkey_vel_min)/monkey_vel_szbin)
		tree_dist_hash = np.floor((tree_dist_value - tree_dist_min)/tree_dist_szbin)

		return (d_top_hash, monkey_vel_hash, tree_dist_hash, self.gravity)





def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
		
	high = 0
	avg = 0.0
	
	for ii in range(iters):

		# # don't need because learner.reset() increments it
		# learner.epoch = ii


		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			# print learner.last_state
			# print learner.last_reward
			# print learner.last_action
			# print swing.gravity, learner.gravity
			# print (learner.last_state['monkey']['top'] - learner.last_state['monkey']['bot']) != 56
			# print learner.last_state['tree']['dist']
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


