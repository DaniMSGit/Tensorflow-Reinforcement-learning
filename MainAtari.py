import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import Trainer as tr
import Tester as ts
import time

"""
The code of this file executes the emulation in training and test 
with the virtual environment "Breakout" of Atari. No server is 
necessary, "Breakout" is a Python object.
"""

"""This function resizes and converts an image to grayscale."""
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


"""Main thread of execution."""
env = gym.make('BreakoutDeterministic-v4')
#emulation = tr.Training()
emulation = ts.Testing()
state = None
score = 0
numaction = 4
lives = 5
done = False

for i in range(10000000):
    if i == 0 or done:
        state = env.reset()
        state = pre_processing(state)
        score = 0
        lives = 5
        msg = str("#" + str(int(score)) + "#" + str(int(lives)) + "#" + str(int(numaction)))
        action = emulation.step(tr.Observation(state, msg))
        action = action + 1
    else:
        action = emulation.step(tr.Observation(state, msg))
        action = action + 1

    env.render()
    time.sleep(0.01)
    #print("ACTION EXECUTED:" + str(action))

    next_state, reward, done, info = env.step(action)
    #print("REWARD:" + str(reward))
    next_state = pre_processing(next_state)
    state = next_state
    lives = info['ale.lives']
    score += reward
    msg = str("#" + str(int(score)) + "#" + str(int(lives)) + "#" + str(int(numaction)))




env.close()