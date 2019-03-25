import Params
import numpy as np
from collections import deque
import random
from datetime import datetime
import os
import network as network

"""
The code of this file contains all the classes and functions necessary 
for the execution of the test phase of the environment.
"""

"""
This class creates the object "observation" from a "frame" received 
from the environment, the score and the number of lives.
"""
class Observation:
    def __init__(self, image, info):
        self.image = image
        infos = info.split("#")
        self.score = int(infos[1])
        self.lifes = int(infos[2])
        self.actions = int(infos[3])

    def print(self):
        print(self.image.shape)
        print("score: " + str(self.score) + "\nNumber of lifes: " + str(self.lifes) + "\nNumber of actions: " + str(self.actions))


"""
This class has all the functions and methods necessary to carry 
out the test phase.
"""
class Testing:

    """
      The constructor of the class contains and initializes the training parameters,
      the control variables, the "replay memory", an instance of the network class
      and the log file.
    """
    def __init__(self):

        np.random.seed(1)

        self.params = Params.Hyperparams()
        self.reswidth = int(self.params.reswidth)
        self.reshight = int(self.params.reshight)
        self.minibatchsize = int(self.params.minibatchsize)
        self.agenthistorylength = int(self.params.agenthistorylength)
        self.discountfactor = float(self.params.discountfactor)
        self.actionrepeat = int(self.params.actionrepeat)
        self.numminframestate = int(self.params.numminframestate)
        self.learningrate = float(self.params.learningrate)
        self.maxnumNoop = int(self.params.maxnumNoop)
        self.numberactions = int(self.params.numberactions)
        self.actiondonothing = int(self.params.actiondonothing)
        self.pathjobdirectory = str(self.params.pathjobdirectory)
        self.pathtestdirectory = str(self.params.pathtestdirectory)
        self.numnetstest = int(self.params.numnetstest)
        self.numgamesbynet = int(self.params.numgamesbynet)
        self.firstnettotest = int(self.params.firstnettotest)
        self.explorationratetest = float(self.params.explorationratetest)

        self.action = self.actiondonothing
        self.endconfirmed = False

        self.memoryreplayframes = deque(maxlen=self.agenthistorylength)
        self.memoryreplayactions = deque(maxlen=self.agenthistorylength)
        self.memoryreplayrewards = deque(maxlen=self.agenthistorylength)
        self.memoryreplayterminal = deque(maxlen=self.agenthistorylength)

        self.actionrepeattemp = self.actionrepeat
        self.numminframestatetemp = self.numminframestate
        self.numnetstestcount = 1
        self.numgamesbynetcount = 1
        self.scoreacumulator = 0

        self.maxnumNooptemp = random.randint(0, self.maxnumNoop)
        self.framecounter = -1
        self.explorationrate = self.explorationratetest
        self.previousobservation = None
        self.terminal = None
        self.final = None
        self.sequence = None
        self.loss = None

        self.net = network.DQN(self.numberactions, (self.reshight, self.reswidth, self.agenthistorylength), self.learningrate, self.minibatchsize, self.discountfactor)

        self.directorynets = self.pathtestdirectory + "/" + "outputnets"
        self.directoryString = self.pathtestdirectory + "/" + "test_" + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        os.makedirs(self.directoryString)
        self.file = open(self.directoryString + "/" + "loggames.txt", "w")
        self.file2 = open(self.directoryString + "/" + "logmeannets.txt", "w")

        self.net.loadmodel(self.directorynets + "/" + str(self.firstnettotest) + ".h5")

    """
        The functions step and processobservation are the fundamental part of the system. 
        They implement the "deep Q-learning" algorithm. It is responsible for constructing
        the frame sequences, filling the "replay memory", testing the neural network and 
        recording the log information.
    """
    def step(self, observation):

        if self.processobservation(observation) == 0:
            return self.action

        return self.action


    def processobservation(self,observation):

        self.actionrepeattemp -= 1
        if self.actionrepeattemp == 0:
            self.actionrepeattemp = self.actionrepeat

            if self.maxnumNooptemp > 0:
                self.maxnumNooptemp -= 1
                #print("NOOP")
                return 0

            if self.previousobservation is not None:
                self.isterminal(observation)

            if self.numminframestatetemp == 0:
                self.framecounter += 1
                if not self.final:
                    self.calcnewaction()
                    self.addelementmemoryreplay(observation)
                    if len(self.memoryreplayframes) >= self.agenthistorylength:
                        self.memoryreplayrewards[1] = self.calcreward(observation)
                        self.memoryreplayterminal[1] = self.terminal
                else:
                    self.addelementmemoryreplay(observation)
                    self.memoryreplayrewards[1] = self.calcreward(observation)
                    self.memoryreplayterminal[1] = self.terminal
                    self.numminframestatetemp = self.numminframestate
                    self.action = self.actiondonothing
                    self.maxnumNooptemp = random.randint(0, self.maxnumNoop)
                    print("Frame:" + str(self.framecounter - 1) + " Score:" + str(self.previousobservation.score) + " Explorationrate:" + str(self.explorationrate))
                    self.scoreacumulator += self.previousobservation.score
                    self.file.write(str(self.previousobservation.score) + "\n")
                    self.file.flush()

                    self.numgamesbynetcount += 1
                    if self.numgamesbynetcount > self.numgamesbynet:
                        self.numnetstestcount += 1
                        if self.numnetstestcount > self.numnetstest:
                            self.file2.write(str(self.scoreacumulator / self.numgamesbynet) + "\n")
                            self.file2.flush()
                            self.endconfirmed = True
                        else:
                            self.numgamesbynetcount = 1
                            self.net.loadmodel(self.directorynets + "/" + str(self.numnetstestcount) + ".h5")
                            self.file2.write(str(self.scoreacumulator / self.numgamesbynet) + "\n")
                            self.file2.flush()
                            self.scoreacumulator = 0

            else:
                self.addelementmemoryreplay(observation)
                self.numminframestatetemp -= 1

            if not self.final:
                self.previousobservation = observation
            else:
                self.previousobservation = None
                self.final = False
        else:
            return 0

    """
      This function returns the action. Calculated by the network or random.
    """
    def calcnewaction(self):

        if np.random.uniform() > self.explorationratetest:
            self.getstateimage()
            self.action = self.net.get_action_test(self.sequence)
        else:
            self.action = random.randint(0, self.numberactions - 1)

    """This function returns the sequence of frames that is the current state."""
    def getstateimage(self):
        self.sequence = np.zeros((self.reshight, self.reswidth, self.agenthistorylength), dtype=np.uint8)
        for i in range((self.agenthistorylength))[::-1]:
            #figure1 = plt.figure(1)
            #plt.imshow(self.memoryreplayframes[i], cmap='gray', interpolation='nearest')
            #plt.show()
            self.sequence[:, :, i] = self.memoryreplayframes[i]

    """This method adds an "observation" to the "replay memory"."""
    def addelementmemoryreplay(self, observation):
        self.memoryreplayframes.appendleft(observation.image)
        self.memoryreplayactions.appendleft(self.action)
        self.memoryreplayrewards.appendleft(-2.0)
        self.memoryreplayterminal.appendleft(True)

    """This function returns the reward obtained between a current observation and the previous one."""
    def calcreward(self,observation):
        if observation.lifes < self.previousobservation.lifes or observation.lifes > self.previousobservation.lifes:
            return 0.0
        if observation.score > self.previousobservation.score:
            return 1.0
        if observation.score == self.previousobservation.score:
            return 0.0

    """This method verifies whether the new received observation belongs to a terminal state."""
    def isterminal(self,observation):
        if observation.lifes > self.previousobservation.lifes or observation.lifes < self.previousobservation.lifes:
            if observation.lifes > self.previousobservation.lifes:
                self.final = True
            else:
                self.final = False
            self.terminal = True
        else:
            self.terminal = False

    """This function returns the number represented by the "do nothing" action."""
    def getactiondonothing(self):
        return self.actiondonothing




