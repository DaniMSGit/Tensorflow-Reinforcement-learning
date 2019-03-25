import Params
import numpy as np
from collections import deque
import random
from datetime import datetime
import os
import shutil
import network as network


"""
The code of this file contains all the classes and functions necessary
for the execution of the training phase of the environment.
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
This class has all the functions and methods necessary to carry out
the training. It also contains the "replay memory" and an instance 
of the "network" class.
"""
class Training:

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
        self.replaymemorysize = int(self.params.replaymemorysize)
        self.agenthistorylength = int(self.params.agenthistorylength)
        self.targetnetworksupdatefrequency = int(self.params.targetnetworksupdatefrequency)
        self.discountfactor = float(self.params.discountfactor)
        self.actionrepeat = int(self.params.actionrepeat)
        self.numminframestate = int(self.params.numminframestate)
        self.updatefrequency = int(self.params.updatefrequency)
        self.learningrate = float(self.params.learningrate)
        self.initialexploration = float(self.params.initialexploration)
        self.finalexploration = float(self.params.finalexploration)
        self.finalexplorationframe = int(self.params.finalexplorationframe)
        self.replaystartsize = int(self.params.replaystartsize)
        self.maxnumNoop = int(self.params.maxnumNoop)
        self.numframesavenet = int(self.params.numframesavenet)
        self.numberactions = int(self.params.numberactions)
        self.actiondonothing = int(self.params.actiondonothing)
        self.pathjobdirectory = str(self.params.pathjobdirectory)
        self.moderun = "normal"

        self.action = self.actiondonothing
        self.endconfirmed = False

        self.memoryreplayframes = deque(maxlen=self.replaymemorysize)
        self.memoryreplayactions = deque(maxlen=self.replaymemorysize)
        self.memoryreplayrewards = deque(maxlen=self.replaymemorysize)
        self.memoryreplayterminal = deque(maxlen=self.replaymemorysize)

        self.actionrepeattemp = self.actionrepeat
        self.numminframestatetemp = self.numminframestate
        self.replaystartsizetemp = self.replaystartsize
        self.maxnumNooptemp = random.randint(0, self.maxnumNoop)
        self.updatefrequencytemp = 0
        self.targetnetworksupdatefrequencytemp = 0
        self.framecounter = -1
        self.explorationrate = 0
        self.countnetssaved = 0
        self.previousobservation = None
        self.terminal = None
        self.final = None
        self.sequence = None
        self.loss = None


        self.net = network.DQN(self.numberactions, (self.reshight, self.reswidth, self.agenthistorylength), self.learningrate, self.minibatchsize, self.discountfactor)

        if not os.path.exists(self.pathjobdirectory):
            os.makedirs(self.pathjobdirectory)

        self.directoryString = self.pathjobdirectory + "/" + datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        os.makedirs(self.directoryString)
        shutil.copy("./network.py", self.directoryString + "/network.py")
        shutil.copy("./hyperparameters.json", self.directoryString + "/hyperparameters.json")
        self.outputnetstrain = self.directoryString + "/" + "outputnets"
        os.makedirs(self.outputnetstrain)
        self.outputlogstrain = self.directoryString + "/" + "outputlogs"
        os.makedirs(self.outputlogstrain)
        self.file = open(self.outputlogstrain + "/" + "log.txt", "w")

    """
    The functions step and processobservation are the fundamental part of the system. 
    They implement the "deep Q-learning" algorithm. It is responsible for constructing
    the frame sequences, filling the "replay memory", training the neural network and 
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
            if self.numminframestatetemp != 0:
                self.numminframestatetemp -= 1
            if self.numminframestatetemp == 0:
                self.framecounter += 1
                if self.framecounter % self.numframesavenet == 0 or self.framecounter == 0:
                    self.countnetssaved += 1
                    self.net.savenet(self.outputnetstrain + "/" + str(self.countnetssaved) + ".h5")
                if not self.final:
                    self.calcexplorationrate()
                    self.calcnewaction()
                    self.addelementmemoryreplay(observation)
                    if len(self.memoryreplayframes) >= self.agenthistorylength:
                        self.memoryreplayrewards[1] = self.calcreward(observation)
                        #print("REWARD:" + str(self.memoryreplayrewards[1]))
                        self.memoryreplayterminal[1] = self.terminal
                    if self.replaystartsizetemp == 0:
                        self.istraining()
                    else:
                        self.replaystartsizetemp -= 1
                else:
                    self.addelementmemoryreplay(observation)
                    self.memoryreplayrewards[1] = self.calcreward(observation)
                    self.memoryreplayterminal[1] = self.terminal
                    self.numminframestatetemp = self.numminframestate
                    self.action = self.actiondonothing
                    self.maxnumNooptemp = random.randint(0, self.maxnumNoop)
                    self.updatefrequencytemp = 0
                    print("Frame:" + str(self.framecounter - 1) + " Score:" + str(self.previousobservation.score) + " Explorationrate:" + str(self.explorationrate))
                    #print("Epsilon:" + str(self.explorationrate))
                    self.file.write(str(self.previousobservation.score) + "\n")
                    self.file.flush()
                    #print("Num no OP:" + str(self.maxnumNooptemp))
            else:
                self.addelementmemoryreplay(observation)
            if not self.final:
                self.previousobservation = observation
            else:
                self.previousobservation = None
                self.final = False
        else:
            #print("FRAMESKIPPING" + "ACTION:" + str(self.action))
            return 0
    """
    This function returns the action. Calculated by the network or random.
    """
    def calcnewaction(self):

        if self.replaystartsizetemp > 0:
            self.action = random.randint(0, self.numberactions - 1)
        elif np.random.uniform() > self.explorationrate:
            self.getstateimage()
            self.action = self.net.get_action(self.sequence)
        else:
            self.action = random.randint(0, self.numberactions - 1)

    """
    This function initializes the training of the network when necessary.
    """
    def istraining(self):
        self.updatefrequencytemp += 1
        if self.updatefrequency == self.updatefrequencytemp:
            #print("TRAINING")
            self.updatefrequencytemp = 0
            self.targetnetworksupdatefrequencytemp += 1
            #print("UPDATENET:" + str(self.targetnetworksupdatefrequencytemp))
            if self.targetnetworksupdatefrequency == self.targetnetworksupdatefrequencytemp:
                #print("UPDATETARGETNET")
                self.net.updatenet()
                self.targetnetworksupdatefrequencytemp = 0
            self.getbatch()
            self.loss = self.net.train_memory_batch(self.batchS, self.batchSplus, self.batchactions, self.batchrewards, self.batchterminal)
            #self.loss = self.networks.getLoss()
            #print("LOSS:" + str(self.loss))

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

    """This method builds a batch of information that will be used to train the neural network."""
    def getbatch(self):
        self.batchS = np.zeros([self.minibatchsize, self.reshight, self.reswidth, self.agenthistorylength])
        self.batchSplus = np.zeros([self.minibatchsize, self.reshight, self.reswidth, self.agenthistorylength])
        self.batchactions = []
        self.batchrewards = []
        self.batchterminal = []
        index = random.sample(range(1, len(self.memoryreplayframes)-(self.agenthistorylength-1)), self.minibatchsize)

        for i in range(len(index)):
            ind = index[i]
            while self.memoryreplayrewards[ind] == -2:
                ind -= 1
                while ind < 0:
                    ind = random.randint(1, len(self.memoryreplayframes)-(self.agenthistorylength-1))
            self.batchS[i,:, :, :] = self.getsequence(ind)
            self.batchSplus[i, :, :, :] = self.getsequence(ind-1)
            self.batchactions.append(self.memoryreplayactions[ind])
            self.batchrewards.append(self.memoryreplayrewards[ind])
            self.batchterminal.append(self.memoryreplayterminal[ind])

        #self.batchS = np.float32(self.batchS / 255.)
        #self.batchSplus = np.float32(self.batchSplus / 255.)

        """
        #if self.framecounter > 250:
            #if not os.path.exists("b"):
            #    os.makedirs("b")
        for k in range(0, len(self.batchS)):
            if (self.batchrewards[k] > 0.0):
                name =  "b/Sequence" + str(k+1) + ".png"
                figure1 = plt.figure(k)
                figure1.suptitle("ACTION: " + str(self.batchactions[k]) + " REWARD: " + str(self.batchrewards[k]) + "TERMINAL: " + str(self.batchterminal[k]))
                gridspec.GridSpec(self.agenthistorylength, self.agenthistorylength)
                for i in range(self.agenthistorylength-1, -1, -1):
                    plt.subplot2grid((self.agenthistorylength, 2), (i, 0))
                    plt.imshow(self.batchS[k, :, :, i], cmap='gray', interpolation='nearest')
                    plt.axis('off')
                    plt.subplot2grid((self.agenthistorylength, 2), (i, 1))
                    plt.imshow(self.batchSplus[k, :, :, i], cmap='gray', interpolation='nearest')
                    plt.axis('off')
                #figure1.savefig(name)
                plt.show()
        """

    """This function returns a sequence of "frames" in a given position of the "replay memory"."""
    def getsequence(self,ind):
        sq = np.zeros((self.reshight, self.reswidth, self.agenthistorylength), dtype=np.uint8)
        j = self.agenthistorylength - 1
        for i in range(ind, ind+self.agenthistorylength)[::-1]:
            sq[:, :, j] = self.memoryreplayframes[i]
            j -= 1
        return sq

    """This method recalculates the value of the exploration ratio according to the number of execution frames."""
    def calcexplorationrate(self):

        self.explorationrate = self.initialexploration - (((self.initialexploration-self.finalexploration) / self.finalexplorationframe) * (self.framecounter))
        if self.explorationrate <= self.finalexploration:
            self.explorationrate = self.finalexploration
        #print("Exploration Rate:" + str(self.explorationrate))


    """This function returns the reward obtained between a current observation and the previous one."""
    def calcreward(self,observation):
        if observation.lifes < self.previousobservation.lifes or observation.lifes > self.previousobservation.lifes:
            return -1.0
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




