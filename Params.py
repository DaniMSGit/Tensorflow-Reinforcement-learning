import json

"""
The code of this file contains the class "Params". It contains the training 
and test parameters used by the other classes. The parameters are read from
the file "hyperparameters.json".
"""

class Hyperparams:
    def __init__(self):
        self.data = json.load(open('hyperparameters.json'))
        self.reswidth = self.data["reswidth"]
        self.reshight = self.data["reshight"]
        self.minibatchsize = self.data["minibatchsize"]
        self.replaymemorysize = self.data["replaymemorysize"]
        self.agenthistorylength = self.data["agenthistorylength"]
        self.targetnetworksupdatefrequency = self.data["targetnetworksupdatefrequency"]
        self.discountfactor = self.data["discountfactor"]
        self.actionrepeat = self.data["actionrepeat"]
        self.numminframestate = self.data["numminframestate"]
        self.updatefrequency = self.data["updatefrequency"]
        self.learningrate = self.data["learningrate"]
        self.initialexploration = self.data["initialexploration"]
        self.finalexploration = self.data["finalexploration"]
        self.finalexplorationframe = self.data["finalexplorationframe"]
        self.replaystartsize = self.data["replaystartsize"]
        self.maxnumNoop = self.data["maxnumNoop"]
        self.numframesavenet = self.data["numframesavenet"]
        self.numberactions = self.data["numberactions"]
        self.actiondonothing = self.data["actiondonothing"]
        self.pathjobdirectory = self.data["pathjobdirectory"]
        self.pathtestdirectory = self.data["pathtestdirectory"]
        self.numnetstest = self.data["numnetstest"]
        self.numgamesbynet = self.data["numgamesbynet"]
        self.firstnettotest = self.data["firstnettotest"]
        self.explorationratetest = self.data["explorationratetest"]




