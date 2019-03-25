import asyncore
import socket
import threading
import io
import numpy as np
from PIL import Image
import Trainer as tr
import Tester as test
import subprocess
from time import sleep
from skimage.color import rgb2gray
from skimage.transform import resize

"""
The code in this file is used to execute the emulation, in training and test. 
First, the type of emulation object is created, then the server that listens 
to the virtual environment is executed and finally the virtual environment 
is executed.
"""


"""This function resizes and converts an image to grayscale."""
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


"""This class is the server. Receive the information of the virtual 
environment and send actions."""
class EchoHandler(asyncore.dispatcher_with_send):

    inforec = False
    info = None


    def handle_read(self):
        data = self.recv(8192)
        if data:
            try:
                self.info = data.decode('utf-8')
                #print(info)
                inforec = True
            except:
                inforec = False

            if inforec:
                self.send(str.encode("5"))
            else:
                try:
                    global emulation
                    global endconfirmed
                    global lastframe
                    img = Image.open(io.BytesIO(data)).convert('L')
                    lastframe = img
                    im2arr = np.array(img, dtype=np.uint8)
                    im2arr = pre_processing(im2arr)
                    if emulation.endconfirmed:
                        endconfirmed = True
                    action = emulation.step(tr.Observation(im2arr, self.info))
                    answer = str(action)
                    self.send(str.encode(answer))
                except Exception as e:
                    #print("Error read frame.")
                    #print(e)
                    img = lastframe
                    im2arr = np.array(img, dtype=np.uint8)
                    im2arr = pre_processing(im2arr)
                    if emulation.endconfirmed:
                        endconfirmed = True
                    action = emulation.step(tr.Observation(im2arr, self.info))
                    answer = str(action)
                    self.send(str.encode(answer))


"""This class maintains the connection with the virtual environment."""
class EchoServer(asyncore.dispatcher):

    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)

    def handle_accept(self):
        pair = self.accept()
        if pair is not None:
            sock, addr = pair
            #print ('Incoming connection from %s' % repr(addr))
            handler = EchoHandler(sock)


"""Main thread of execution."""

def startserver():
    threading.Thread(target=asyncore.loop, name="Server loop").start()

global emulation
global endconfirmed
global lastframe

#emulation = tr.Training()
emulation = test.Testing()
conex = EchoServer('localhost', 50002)
lastframe = np.zeros((84,84,4))

startserver()
game = subprocess.Popen([r"asteroids.exe"])
endconfirmed = False
while not endconfirmed:
    sleep(0.5)
game.terminate()
conex.close()



