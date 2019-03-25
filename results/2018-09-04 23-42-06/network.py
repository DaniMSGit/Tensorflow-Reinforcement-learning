import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import load_model
from keras.models import clone_model

class DQN:

    def __init__(self, _action_size, _input_shape, _learning_rate, _batch_size, _discountfactor):
        self.input_shape = _input_shape
        self.action_size = _action_size
        self.learning_rate = _learning_rate
        self.batch_size = _batch_size
        self.discountfactor = _discountfactor
        self.model = self.model_net()
        self.model_target = clone_model(self.model)
        self.model_target.set_weights(self.model.get_weights())
        self.graph = tf.get_default_graph()

    def huber_loss(self,y, q_value):
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        return loss

    def model_net(self):

        frames_input = layers.Input(self.input_shape, name='frames')
        actions_input = layers.Input((self.action_size,), name='action_mask')

        normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

        conv_1 = layers.convolutional.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(normalized)
        conv_2 = layers.convolutional.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv_1)
        conv_3 = layers.convolutional.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv_2)

        conv_flattened = layers.core.Flatten()(conv_3)

        hidden = layers.Dense(512, activation='relu')(conv_flattened)

        output = layers.Dense(self.action_size)(hidden)

        filtered_output = layers.Multiply(name='QValue')([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.summary()
        optimizer = RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss=self.huber_loss)

        return model

    def get_one_hot(self,targets, nb_classes):
        return np.eye(nb_classes)[np.array(targets).reshape(-1)]

    def train_memory_batch(self, batchS, batchSplus, batchactions, batchrewards, batchterminal):

        target = np.zeros((self.batch_size,))

        actions_mask = np.ones((self.batch_size, self.action_size))

        with self.graph.as_default():
            next_Q_values = self.model.predict([batchSplus, actions_mask])

        for i in range(self.batch_size):
            if batchterminal[i]:
                target[i] = -1
                # target[i] = reward[i]
            else:
                target[i] = batchrewards[i] + self.discountfactor * np.amax(next_Q_values[i])

        action_one_hot = self.get_one_hot(batchactions, self.action_size)
        target_one_hot = action_one_hot * target[:, None]

        with self.graph.as_default():
            h = self.model.fit([batchS, action_one_hot], target_one_hot, epochs=1, batch_size= self.batch_size, verbose=0)
        return h.history['loss'][0]

    def get_action(self, history):
        history = np.reshape([history], (1, 84, 84, 4))
        with self.graph.as_default():
            q_value = self.model_target.predict([history, np.ones(self.action_size).reshape(1, self.action_size)])
        return np.argmax(q_value[0])

    def get_action_test(self, history):
        history = np.reshape([history], (1, 84, 84, 4))
        with self.graph.as_default():
            q_value = self.model.predict([history, np.ones(self.action_size).reshape(1, self.action_size)])
        return np.argmax(q_value[0])

    def updatenet(self):
        self.model_target.set_weights(self.model.get_weights())

    def savenet(self,path):
        self.model.save(path)

    def loadmodel(self,path):
        with self.graph.as_default():
            self.model = load_model(path, custom_objects={'huber_loss': self.huber_loss})