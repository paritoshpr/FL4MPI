# This project enables large scale Federated Learning for HPC using MPI
# Copyright (C) 2022 Paritosh Ramanan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os, glob
from sklearn.metrics import accuracy_score
import numpy as np
from ConfigReader import ConfigReader
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Activation
from keras.optimizers import Adam

class Learner:

    def __init__(self,device_id,no_devices):

        self.model=None

        self.data_root_dir = ""
        self.test_dir = ""
        self.val_dir = ""
        self.model_save_dir = ""

        self.batch_size = 0
        self.num_epochs = 0
        self.max_rounds = 0
        self.no_devices = no_devices

        #input feature dimensions
        self.P = 10

        self.initModel(device_id, no_devices)

        self.fineTuneCreateModel()


    def initModel(self,device_id,num_devices):
        cr = ConfigReader()
        self.data_root_dir = cr.data_root_dir
        self.test_dir = cr.test_dir
        self.val_dir = cr.val_dir
        self.model_save_dir = cr.model_save_dir

        self.batch_size = cr.batch_size
        self.num_epochs = cr.num_epochs
        self.max_rounds = cr.max_rounds

        self.num_devices = num_devices


    def fineTuneCreateModel(self):
        x_in = Input((self.P ))
        x = Dense(100, input_dim=self.P )(x_in)
        x = Activation('relu')(x)
        x = Dense(100)(x)
        x = Activation('relu')(x)
        y = Dense(self.P)(x)
        self.model = Model(inputs=x_in, outputs=y)

        # Compile model
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mean_squared_error', optimizer=adam)

    def learn(self,inputs,labels):
        self.model.fit(inputs, labels, epochs=self.num_epochs, batch_size=self.batch_size)
        self.weights = self.model.get_weights()
        self.model_weight_buffer = np.copy(self.model.get_weights())

    def setGlobalWeight(self):
        self.model.set_weights(self.model_weight_buffer)

    def saveModel(self,round):
        model_name = self.model_save_dir +"_"+ str(round) + "_" +str(
            self.batch_size) + "_" + str(self.no_devices) + "_" + str(self.max_rounds) + "_" + str(self.num_epochs)
        #use a framework to save here
        return model_name

    def get_predictions(self,folder,saved_model_dir):

        #do prediction based on data saved in folder and model saved in saved_model_dir and return lables
        labels = ""
        return labels

