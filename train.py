import tensorflow
import copy
import numpy as np
import random
import os
import sys
from math import ceil, floor

# dictionary, act to index
get_pos = { 0:(0, 0), 1:(0, 1), 2:(0, 2), 3:(1, 0), 4:(1, 1), 5:(1, 2), 6:(2, 0), 7:(2, 1), 8:(2, 2) }

class GameField:
    def __init__(self):
        self.field = [[-1 for i in range(3)] for j in range(3)]
        
    def debug_print(self):
        for i in range(3):
            for j in range(3):
                print("%d " % self.field[i][j], end = " ")
            print("")
            
    def referee(self):
        if self.field[0][0] == self.field[0][1] and self.field[0][1] == self.field[0][2] and self.field[0][0] != -1:
            return self.field[0][0]
        if self.field[1][0] == self.field[1][1] and self.field[1][1] == self.field[1][2] and self.field[1][0] != -1:
            return self.field[1][0]       
        if self.field[2][0] == self.field[2][1] and self.field[2][1] == self.field[2][2] and self.field[2][0] != -1:
            return self.field[2][0]
        
        if self.field[0][0] == self.field[1][0] and self.field[1][0] == self.field[2][0] and self.field[0][0] != -1:
            return self.field[0][0]
        if self.field[0][1] == self.field[1][1] and self.field[1][1] == self.field[2][1] and self.field[0][1] != -1:
            return self.field[0][1]       
        if self.field[0][2] == self.field[1][2] and self.field[1][2] == self.field[2][2] and self.field[0][2] != -1:
            return self.field[0][2]
        
        if self.field[0][0] == self.field[1][1] and self.field[1][1] == self.field[2][2] and self.field[0][0] != -1:
            return self.field[0][0]
        if self.field[2][0] == self.field[1][1] and self.field[1][1] == self.field[0][2] and self.field[2][0] != -1:
            return self.field[2][0]
        
        for i in range(3):
            for j in range(3):
                if self.field[i][j] == -1:
                    return -2
                    
        return -1
    
    def clear(self):
        for i in range(3):
            for j in range(3):
                self.field[i][j] = -1
                
    def get_current(self, act):
        self.clear()
        for i in range(len(act)):
            #print("---------------")
            #print(act[i])
            (x, y) = get_pos[act[i]]
            self.field[x][y] = i % 2
        
    def get_net_input(self, player):
        inputLis = np.zeros((1, 3, 3, 4))
        for i in range(3):
            for j in range(3):
                if self.field[i][j] == 0:
                    inputLis[0][i][j][0] = 1
                elif self.field[i][j] == 1:
                    inputLis[0][i][j][1] = 1
        
        for i in range(3):
            for j in range(3):        
                inputLis[0][i][j][player + 2] = 1
        
        return inputLis
    
    def action_valid(self, act):
        (x, y) = get_pos[act]
        return self.field[x][y] == -1
    
    def play(self, act, player):
        (x, y) = get_pos[act]
        self.field[x][y] = player

## load net
from tensorflow.keras.layers import Input, Add, Dense, Activation,  BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform

with open('./model_json.json') as json_file:
    json_config = json_file.read()
    
model = tensorflow.keras.models.model_from_json(json_config)    
model.load_weights('./model_weights_only.h5')

model.compile(optimizer = 'adam', loss = ['mean_squared_error', 'categorical_crossentropy'])

## read game
def readObj(path):
    import pickle
    pkl = open(path, 'rb')
    obj = pickle.load(pkl)
    pkl.close()
    return obj

totGameCnt = 100

# train data
x_train = np.zeros((0, 3, 3, 4))
y_probs = np.zeros((0, 9))
y_result = np.zeros((0, 1))    


for gameCnt in range(totGameCnt):
    acts_p = "./game/" + str(gameCnt) + "/act.txt"
    prob_p = "./game/" + str(gameCnt) + "/pro.txt"
    res_p = "./game/" + str(gameCnt) + "/res.txt"
    
    acts = readObj(acts_p)
    probs = readObj(prob_p)
    res = readObj(res_p)
    
    #print(acts)
    
    # field
    gameField = GameField()

    # get y_result
    if res == -1:
        res_ar = np.array([0.])
        for act in acts:
            y_result = np.vstack((y_result, res_ar))
    else:
        for index in range(len(acts)):
            if index % 2 == res:
                y_result = np.vstack((y_result, np.array([1.])))
            else:
                y_result = np.vstack((y_result, np.array([-1.])))
    # get y_probs
    for prob in probs:
        y_probs = np.vstack((y_probs, np.array(prob)))
    # get x_train
    for index in range(len(acts)):
        x_in = gameField.get_net_input(index % 2)
        x_train = np.vstack((x_train, x_in))
        gameField.play(acts[index], index % 2)
    
model.fit([x_train], [y_result, y_probs], verbose = 1, validation_split = 0.01, batch_size = 16, epochs = 80)    

model.save_weights('./model_weights_only.h5')
json_config = model.to_json()
with open('./model_json.json', 'w') as json_file:
    json_file.write(json_config)
print("saved")    
