import tensorflow
import numpy as np
import copy

# load net
with open('./model_json.json') as json_file:
    json_config = json_file.read()
    
model = tensorflow.keras.models.model_from_json(json_config)    
model.load_weights('./model_weights_only.h5')

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
        if x >= 0 and y >= 0:
            self.field[x][y] = player
    
class Tree_node(object):
    def __init__(self, parent, pri_p, player):
        self.parent = parent
        self.children = {}
        self.visCnt = 0
        self.Q = 0 # value
        self.U = 0 # upper confidence
        self.P = pri_p # prior probability chosen
        self.player = player
        
    def is_leaf(self):
        return self.children == {}
    def is_root(self):
        return self.parent == None
    def get_value(self, c_puct):
        self.U = c_puct * self.P * np.sqrt(self.parent.visCnt) / (1 + self.visCnt)
        return self.U + self.Q
    '''
        action_priors := 
    '''
    def expand(self, action_priors):
        for items in action_priors.items():
            act, prob = items[0], items[1]
            if act not in self.children:
                self.children[act] = Tree_node(self, prob, self.player ^ 1)
                
    def select(self, c_puct):
        for child in self.children.values():
            child.get_value(c_puct)
        act = 0
        max_u = self.children[0].U
        for child in self.children.items():
            if child[1].U > max_u:
                max_u = child[1].U
                act = child[0]
        return act, self.children[act]
    
    def update(self, winner):
        if self.player == winner:
            add_val = 1.
        elif winner == .5:
            add_val = .5
        else:
            add_val = 0
            
        self.Q = self.visCnt * self.Q + add_val
        self.visCnt += 1
        self.Q /= self.visCnt
    
    def update_recur(self, val_res):
        self.update(val_res)
        if self.parent:
            self.parent.update_recur(val_res)
            
    def debug_print(self):
        print("----------------------")
        print("visCnt = %d " % self.visCnt, end = '')
        print("U = %d" % self.U)
        print(self.Q)
        print("----------------------")

gameField = GameField() 

# get input
import json
full_input = json.loads(input())
all_requests = full_input["requests"]
all_responses = full_input["responses"]

# get my color
if all_requests[0]["x"] == -1:
    mycolor = 0
else:
    mycolor = 1
    
# get field
for req in all_requests:
    if req["y"] >= 0 and req["x"] >= 0:
        gameField.field[req["y"]][req["x"]] = (mycolor ^ 1)
for resp in all_responses:
    gameField.field[resp["y"]][resp["x"]] = mycolor
    
gameField.debug_print()

x_input = gameField.get_net_input(mycolor)
predict = model.predict(x_input)

max_act_index = 0
max_act_prob = -1
for act in range(9):
    print(predict[1][0][act])
    if gameField.action_valid(act):
        if predict[1][0][act] > max_act_prob:
            max_act_index = act
            max_act_prob = predict[1][0][act]
# move
(y, x) = get_pos[max_act_index]
my_action = { "x":x, "y":y }

print(json.dumps({
    "response": my_action,
}))