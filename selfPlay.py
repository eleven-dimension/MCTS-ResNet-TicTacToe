# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:32:23 2020

@author: Sophon
"""

import tensorflow
import numpy as np
import copy

def normalize(lis):
    tot = 0.0
    for i in lis:
        tot += i
    if tot == 0:
        return lis
    for i in lis:
        i /= tot
    return lis

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

# -------------------------------------------------

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
            if (act not in self.children) and prob != 0:
                self.children[act] = Tree_node(self, prob, self.player ^ 1)
                
    def select(self, c_puct):
        max_child_val = -10000.0
        max_child = 0
        
        for child in self.children.items():
            child_val = child[1].get_value(c_puct)
            if child_val > max_child_val:
                max_child = child[0]
                max_child_val = child_val
        return max_child, self.children[max_child]
      
    def update(self, leaf_value):
        self.visCnt += 1
        self.Q += 1. * (leaf_value - self.Q) / self.visCnt
    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)
        
    def debug_print(self):
        print("----------------------")
        print("visCnt = %d " % self.visCnt, end = '')
        print("U = %d" % self.U)
        print(self.Q)
        print("----------------------")


class MCTS:
    def __init__(self, cur_player, c_puct = 5, play_rounds = 500):
        self.root = Tree_node(None, 1.0, 0)
        self.rounds = play_rounds
        self.c_puct = c_puct
        
        self.past_move = []
    
    def set_situation(self):
        self.past_move = [4, 8, 3, 7]
    
    def play_out(self, gameField):
        moving_node = self.root
        moving_path = list(self.past_move)
        
        while not moving_node.is_leaf():
            act, moving_node = moving_node.select(self.c_puct)
            moving_path.append(act)
            
        gameField.get_current(moving_path)
        
        game_status = gameField.referee()

        if game_status == -2:
            act_probs = {}
            input_mat = gameField.get_net_input(moving_node.player)
            predict = model.predict(input_mat)            
            leaf_value = predict[0][0]

            tot_valid_prob = 0
            for act in range(9):
                if gameField.action_valid(act):
                    tot_valid_prob += predict[1][0][act]            
            for act in range(9):
                if gameField.action_valid(act):
                    act_probs[act] = predict[1][0][act] / tot_valid_prob    
                else:
                    act_probs[act] = 0            
            moving_node.expand(act_probs)
        else:
            if game_status == -1:
                leaf_value = .0
            else:
                leaf_value = (1.0 if moving_node.player == game_status else -1.0)
        moving_node.update_recursive(-leaf_value)
                
    def final_move(self, act):
        self.past_move.append(act)
        if act in self.root.children:
            self.root = self.root.children[act]
            self.root.parent = None
        else:
            self.root = Tree_node(None, 1.0, self.root.player ^ 1)
            
    def debug_print(self):
        for pr in self.root.children.items():
            print(pr[0])
            pr[1].debug_print()

    def get_act_prob(self, gameField, temp = 1e-7):
        for play_cnt in range(self.rounds):
            game = copy.deepcopy(gameField)
            self.play_out(game)

        child_vis_prob = [0 for i in range(9)]
        
        tot_vis = 0
        for child in self.root.children.items():
            tot_vis += child[1].visCnt
            
        for child in self.root.children.items():    
            child_vis_prob[child[0]] = child[1].visCnt / (tot_vis)
        return child_vis_prob

# -------------------------------------------------

# save file
def writeObj(obj, path):
    import pickle
    pkl = open(path, 'wb')
    pickle.dump(obj, pkl)
    pkl.close()

def mkFolder(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
        
# hold games
gameNum = 100
gameField = GameField()

for gameCounter in range(gameNum):
    print("-----------")
    print(gameCounter)
    
    gameField.clear()
    
    mkFolder('./game/' + str(gameCounter))
    mcts = MCTS(0)
    
    all_game_action = []
    all_act_prob = []
    
    cur_player = 1
    while gameField.referee() == -2:
        cur_player ^= 1
        acts_vis_prob = list(mcts.get_act_prob(gameField))        

        weight = np.ones(9)
        for index in range(9):
            if acts_vis_prob[index] == 0:
                weight[index] = 0
                
        all_acts = [i for i in range(9)]
        
        ## adjust prob vector
        proba = 0.75 * np.array(acts_vis_prob) + 0.25 * weight * np.random.dirichlet(0.3 * np.ones(9))
        
        gameField.debug_print()

        proba /= proba.sum()

        final_act = np.random.choice(all_acts, p = proba)
        mcts.final_move(final_act)
        
        all_game_action.append(final_act)
        all_act_prob.append(acts_vis_prob)
        
        gameField.play(final_act, cur_player)
    
    gameRes = gameField.referee()
    
    print(all_game_action)
    #print(all_act_prob)
    #print(gameRes)
    writeObj(all_game_action, "./game/" + str(gameCounter) + "/act.txt")
    writeObj(all_act_prob, "./game/" + str(gameCounter) + "/pro.txt")
    writeObj(gameRes, "./game/" + str(gameCounter) + "/res.txt")
    
    