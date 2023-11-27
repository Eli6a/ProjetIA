import copy
import math
import random
from math import log, sqrt, inf
from random import randrange
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress

import classes.logic as logic

# When implementing a new strategy add it to the `str2strat`
# dictionary at the end of the file


class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player

    def start(self):
        """
        This function select a tile from the board.

        @returns    (x, y) A tuple of integer corresponding to a valid
                    and free tile on the board.
        """
        raise NotImplementedError

class Node(object):
    """
    This class implements the main object that you will manipulate : nodes.
    Nodes include the state of the game (i.e. the 2D board), children (i.e. other children nodes), a list of
    untried moves, etc...
    """
    def __init__(self, board, move=(None, None),
                 wins=0, visits=0, children=None):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class Random(PlayerStrat):
    # Build here the class for a random player
    def start(self):
        possible_moves = logic.get_possible_moves(self.root_state)
        move = random.choice(possible_moves)
        return move

class MiniMax(PlayerStrat):
    # Build here the class implementing the MiniMax strategy
    def start(self):
        node = Node(self.root_state)
        
        depth = 2
        v = -inf
        alpha = -inf
        beta = inf
        moveToDo = None
        
        # cas l'adversaire gagne : le noeud prend la valeur -1 et on ne génère pas la suite de l'arbre/enfants
        # cas on gagne : le noeud prend la valeur 1 
        # sinon, le noeud prend une valeur entre 0 et 1 (exclus)
        
        node.move = self.minimax(node, depth, True)
        return node.move
        
    def minimax(self, node, depth, maximizingAI):
        list = logic.get_possible_moves(node.state)
        value, move = self.max_value(node.state, depth, node, -inf, inf, True)
        print(move)
        return move
    
    def max_value(self, state, depth, node, alpha, beta, maximizingAI):
        if depth == 0 :
            return 0, node.move
        elif (logic.is_game_over(playerNb, node.state) and not maximizingAI) :
            return -1, node.move
        elif (logic.is_game_over(minimaxNb, node.state) and maximizingAI):   
            return 1, node.move
            
            
            
            if (maximizingAI):
                print("max : Je suis ia :", self.player, " " , maximizingAI)
                return 1, node.move
            elif (logic.is_game_over(self.player, node.state) is None):
                print("max : Je suis ia :", self.player, " " , maximizingAI)
                return 0, (node.move)
            else:
                print("max : Je suis ia :", self.player, " " , maximizingAI)
                return 1, node.move
        
        v = -inf
        a1 = (-1, -1)
        
        list = logic.get_possible_moves(node.state)
        for element in list:
            copyState = copy.deepcopy(node.state)
            
            if (self.player == 1):
                copyState[element[0]][element[1]] = 1
            else:
                copyState[element[0]][element[1]] = 2
            
            child = Node(copyState, element) 
            node.add_child(child)
        
        for child in node.children:
            print(child.state)
            v2, a2 = self.min_value(child.state, depth - 1, child, alpha, beta, not maximizingAI)
            if v2 > v:
                v, a1 = v2, a2  
                if (v2 != 0):
                    print(a2)
        return v, a1
        
    def min_value(self, state, depth, node, alpha, beta, maximizingAI):
        
        if depth == 0 :
            return 0, node.move
        elif (logic.is_game_over(playerNb, node.state) and not maximizingAI) :
            return 1, node.move
        elif (logic.is_game_over(minimaxNb, node.state) and maximizingAI):    
            return -1, node.move
        
        v = +inf
        a1 = (-1, -1)
        
        list = logic.get_possible_moves(node.state)
        for element in list:
            copyState = copy.deepcopy(node.state)
            
            if (self.player == 1):
                copyState[element[0]][element[1]] = 1
            else:
                copyState[element[0]][element[1]] = 2
            
            child = Node(copyState, element) 
            node.add_child(child)
            
        for child in node.children:
            v2, a2 = self.max_value(child.state, depth - 1, child, alpha, beta, not maximizingAI)
            if v2 < v:
                v, a1 = v2, a2
                if (v2 != 0):
                    print(a2)

        return v, a1
playerNb = 0
randomNb = 1
minimaxNb = 2   

str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
}