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
                 wins=0, visits=0, children=None, player=2):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)
        
        self.player = player

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
        node = Node(self.root_state, player=self.player)
        
        depth = 2
        v = -inf
        alpha = -inf
        beta = inf
        moveToDo = None
        
        # cas l'adversaire gagne : le noeud prend la valeur -1 et on ne génère pas la suite de l'arbre/enfants
        # cas on gagne : le noeud prend la valeur 1 
        # sinon, le noeud prend une valeur entre 0 et 1 (exclus)
        
        node.move = self.minimax(node)
        return node.move
        
    def minimax(self, node):
        
        if (node.player == logic.BLACK_PLAYER):
            print("noir")
        else:
            print("blanc")
        list = logic.get_possible_moves(node.state)
        if (len(list) > 14):
            return random.choice(list)
        value, move = self.max_value(node.state, node, -inf, inf)
        return move
    
    def max_value(self, state, node, alpha, beta):
        if (logic.is_game_over(logic.BLACK_PLAYER, node.state) == logic.BLACK_PLAYER) :
            if (node.player == logic.BLACK_PLAYER):
                return 1, node.move
            else:
                return -1, node.move            
        elif (logic.is_game_over(logic.WHITE_PLAYER, node.state) == logic.WHITE_PLAYER) :
            if (node.player == logic.WHITE_PLAYER):
                return 2, node.move
            else:
                return -2, node.move  
        
        v = -inf
        a1 = (-1, -1)
        
        list = logic.get_possible_moves(node.state)
            
        for element in list:
            copyState = copy.deepcopy(node.state)
            
            child = Node

            if (node.player == logic.WHITE_PLAYER):
                copyState[element[0]][element[1]] = logic.WHITE_PLAYER
                child = Node(copyState, element, player=logic.BLACK_PLAYER)
            else:
                copyState[element[0]][element[1]] = logic.BLACK_PLAYER
                child = Node(copyState, element, player=logic.WHITE_PLAYER)
                
            node.add_child(child)
        
        for child in node.children:
            v2, a2 = self.min_value(child.state, child, alpha, beta)
            if v2 > v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if (v >= beta):
                return v, a1
        return v, a1
        
    def min_value(self, state, node, alpha, beta):
        if (logic.is_game_over(logic.BLACK_PLAYER, node.state) == logic.BLACK_PLAYER) :
            if (node.player == logic.BLACK_PLAYER):
                return 2, node.move
            else:
                return -2, node.move            
        elif (logic.is_game_over(logic.WHITE_PLAYER, node.state) == logic.WHITE_PLAYER) :
            if (node.player == logic.WHITE_PLAYER):
                return 2, node.move
            else:
                return -2, node.move 
        
        v = +inf
        a1 = (-1, -1)
        
        list = logic.get_possible_moves(node.state)
            
        for element in list:
            copyState = copy.deepcopy(node.state)
            
            child = Node

            if (node.player == logic.WHITE_PLAYER):
                copyState[element[0]][element[1]] = logic.WHITE_PLAYER
                child = Node(copyState, element, player=logic.BLACK_PLAYER)
            else:
                copyState[element[0]][element[1]] = logic.WHITE_PLAYER
                child = Node(copyState, element, player=logic.WHITE_PLAYER)
                
            node.add_child(child)
            
        for child in node.children:
            v2, a2 = self.max_value(child.state, child, alpha, beta)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                
                if (v <= alpha):
                    return v, a1

        return v, a1

str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
}