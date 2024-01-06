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
        
    def add_children(self):
        list = logic.get_possible_moves(self.state)
        
        childPlayer = 0
        
        if (self.player == logic.WHITE_PLAYER):
            childPlayer=logic.BLACK_PLAYER
        else:
            childPlayer=logic.WHITE_PLAYER
            
        for element in list:
            copyState = copy.deepcopy(self.state)
            
            child = Node
            copyState[element[0]][element[1]] = self.player
            child = Node(copyState, element, player=childPlayer)
                
            self.add_child(child)


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
                
        if (len(node.untried_moves) > 14):
            possible_moves = logic.get_possible_moves(self.root_state)
            move = random.choice(possible_moves)
            return move
            
        node.move = self.minimax(node)
        return node.move
        
    def minimax(self, node):    
        alpha = -inf
        beta = inf
        value, move = self.min_value(node, alpha, beta)
        return move
    
    def max_value(self, node, alpha, beta):
        
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta)
            if v2 > v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if (v >= beta):
                return v, a1
        return v, a1
        
    def min_value(self, node, alpha, beta):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
         
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if (v <= alpha):
                    return v, a1

        return v, a1
    
    def haveWinner(self, node):
        otherPlayer = 0
        if (self.player == logic.BLACK_PLAYER):
            otherPlayer = logic.WHITE_PLAYER
        else:
            otherPlayer = logic.BLACK_PLAYER
            
        if (logic.is_game_over(self.player, node.state) == self.player) :
            return True, 200, node.move        
        elif (logic.is_game_over(otherPlayer, node.state) == otherPlayer) :
            return True, -200, node.move   
        else :
            return False, 0, node.move
        
class Evaluate(PlayerStrat):
    # Build here the class implementing the MiniMax strategy
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        depth = 0
        
        if (len(node.untried_moves) > 14):
            possible_moves = logic.get_possible_moves(self.root_state)
            move = random.choice(possible_moves)
            return move
        elif (len(node.untried_moves) > 10):
            depth = 4
        else :
            depth = 7
            
        v = -inf
        alpha = -inf
        beta = inf
        node.move = self.minimax(node, depth)
        return node.move
        
    def minimax(self, node, depth):
            
        value, move = self.max_value(node, -inf, inf, depth)
        return move
    
    def max_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth == 0):
            return self.evaluate(node)
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta, depth-1)
            if v2 > v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if (v >= beta):
                return v, a1
        return v, a1
        
    def min_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth == 0):
            return self.evaluate(node)
         
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta, depth-1)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if (v <= alpha):
                    return v, a1

        return v, a1
    
    def haveWinner(self, node):
        otherPlayer = 0
        if (self.player == logic.BLACK_PLAYER):
            otherPlayer = logic.WHITE_PLAYER
        else:
            otherPlayer = logic.BLACK_PLAYER
            
        if (logic.is_game_over(self.player, node.state) == self.player) :
            return True, 200, node.move        
        elif (logic.is_game_over(otherPlayer, node.state) == otherPlayer) :
            return True, -200, node.move   
        else :
            return False, 0, node.move
        
    
    def evaluate(self, node):
        one_third = len(node.state) // 3
        if (node.move[0] <= one_third or node.move[0] >= len(node.state) - one_third):
            return 1, node.move
        else:
            return 9, node.move

str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
        "evaluate": Evaluate,
}