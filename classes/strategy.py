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
        depth = 3
        node = Node(self.root_state)
        node.children = logic.get_possible_moves(self.root_state)
        
        v = -inf
        alpha = -inf
        beta = inf
        moveToDo = None
        
        # cas l'adversaire gagne : le noeud prend la valeur -1 et on ne génère pas la suite de l'arbre/enfants
        # cas on gagne : le noeud prend la valeur 1 
        # sinon, le noeud prend une valeur entre 0 et 1 (exclus)
        
        move = self.minimax(node, depth, True)
        
    def minimax(self, node, depth, maximizingPlayer):
        if depth == 0 or logic.is_game_over(self.player, self.root_state):
            return node
        
        if maximizingPlayer:
            value = -inf
            for child in node.children:
                value = max(value, self.minimax(child, depth - 1, False))       
        else:
            value = inf
            for child in node.children:
                value = min(value, self.minimax(child, depth - 1, True))
        
        return value
    
    def max_value(self, state, alpha, beta):
        
        return
    
    pass
        

str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
}