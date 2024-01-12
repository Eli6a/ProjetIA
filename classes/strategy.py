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

import time
import classes.logic as logic
import classes.graph as graph

# When implementing a new strategy add it to the `str2strat`
# dictionary at the end of the file


class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player
        
        if (self.player == logic.BLACK_PLAYER):
            self.otherPlayer = logic.WHITE_PLAYER
        else:
            self.otherPlayer = logic.BLACK_PLAYER

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
            
        startTime = time.time()
        node.move = self.minimax(node)
        self.displayTime(startTime)
        
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
        if (logic.is_game_over(self.player, node.state) == self.player) :
            return True, 200, node.move        
        elif (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer) :
            return True, 200, node.move   
        else :
            return False, 0, node.move
        
    def displayTime(self, startTime):
        timeElapsed = time.time() - startTime
        hours, rem = divmod(timeElapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        timeString = ""
        if (hours > 0):
            timeString += "{:0>2}h".format(int(hours)) + " "
        if (minutes > 0):
            timeString += "{:0>2}m".format(int(minutes)) + " "
        if (seconds > 0):
            timeString += "{:05.2f}s".format(seconds)

        print(timeString)
        
class Evaluate(PlayerStrat):
    """Joue d'abord au centre, puis essaie de faire une ligne (verticale ou horizontale en fonction de la couleur)"""
    # Build here the class implementing the MiniMax strategy
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        matrixLength = len(self.root_state)
        middle = matrixLength // 2

        # si le centre d'un plateau impair est vide, on joue dessus
        if (len(self.root_state) % 2 != 0 and self.root_state[middle][middle] == 0):
            return middle, middle
        
        depth = 0
        if (len(node.untried_moves) > 10):
            self.maxDepth = 4
        else:
            self.maxDepth = 7
            
        v = -inf
        alpha = -inf
        beta = inf
        
        startTime = time.time()
        node.move = self.minimax(node, depth)
        self.displayTime(startTime)
        
        return node.move
        
    def minimax(self, node, depth):  
        value, move = self.max_value(node, -inf, inf, depth)
        return move
    
    def max_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth >= self.maxDepth):
            # si l'adversaire va gagner, on essaie de le bloquer
            for row in range(len(self.root_state)):
                for col in range(len(self.root_state)):
                    if (self.root_state[row][col] == 0):
                        willWin, value, move = self.willWin(Node(self.root_state, player=self.player), row, col)
                        if (willWin):
                            return value, move
            return self.evaluate()
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta, depth+1)
            if v2 >= v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if (v >= beta):
                return v, a1
        return v, a1
        
    def min_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth >= self.maxDepth):
            return self.evaluate()
        
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta, depth+1)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if (v <= alpha):
                    return v, a1

        return v, a1
    
    def haveWinner(self, node):            
        if (logic.is_game_over(self.player, node.state) == self.player) :
            return True, 200, node.move        
        elif (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer) :
            return True, 200, node.move   
        else :
            return False, 0, node.move   
             
    def willWin(self, node, x, y):
        
        originalStateXY = node.state[x][y]
        
        node.state[x][y] = self.player
        if (logic.is_game_over(self.player, node.state) == self.player and originalStateXY != self.otherPlayer) :
            node.state[x][y] = originalStateXY
            return True, 200, (x, y)  
        
        node.state[x][y] = self.otherPlayer      
        if (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer and originalStateXY != self.player) :
            node.state[x][y] = originalStateXY
            return True, 200, (x, y)   
        
        node.state[x][y] = originalStateXY
        return False, 0, node.move
        
    
    def evaluate(self):      
        
        matrixLength = len(self.root_state)
        middle = matrixLength // 2            

        # matrice des valeurs des cases
        valueMatrix = [[0] * matrixLength for _ in range(matrixLength)]
        bestValue = -inf
        bestMove = (-1, -1)
        
        if (self.player == logic.BLACK_PLAYER): 
            for row in range(matrixLength):
                otherOnSameRow = 0
                for col in  range(matrixLength):
                    # si le coup permet de gagner ou bloquer l'adversaire, on le joue
                    willWin, value, move = self.willWin(Node(self.root_state, player=self.player), row, col)
                    if (willWin):
                        return value, move
                    
                    if (self.root_state[row][col] == 0):
                        # donne une valeur forte au milieu du plateau, et plus on s'éloigne, plus la valeur diminue
                        valueMatrix[row][col] += (matrixLength - abs(middle - row) - abs(middle - col)) / 2
                        
                        # donne une valeur forte sur les lignes horizontales centrales
                        valueMatrix[row][col] += (matrixLength - abs(middle - row))
                        
                        # donne une valeur plus forte sur la diagonale bas-gauche haut-droite   
                        distanceDiagonale = abs((row + col) - (matrixLength - 1))
                        valueMatrix[row][col] += (matrixLength - distanceDiagonale) / 4
                            
                    elif (self.root_state[row][col] == self.otherPlayer):
                        otherOnSameRow += 1
                        
                for col in range(matrixLength):
                    # réduit la valeur des cases quand l'adversaire a joué sur la même ligne         
                    if (otherOnSameRow > 0):
                        valueMatrix[row][col] = max(0, valueMatrix[row][col] - otherOnSameRow)
                    
                    # récupère le meilleur coup    
                    if (valueMatrix[row][col] > bestValue):
                        bestValue = valueMatrix[row][col]
                        bestMove = (row, col)
                       
        else:       # le joueur est blanc
            for col in range(matrixLength):
                otherOnSameCol = 0
                for row in  range(matrixLength):
                    # si le coup permet de gagner ou bloquer l'adversaire, on le joue
                    willWin, value, move = self.willWin(Node(self.root_state, player=self.player), row, col)
                    if (willWin):
                        return value, move
                    
                    if (self.root_state[row][col] == 0):
                        # donne une valeur forte au milieu du plateau, et plus on s'éloigne, plus la valeur diminue
                        valueMatrix[row][col] += (matrixLength - abs(middle - row) - abs(middle - col)) / 2
                        
                        # donne une valeur forte sur les lignes verticales centrales
                        valueMatrix[row][col] += (matrixLength - abs(middle - col))
                            
                        # donne une valeur plus forte sur la diagonale bas-gauche haut-droite   
                        distanceDiagonale = abs((row + col) - (matrixLength - 1))
                        valueMatrix[row][col] += (matrixLength - distanceDiagonale) / 4       
                            
                    elif (self.root_state[row][col] == self.otherPlayer):
                        otherOnSameCol += 1
                        
                for row in range(matrixLength):
                    # réduit la valeur des cases quand l'adversaire a joué sur la même colonne           
                    if (otherOnSameCol > 0):
                        valueMatrix[row][col] = max(0, valueMatrix[row][col] - otherOnSameCol)
                        
                    # récupère le meilleur coup    
                    if (valueMatrix[row][col] > bestValue):
                        bestValue = valueMatrix[row][col]
                        bestMove = (row, col)                
             
        return bestValue, bestMove   
    
    def displayTime(self, startTime):
        timeElapsed = time.time() - startTime
        hours, rem = divmod(timeElapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        timeString = ""
        if (hours > 0):
            timeString += "{:0>2}h".format(int(hours)) + " "
        if (minutes > 0):
            timeString += "{:0>2}m".format(int(minutes)) + " "
        if (seconds > 0):
            timeString += "{:05.2f}s".format(seconds)

        print(timeString)
    
class ShortPath(PlayerStrat):
    """Essaie de faire le chemin le plus court entre les bords du plateau
    Utilise prioritairement les lignes sur lesquelles l'IA a le plus joué"""
    # Build here the class implementing the MiniMax strategy
    
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        depth = 0
        if (len(node.untried_moves) > 10):
            self.maxDepth = 4
        else:
            self.maxDepth = 7
            
        v = -inf
        alpha = -inf
        beta = inf
        
        startTime = time.time()
        node.move = self.minimax(node, depth)
        self.displayTime(startTime)
        
        return node.move
        
    def minimax(self, node, depth):  
        value, move = self.max_value(node, -inf, inf, depth)
        return move
    
    def max_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth == self.maxDepth):
            return self.evaluate(node, self.player, self.otherPlayer)
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta, depth+1)
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
            
        if (depth == self.maxDepth):
            evaluation, move = self.evaluate(node, self.otherPlayer, self.player)
            if (evaluation < 0):
                return evaluation, move
            else:
                return evaluation/2, move
        
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta, depth+1)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if (v <= alpha):
                    return v, a1

        return v, a1
    
    def haveWinner(self, node):            
        if (logic.is_game_over(self.player, node.state) == self.player) :
            return True, 200, node.move        
        elif (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer) :
            return True, 200, node.move   
        else :
            return False, 0, node.move
             
    def willWin(self, node, x, y):
        
        originalStateXY = node.state[x][y]
        
        node.state[x][y] = self.player
        if (logic.is_game_over(self.player, node.state) == self.player and originalStateXY != self.otherPlayer) :
            node.state[x][y] = originalStateXY
            return True, 200, (x, y)  
        
        node.state[x][y] = self.otherPlayer      
        if (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer and originalStateXY != self.player) :
            node.state[x][y] = originalStateXY
            return True, 200, (x, y)   
        
        node.state[x][y] = originalStateXY
        return False, 0, node.move    
    
    def evaluate(self, node, player, otherPlayer):
        
        costMin = +inf
        shortestPath = []
        
        g = graph.Graph(self.root_state, player, otherPlayer)

        if (player == logic.BLACK_PLAYER):
            leftEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if y == 0]
            rightEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if y == len(self.root_state)-1]
            
            ownOnSameRows = []
            for row in range(len(self.root_state)):
                ownOnSameRow = 0
                for col in range(len(self.root_state)):
                    if (self.root_state[row][col] == player):
                        ownOnSameRow += 1
                ownOnSameRows.append(ownOnSameRow)
            
            sorted_indices = []
            if (all(element == 0 for element in ownOnSameRows)):
                sorted_indices = [1, 0, len(self.root_state)-2, len(self.root_state)-1]
            else:
                sorted_indices = sorted(range(len(ownOnSameRows)), key=lambda i: ownOnSameRows[i], reverse=True)[:4]
            
            for start in sorted_indices:
                if (not any(couple == (start, 0) for couple in leftEdge)):
                    continue
                for end in sorted_indices:
                    if (not any(couple == (end, len(self.root_state)-1) for couple in rightEdge)):
                        continue
                
                    start_node = graph.Node(start, 0, 0)
                    end_node = graph.Node(end, len(self.root_state)-1, 0)
                    path = graph.astar(g, start_node, end_node)
                    if (path == None):
                        continue
                    cost = graph.path_cost(path, g)
                    if (cost < costMin):
                        costMin = cost 
                        shortestPath = path
            
            # si ne trouve pas un chemin sur les lignes les plus jouées, on prend un chemin aléatoire           
            if (shortestPath == []):
                start = random.choice(leftEdge)
                end = random.choice(rightEdge)
                start_node = graph.Node(start[0], 0, 0)
                end_node = graph.Node(end[0], len(self.root_state)-1, 0)
                path = graph.astar(g, start_node, end_node)
                if (path != None):
                    cost = graph.path_cost(path, g)
                    if (cost < costMin):
                        costMin = cost 
                        shortestPath = path
                    
        else:
            upEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if x == 0]
            bottomEdge = [(x, y) for x, y in logic.get_possible_moves(self.root_state) if x == len(self.root_state)-1]
            
            ownOnSameCols = []
            for col in range(len(self.root_state)):
                ownOnSameCol = 0
                for row in range(len(self.root_state)):
                    if (self.root_state[row][col] == player):
                        ownOnSameCol += 1
                ownOnSameCols.append(ownOnSameCol)
            
            sorted_indices = []
            if (all(element == 0 for element in ownOnSameCols)):
                sorted_indices = [1, 0, len(self.root_state)-2, len(self.root_state)-1]
            else:
                sorted_indices = sorted(range(len(ownOnSameCols)), key=lambda i: ownOnSameCols[i], reverse=True)[:4]
            
            for start in sorted_indices:
                if (not any(couple == (0, start) for couple in upEdge)):
                    continue
                for end in sorted_indices:
                    if (not any(couple == (len(self.root_state)-1, end) for couple in bottomEdge)):
                        continue
                
                    start_node = graph.Node(0, start, 0)
                    end_node = graph.Node(len(self.root_state)-1, end, 0)
                    path = graph.astar(g, start_node, end_node)
                    if (path == None):
                        continue
                    cost = graph.path_cost(path, g)
                    if (cost < costMin):
                        costMin = cost 
                        shortestPath = path
                        
            # si ne trouve pas un chemin sur les lignes les plus jouées, on prend un chemin aléatoire           
            if (shortestPath == []):
                start = random.choice(upEdge)
                end = random.choice(bottomEdge)
                start_node = graph.Node(0, start[1], 0)
                end_node = graph.Node(len(self.root_state)-1, end[1], 0)
                path = graph.astar(g, start_node, end_node)
                if (path != None):
                    cost = graph.path_cost(path, g)
                    if (cost < costMin):
                        costMin = cost 
                        shortestPath = path

        move = (-1, -1)
        if (shortestPath == []):
            return -10, move  
        elif (len(shortestPath) == 1):
            if (logic.is_node_free(shortestPath[0], self.root_state)):
                move = shortestPath[0]
            else:
                return -10, move
        else:              
            move = shortestPath[len(shortestPath)//2]
            
        busyNode = not logic.is_node_free(move, self.root_state)

        while (busyNode):
            shortestPath.pop(len(shortestPath)//2)
            if (len(shortestPath) == 1):
                move = shortestPath[0]
            else:
                move = shortestPath[len(shortestPath)//2]
            busyNode = not logic.is_node_free(move, self.root_state)
        
        if (not logic.is_node_free(move, self.root_state)): 
            return -10, move
        
        return abs(50 - costMin), move
    
    def displayTime(self, startTime):
        timeElapsed = time.time() - startTime
        hours, rem = divmod(timeElapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        timeString = ""
        if (hours > 0):
            timeString += "{:0>2}h".format(int(hours)) + " "
        if (minutes > 0):
            timeString += "{:0>2}m".format(int(minutes)) + " "
        if (seconds > 0):
            timeString += "{:05.2f}s".format(seconds)

        print(timeString)
    
class MonteCarlo(PlayerStrat):
    # Build here the class implementing the MiniMax strategy
    def start(self):
        node = Node(self.root_state, player=self.player)
        
        matrixLength = len(self.root_state)
        middle = matrixLength // 2

        # si le centre d'un plateau impair est vide, on joue dessus
        if (len(self.root_state) % 2 != 0 and self.root_state[middle][middle] == 0):
            return middle, middle
        
        depth = 0
        if (len(node.untried_moves) > 10):
            self.maxDepth = 4
        else:
            self.maxDepth = 7
            
        v = -inf
        alpha = -inf
        beta = inf
           
        startTime = time.time()
        node.move = self.minimax(node, depth)
        self.displayTime(startTime)

        return node.move
        
    def minimax(self, node, depth):  
        value, move = self.max_value(node, -inf, inf, depth)
        return move
    
    def max_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth >= self.maxDepth):
            # si l'adversaire va gagner, on essaie de le bloquer
            for row in range(len(self.root_state)):
                for col in range(len(self.root_state)):
                    if (self.root_state[row][col] == 0):
                        willWin, value, move = self.willWin(Node(self.root_state, player=self.player), row, col)
                        if (willWin):
                            return value, move
            return self.evaluate(node, node.player)
        
        v = -inf
        a1 = (-1, -1)
        
        node.add_children()
        
        for child in node.children:
            v2, a2 = self.min_value(child, alpha, beta, depth+1)
            if v2 >= v:
                v, a1 = v2, a2  
                alpha = max(alpha, v)
            if (v >= beta):
                return v, a1
        return v, a1
        
    def min_value(self, node, alpha, beta, depth):
        haveWinner, value, move = self.haveWinner(node)
        if (haveWinner):
            return value, move
            
        if (depth >= self.maxDepth):
            return self.evaluate(node, node.player)
        
        v = +inf
        a1 = (-1, -1)
                
        node.add_children()
            
        for child in node.children:
            v2, a2 = self.max_value(child, alpha, beta, depth+1)
            if v2 < v:
                v, a1 = v2, a2
                beta = min(beta, v)
                if (v <= alpha):
                    return v, a1

        return v, a1
    
    def haveWinner(self, node):            
        if (logic.is_game_over(self.player, node.state) == self.player) :
            return True, 200, node.move        
        elif (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer) :
            return True, 200, node.move   
        else :
            return False, 0, node.move    
        
    def willWin(self, node, x, y):
        
        originalStateXY = node.state[x][y]
        
        node.state[x][y] = self.player
        if (logic.is_game_over(self.player, node.state) == self.player and originalStateXY != self.otherPlayer) :
            node.state[x][y] = originalStateXY
            return True, 200, (x, y)  
        
        node.state[x][y] = self.otherPlayer      
        if (logic.is_game_over(self.otherPlayer, node.state) == self.otherPlayer and originalStateXY != self.player) :
            node.state[x][y] = originalStateXY
            return True, 200, (x, y)   
        
        node.state[x][y] = originalStateXY
        return False, 0, node.move       
    
    def evaluate(self, node, currentPlayer):      
        haveWinner = False
        
        move = (-1, -1)
        while (not haveWinner):
            move = random.choice(node.untried_moves)
            node = Node(node.state, move, player=currentPlayer)
            
            currentPlayer = self.changePlayer(currentPlayer)
            
            move = random.choice(node.untried_moves)
            node = Node(node.state, move, player=currentPlayer)
            
            currentPlayer = self.changePlayer(currentPlayer)
                
            haveWinner, value, move = self.haveWinner(node)
            
        
        # a un gagnant
        return value, move
      
    def changePlayer(self, player):
        if (player == logic.BLACK_PLAYER):
            return logic.WHITE_PLAYER
        else:
            return logic.BLACK_PLAYER  
        
    def displayTime(self, startTime):
        timeElapsed = time.time() - startTime
        hours, rem = divmod(timeElapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        timeString = ""
        if (hours > 0):
            timeString += "{:0>2}h".format(int(hours)) + " "
        if (minutes > 0):
            timeString += "{:0>2}m".format(int(minutes)) + " "
        if (seconds > 0):
            timeString += "{:05.2f}s".format(seconds)

        print(timeString)   
        
str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
        "evaluate": Evaluate,
        "shortpath": ShortPath,
        "montecarlo": MonteCarlo
}