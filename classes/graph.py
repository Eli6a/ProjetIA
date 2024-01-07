import heapq

class Graph:
    """Représentation du plateau par une liste adjacence d'un graphe non orienté"""
    def __init__(self, board, otherPlayer):
        self.nodes = {}
        for row in range(len(board)):
            for col in range(len(board[0])):
                node = Node(row, col, board[row][col])
                self.add_node(node)

                # N'ajoute pas d'arc si la case est occupée par l'autre joueur
                if (board[row][col] == otherPlayer):
                    continue
                
                # Connexions avec les voisins
                neighbors = [
                    (row - 1, col), (row + 1, col),
                    (row, col - 1), (row, col + 1),
                    (row - 1, col + 1), (row + 1, col - 1)
                ]
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < len(board) and 0 <= neighbor[1] < len(board[0]):
                        # Si la case voisine n'est pas occupée par l'autre joueur, on ajoute un arc
                        if (board[neighbor[0]][neighbor[1]] != otherPlayer):
                            self.add_edge(node, Node(neighbor[0], neighbor[1], board[neighbor[0]][neighbor[1]]))
                                                
    def add_node(self, node):
        if node.position not in self.nodes:
            self.nodes[node.position] = []

    def remove_node(self, node):
        if node.position in self.nodes:
            del self.nodes[node.position]
            for n in self.nodes:
                self.nodes[n] = [neighbor for neighbor in self.nodes[n] if neighbor.position != node.position]

    def add_edge(self, node1, node2, weight=1):
        if node1.position in self.nodes and node2.position in self.nodes:
            self.nodes[node1.position].append((node2, weight))
            self.nodes[node2.position].append((node1, weight))
            
    def displayGraph(self) -> str:
        for node, neighbors in self.nodes.items():
            print(f"{node}: {neighbors}")

class Node:
    def __init__(self, x, y, player):
        self.position = (x, y)
        self.player = player
        self.g_cost = 0 
        self.h_cost = 0  
        self.f_cost = 0 
        self.parent = None
        
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __repr__(self):
        return f"Node({self.position}, {self.player})"


def astar(graph, start, end):
    open_set = []
    closed_set = set()

    start_node = start
    end_node = end

    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.position == end_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)

        neighbors = graph.nodes[current_node.position]
        for neighbor_data in neighbors:
            neighbor, weight = neighbor_data[0], neighbor_data[1]
            neighbor_node = Node(neighbor.position[0], neighbor.position[1], neighbor.player)
            neighbor_node.g_cost = current_node.g_cost + weight
            neighbor_node.h_cost = manhattan_distance(neighbor_node, end_node)
            neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
            neighbor_node.parent = current_node

            if neighbor_node.position in closed_set:
                continue

            if neighbor_node not in open_set:
                heapq.heappush(open_set, neighbor_node)
            elif neighbor_node.g_cost < current_node.g_cost:
                current_node = neighbor_node

    return None

def manhattan_distance(node1, node2):
    return abs(node1.position[0] - node2.position[0]) + abs(node1.position[1] - node2.position[1])


