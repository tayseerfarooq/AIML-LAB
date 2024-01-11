class Graph:
    def __init__(self, graph, heuristic_values, start_node):
        self.graph = graph
        self.H = heuristic_values
        self.start = start_node
        self.parent, self.status, self.solution_graph = {}, {}, {}

    def apply_ao_star(self):
        self.ao_star(self.start, False)

    def get_neighbors(self, v):
        return self.graph.get(v, '')

    def get_status(self, v):
        return self.status.get(v, 0)

    def set_status(self, v, val):
        self.status[v] = val

    def get_heuristic_value(self, n):
        return self.H.get(n, 0)

    def set_heuristic_value(self, n, value):
        self.H[n] = value

    def print_solution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:", self.start)
        print("------------------------------------------------------------")
        print(self.solution_graph)
        print("------------------------------------------------------------")

    def compute_min_cost_child_nodes(self, v):
        min_cost, cost_to_child_nodes_dict = float('inf'), {}
        cost_to_child_nodes_dict[min_cost] = []

        for node_info_list in self.get_neighbors(v) or []:
            cost = sum(self.get_heuristic_value(c) + weight for c, weight in node_info_list)
            if cost < min_cost:
                min_cost = cost
                cost_to_child_nodes_dict[min_cost] = [c for c, _ in node_info_list]

        return min_cost, cost_to_child_nodes_dict[min_cost]

    def choose_parent(self, v, child_node_list):
        parents = [self.parent.get(child_node) for child_node in child_node_list if self.parent.get(child_node)]
        return max(parents, key=lambda p: self.get_heuristic_value(p), default=None)

    def ao_star(self, v, backtracking):
        print("HEURISTIC VALUES  :", self.H)
        print("SOLUTION GRAPH    :", self.solution_graph)
        print("PROCESSING NODE   :", v)
        print("-----------------------------------------------------------------------------------------")

        if self.get_status(v) >= 0:
            min_cost, child_node_list = self.compute_min_cost_child_nodes(v)
            self.set_heuristic_value(v, min_cost)
            self.set_status(v, len(child_node_list))

            solved = all(self.get_status(child_node) != -1 for child_node in child_node_list)

            if solved:
                self.set_status(v, -1)
                self.solution_graph[v] = child_node_list

            if v != self.start and not backtracking:
                self.parent[v] = self.choose_parent(v, child_node_list)
                self.ao_star(self.parent[v], True)

            if not backtracking:
                for child_node in child_node_list:
                    self.set_status(child_node, 0)
                    self.ao_star(child_node, False)


h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]
}
G1 = Graph(graph1, h1, 'A')
G1.apply_ao_star()
G1.print_solution()

h2 = {'A': 1, 'B': 6, 'C': 12, 'D': 10, 'E': 4, 'F': 4, 'G': 5, 'H': 7}
graph2 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'D': [[('E', 1), ('F', 1)]]
}

G2 = Graph(graph2, h2, 'A')
G2.apply_ao_star()
G2.print_solution()
