import matplotlib.pyplot as plt
import networkx as nx

def bellman_ford(graph, start):
    distance = {node: float('inf') for node in graph}
    distance[start] = 0
    predecessor = {node: None for node in graph}

    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u]:
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u

    # Check negative-weight cycles
    for u in graph:
        for v, weight in graph[u]:
            if distance[u] + weight < distance[v]:
                raise Exception("Negative-weight cycle detected!")

    return distance, predecessor

def draw_graph(graph, distances, predecessors, start, end):
    G = nx.DiGraph()
    for u in graph:
        for v, w in graph[u]:
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # Draw all edges/nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Highlight path from start to end
    path = []
    current = end
    while current != start and current is not None:
        path.append((predecessors[current], current))
        current = predecessors[current]

    nx.draw_networkx_edges(G, pos, edgelist=path, edge_color='green', width=2.5)

    plt.title(f"Bellman-Ford Shortest Path from {start} to {end}")
    plt.show()

if __name__ == "__main__":
    graph = {
        'P': [('Q', 2), ('R', 4)],
        'Q': [('S', 2)],
        'R': [('Q', 1), ('T', 3)],
        'T': [('S', -5)],
        'S': []
    }

    start_node = 'P'
    end_node = 'S'

    distances, preds = bellman_ford(graph, start_node)

    print(f"Jarak terpendek dari {start_node} ke semua node:")
    for node in distances:
        print(f"  {start_node} → {node}: {distances[node]}")

    print(f"\nJarak terpendek dari {start_node} ke {end_node}: {distances[end_node]}")
    draw_graph(graph, distances, preds, start_node, end_node)
