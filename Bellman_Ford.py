def bellman_ford(graph, start):
    distance = {node: float('inf') for node in graph}
    distance[start] = 0
    predecessor = {node: None for node in graph}

    # Relaksasi semua edge sebanyak |V| - 1 kali
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u]:
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u

    # Cek siklus negatif
    for u in graph:
        for v, weight in graph[u]:
            if distance[u] + weight < distance[v]:
                raise Exception("Terdapat siklus dengan bobot negatif!")

    return distance, predecessor

def get_path(predecessor, start, end):
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = predecessor[current]
    if path[0] != start:
        return None  # tidak ada jalur
    return path

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

    path_to_S = get_path(preds, start_node, end_node)

    print("\nJalur tercepat dari P ke S:")
    if path_to_S:
        print(" → ".join(path_to_S))
        print(f"Total jarak: {distances[end_node]}")
    else:
        print("Tidak ada jalur dari P ke S")
