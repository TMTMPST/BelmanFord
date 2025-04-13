import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Interactive Bellman-Ford Shortest Path Visualizer", layout="wide")
st.title("Interactive Bellman-Ford Shortest Path Visualizer")

# Initialize session state variables if they don't exist
if 'graph' not in st.session_state:
    st.session_state.graph = {}
if 'nodes' not in st.session_state:
    st.session_state.nodes = []
if 'start_node' not in st.session_state:
    st.session_state.start_node = None
if 'end_node' not in st.session_state:
    st.session_state.end_node = None
if 'node_positions' not in st.session_state:
    st.session_state.node_positions = {}
if 'graph_modified' not in st.session_state:
    st.session_state.graph_modified = True
if 'edge_table' not in st.session_state:
    st.session_state.edge_table = pd.DataFrame(columns=['Source', 'Target', 'Weight'])

def bellman_ford(graph, start):
    distance = {node: float('inf') for node in graph}
    distance[start] = 0
    predecessor = {node: None for node in graph}
    
    # Relax all edges |V| - 1 times
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u]:
                if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    predecessor[v] = u
    
    # Check for negative weight cycles
    for u in graph:
        for v, weight in graph[u]:
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                return None, None, "Graph contains a negative weight cycle!"
    
    return distance, predecessor, None

def get_path(predecessor, start, end):
    if predecessor is None:
        return None
    
    path = []
    current = end
    
    while current is not None:
        path.insert(0, current)
        if current == start:
            break
        current = predecessor.get(current)
        
        # If we can't reach the start node
        if current is None and path[0] != start:
            return None
    
    # Check if path actually starts with the start node
    if not path or path[0] != start:
        return None
        
    return path

def generate_positions(graph):
    # Create a networkx graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in graph:
        G.add_node(node)
    
    for node, edges in graph.items():
        for target, weight in edges:
            G.add_edge(node, target, weight=weight)
    
    # Generate positions
    return nx.spring_layout(G, seed=42)

def draw_graph(graph, start_node=None, end_node=None, path=None, distances=None):
    if not graph:  # If graph is empty
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Graph is empty. Please add nodes and edges.", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.set_axis_off()
        return fig
    
    # Create a networkx graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in graph:
        G.add_node(node)
    
    for node, edges in graph.items():
        for target, weight in edges:
            G.add_edge(node, target, weight=weight)
    
    # Generate or use existing positions
    if st.session_state.graph_modified or not st.session_state.node_positions or len(st.session_state.node_positions) != len(graph):
        st.session_state.node_positions = generate_positions(graph)
        st.session_state.graph_modified = False
    
    pos = st.session_state.node_positions
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Node colors
    node_colors = ['lightblue'] * len(graph)
    
    # Prepare node list for coloring
    nodes_list = list(graph.keys())
    
    # Highlight the path nodes
    if path:
        path_nodes = {node: i for i, node in enumerate(path)}
        for i, node in enumerate(nodes_list):
            if node in path_nodes:
                node_colors[i] = 'lightgreen'
    
    # Highlight start and end nodes
    if start_node and start_node in nodes_list:
        start_idx = nodes_list.index(start_node)
        node_colors[start_idx] = 'green'
    
    if end_node and end_node in nodes_list:
        end_idx = nodes_list.index(end_node)
        node_colors[end_idx] = 'red'
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
    
    # Draw edges
    edges = list(G.edges())
    edge_colors = ['black'] * len(edges)
    
    # Highlight the path edges
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        for i, edge in enumerate(edges):
            if edge in path_edges:
                edge_colors[i] = 'green'
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, 
                          connectionstyle='arc3, rad=0.1', arrows=True, 
                          arrowsize=15, width=1.5, ax=ax)
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    # Draw node labels
    labels = {}
    for node in graph:
        if distances and node in distances:
            if distances[node] == float('inf'):
                labels[node] = f"{node}\n∞"
            else:
                labels[node] = f"{node}\n{distances[node]}"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, ax=ax)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Node'),
        Patch(facecolor='green', edgecolor='black', label='Start Node'),
        Patch(facecolor='red', edgecolor='black', label='End Node'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Path Node')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove axis
    ax.set_axis_off()
    
    return fig

def update_graph_from_edge_table():
    # Clear the current graph but keep the nodes
    st.session_state.graph = {node: [] for node in st.session_state.nodes}
    
    # Add edges from the edge table
    for _, row in st.session_state.edge_table.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row['Weight']
        
        # Add the edge
        st.session_state.graph[source].append((target, weight))
    
    st.session_state.graph_modified = True

# Tutorial expander
with st.expander("How to use this app", expanded=True):
    st.markdown("""
    ### Steps to use this interactive Bellman-Ford Shortest Path Visualizer:
    
    1. **Create Nodes**: First, add nodes to your graph using the 'Node Configuration' section.
    2. **Add Edges**: Once nodes are created, define connections between them with weights using the 'Edge Configuration' section.
    3. **Select Start/End Nodes**: Choose which nodes to calculate the shortest path between.
    4. **View Results**: The visualization will update to show the shortest path, and detailed results will appear on the right panel.
    
    **Tips:**
    - You can edit or delete existing edges using the edge table.
    - Use negative weights to test the algorithm's capability with negative values.
    - Watch for negative cycles which Bellman-Ford can detect.
    - The distances are displayed under each node in the visualization.
    """)

# Main layout
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Graph Configuration")
    
    # Node Configuration
    st.subheader("Node Configuration")
    
    node_options = st.radio("Choose an option for nodes:", ["Generate Nodes", "Custom Nodes"])
    
    if node_options == "Generate Nodes":
        num_nodes = st.number_input("Number of Nodes", min_value=1, max_value=26, value=5, step=1)
        
        if st.button("Generate Nodes"):
            # Use uppercase letters for node names (A, B, C, ...)
            st.session_state.nodes = [chr(65 + i) for i in range(min(num_nodes, 26))]
            # Initialize an empty graph with these nodes
            st.session_state.graph = {node: [] for node in st.session_state.nodes}
            # Reset edge table
            st.session_state.edge_table = pd.DataFrame(columns=['Source', 'Target', 'Weight'])
            st.session_state.graph_modified = True
            st.session_state.start_node = None
            st.session_state.end_node = None
            st.success(f"Created {num_nodes} nodes: {', '.join(st.session_state.nodes)}")
    else:
        custom_nodes = st.text_input("Enter custom node names (separated by commas)", "A,B,C,D,E")
        
        if st.button("Create Custom Nodes"):
            nodes = [node.strip() for node in custom_nodes.split(",") if node.strip()]
            if len(nodes) == len(set(nodes)):  # Check for duplicates
                st.session_state.nodes = nodes
                # Initialize an empty graph with these nodes
                st.session_state.graph = {node: [] for node in st.session_state.nodes}
                # Reset edge table
                st.session_state.edge_table = pd.DataFrame(columns=['Source', 'Target', 'Weight'])
                st.session_state.graph_modified = True
                st.session_state.start_node = None
                st.session_state.end_node = None
                st.success(f"Created custom nodes: {', '.join(st.session_state.nodes)}")
            else:
                st.error("Duplicate node names are not allowed!")
    
    # Edge Configuration
    if st.session_state.nodes:
        st.subheader("Edge Configuration")
        
        # Create columns for the edge input form
        source_col, target_col, weight_col, button_col = st.columns([1, 1, 1, 1])
        
        with source_col:
            source_node = st.selectbox("Source", st.session_state.nodes, key="source_select")
        
        with target_col:
            target_options = [n for n in st.session_state.nodes if n != source_node]
            target_node = st.selectbox("Target", target_options if target_options else ["None"], key="target_select")
        
        with weight_col:
            edge_weight = st.number_input("Weight", value=1, step=1, key="weight_input")
        
        with button_col:
            st.write("") # Spacer for alignment
            st.write("") # Spacer for alignment
            if st.button("Add Edge") and target_node != "None":
                # Check if edge already exists
                exists = False
                for idx, row in st.session_state.edge_table.iterrows():
                    if row['Source'] == source_node and row['Target'] == target_node:
                        # Update existing edge
                        st.session_state.edge_table.at[idx, 'Weight'] = edge_weight
                        exists = True
                        st.success(f"Updated edge: {source_node} → {target_node} with weight {edge_weight}")
                        break
                
                if not exists:
                    # Add new edge
                    new_row = pd.DataFrame({'Source': [source_node], 'Target': [target_node], 'Weight': [edge_weight]})
                    st.session_state.edge_table = pd.concat([st.session_state.edge_table, new_row], ignore_index=True)
                    st.success(f"Added edge: {source_node} → {target_node} with weight {edge_weight}")
                
                update_graph_from_edge_table()
        
        # Display edge table with edit/delete options
        if not st.session_state.edge_table.empty:
            st.subheader("Current Edges")
            
            # Show the edge table with an edit option
            edited_df = st.data_editor(
                st.session_state.edge_table,
                column_config={
                    "Source": st.column_config.SelectboxColumn(
                        "Source",
                        options=st.session_state.nodes,
                        required=True
                    ),
                    "Target": st.column_config.SelectboxColumn(
                        "Target",
                        options=st.session_state.nodes,
                        required=True
                    ),
                    "Weight": st.column_config.NumberColumn(
                        "Weight",
                        required=True
                    )
                },
                hide_index=True,
                num_rows="dynamic"
            )
            
            # Update edge table and graph if changes were made
            if not edited_df.equals(st.session_state.edge_table):
                st.session_state.edge_table = edited_df
                update_graph_from_edge_table()
                st.success("Edge table updated!")
    
    # Start and End Node Selection
    if st.session_state.nodes:
        st.subheader("Select Start and End Nodes")
        
        start_col, end_col = st.columns(2)
        
        with start_col:
            start_options = ["None"] + st.session_state.nodes
            selected_start = st.selectbox(
                "Start Node:", 
                start_options,
                index=0 if st.session_state.start_node is None else start_options.index(st.session_state.start_node),
                key="start_node_select"
            )
            
            if selected_start != "None":
                st.session_state.start_node = selected_start
            else:
                st.session_state.start_node = None
        
        with end_col:
            end_options = ["None"] + st.session_state.nodes
            selected_end = st.selectbox(
                "End Node:", 
                end_options,
                index=0 if st.session_state.end_node is None else end_options.index(st.session_state.end_node),
                key="end_node_select"
            )
            
            if selected_end != "None":
                st.session_state.end_node = selected_end
            else:
                st.session_state.end_node = None

with col2:
    st.header("Visualization & Results")
    
    # Run Bellman-Ford and display results
    if st.session_state.graph and st.session_state.start_node:
        distances, predecessors, error_message = bellman_ford(st.session_state.graph, st.session_state.start_node)
        
        if error_message:
            st.error(error_message)
            fig = draw_graph(st.session_state.graph, st.session_state.start_node, st.session_state.end_node)
        else:
            path = None
            if st.session_state.end_node:
                path = get_path(predecessors, st.session_state.start_node, st.session_state.end_node)
            
            fig = draw_graph(st.session_state.graph, st.session_state.start_node, st.session_state.end_node, path, distances)
            
            # Display algorithm results
            st.subheader("Algorithm Results")
            
            # Display distances
            st.write(f"**Shortest distances from {st.session_state.start_node}:**")
            
            # Create a table for distances
            distance_data = []
            for node, distance in distances.items():
                if distance == float('inf'):
                    distance_data.append({"Node": node, "Distance": "∞ (unreachable)"})
                else:
                    distance_data.append({"Node": node, "Distance": str(distance)})
            
            distance_df = pd.DataFrame(distance_data)
            st.dataframe(distance_df, hide_index=True)
            
            # Display path if end node is selected
            if st.session_state.end_node:
                st.write(f"**Path from {st.session_state.start_node} to {st.session_state.end_node}:**")
                
                if path:
                    st.write(" → ".join(path))
                    st.success(f"Total distance: {distances[st.session_state.end_node]}")
                else:
                    st.warning(f"No path exists from {st.session_state.start_node} to {st.session_state.end_node}")
    else:
        # Display empty graph or instruction
        if st.session_state.graph:
            fig = draw_graph(st.session_state.graph)
            st.write("Select a start node to run the Bellman-Ford algorithm.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "Create nodes and edges to start", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16)
            ax.set_axis_off()
    
    # Display the graph visualization
    st.pyplot(fig)
    
    # About the algorithm
    with st.expander("About Bellman-Ford Algorithm"):
        st.markdown("""
        ### Bellman-Ford Algorithm
        
        The Bellman-Ford algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph. Unlike Dijkstra's algorithm, it can handle graphs with negative weight edges.
        
        **Key Features:**
        - Works with directed and undirected graphs
        - Can handle negative edge weights
        - Detects negative weight cycles
        - Time complexity: O(V × E) where V is the number of vertices and E is the number of edges
        
        **How it works:**
        1. Initialize distances from source to all vertices as infinite, and source to itself as 0
        2. Relax all edges |V|−1 times, where |V| is the number of vertices
        3. Check for negative weight cycles by attempting one more relaxation - if any distance improves, there is a negative cycle
        
        **Applications:**
        - Routing protocols like RIP (Routing Information Protocol)
        - Currency exchange rate calculations
        - Finding arbitrage opportunities in trading
        - Network routing with bandwidth or delay constraints
        """)

    # Example scenarios
    with st.expander("Example Scenarios to Try"):
        st.markdown("""
        ### Try these scenarios:
        
        1. **Simple Path Finding**:
           - Create 4 nodes: A, B, C, D
           - Add edges: A→B (1), B→C (2), C→D (3), A→D (10)
           - Find path from A to D (should choose A→B→C→D with total cost 6)
        
        2. **Negative Edges (Valid)**:
           - Create 5 nodes: A, B, C, D, E
           - Add edges: A→B (2), B→C (1), C→D (2), D→B (-1), A→E (8), E→D (1)
           - Find path from A to E (should use A→B→C→D→B→C→D→B→C→D→E for lower cost)
        
        3. **Negative Cycle Detection**:
           - Create 4 nodes: A, B, C, D
           - Add edges: A→B (1), B→C (2), C→D (3), D→B (-7)
           - Run the algorithm (should detect a negative cycle)
        
        4. **Unreachable Nodes**:
           - Create 5 nodes: A, B, C, D, E
           - Add edges: A→B (1), B→C (2), D→E (1)
           - Find path from A to E (should be unreachable)
        """)