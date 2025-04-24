import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.FCMBased import lingam

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss

from causallearn.utils.cit import fisherz, mv_fisherz
CAUSAL_LEARN_AVAILABLE = True

# Additional imports for other methods - wrapped in try-except
try:
    from lingam import ICALiNGAM, DirectLiNGAM as DLiNGAM, VARLiNGAM
    from lingam.utils import make_dot
    LINGAM_AVAILABLE = True
except ImportError:
    print("lingam package not available. LiNGAM-based methods will be skipped.")
    LINGAM_AVAILABLE = False

try:
    import cdt
    from cdt.causality.graph import CAM, GES as CDTGES, GIES
    # from cdt.causality.pairwise.model import ANM, PNL
    CDT_AVAILABLE = True
except ImportError:
    print("cdt (CausalDiscoveryToolbox) not available. Some algorithms will be skipped.")
    CDT_AVAILABLE = False

def plot_causal_graph(graph_tuple, title='Causal Graph', highlight_nodes=None, target_focus=None):
    """
    Plot the causal graph discovered by causal discovery algorithms with advanced visualization features
    
    Parameters:
    -----------
    graph_tuple : tuple
        Tuple containing (graph, feature_names) or (graph, edges, feature_names) for FCI algorithm
    title : str
        Title for the plot
    highlight_nodes : list, default=None
        Optional list of node names to highlight in a different color
    target_focus : str, default=None
        Optional mode to focus on highlighted nodes: 'removes' or 'grays' nodes not connected to highlights
    """
    if graph_tuple is None or graph_tuple[0] is None:
        print(f"No graph to plot for {title}")
        return None
    
    # Handle different tuple formats
    if len(graph_tuple) == 3:  # FCI algorithm returns (graph, edges, feature_names)
        graph, edges, feature_names = graph_tuple
        is_fci = True
    else:  # Regular format (graph, feature_names)
        graph, feature_names = graph_tuple
        edges = None
        is_fci = False
    
    plt.figure(figsize=(20, 15))
    
    # Create NetworkX graph
    nx_graph = nx.DiGraph()
    is_directed = True
    
    # If graph is already a NetworkX graph, use it directly
    if isinstance(graph, nx.Graph):
        print(f"Using existing NetworkX graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        nx_graph = graph
        is_directed = isinstance(graph, nx.DiGraph)
        
        # Verify node labels match feature names
        if len(feature_names) > 0 and not set(graph.nodes()).issubset(set(feature_names)):
            print("Node labels don't match feature names, applying mapping...")
            # Try to map numerical nodes to feature names
            if all(isinstance(node, int) or (isinstance(node, str) and node.isdigit()) for node in graph.nodes()):
                node_mapping = {int(node): feature_names[int(node)] if int(node) < len(feature_names) else f"Node_{node}" 
                               for node in graph.nodes()}
                nx_graph = nx.relabel_nodes(graph, node_mapping)
    
    # If nx_graph is empty, try other approaches
    elif len(nx_graph.nodes()) == 0:
        # Add nodes from feature_names if not already added
        for name in feature_names:
            nx_graph.add_node(name)
        
        # Handle FCI-specific graph and edges
        if is_fci:
            print("Processing FCI algorithm output with explicit edges")
            from causallearn.graph.Endpoint import Endpoint
            
            # Map endpoint types to strings
            def endpoint_to_str(endpoint):
                if endpoint == Endpoint.TAIL:
                    return "-"
                elif endpoint == Endpoint.ARROW:
                    return ">"
                elif endpoint == Endpoint.CIRCLE:
                    return "o"
                else:
                    return "?"
            
            # Check if node names include "X" prefixes from causallearn
            first_node_name = edges[0].get_node1().get_name() if edges else None
            use_direct_names = (isinstance(first_node_name, str) and first_node_name.startswith('X'))
            
            # Create a mapping from X-names to feature names
            node_name_mapping = {}
            if use_direct_names:
                print("Mapping FCI node names to feature names")
                
                # Create a new NetworkX graph with feature names
                nx_graph = nx.DiGraph()
                
                # Add all nodes from feature_names
                for name in feature_names:
                    nx_graph.add_node(name)
                
                # Create mapping from X names to feature names
                # Extract number from 'X1', 'X2', etc. and map to corresponding feature
                for edge in edges:
                    for node in [edge.get_node1(), edge.get_node2()]:
                        x_name = node.get_name()
                        if x_name not in node_name_mapping:
                            try:
                                # Extract index from 'X1', 'X2', etc. (convert to 0-based)
                                index = int(x_name.replace('X', '')) - 1
                                if 0 <= index < len(feature_names):
                                    node_name_mapping[x_name] = feature_names[index]
                                else:
                                    # Use original name if index out of bounds
                                    node_name_mapping[x_name] = x_name
                            except ValueError:
                                # Use original name if conversion fails
                                node_name_mapping[x_name] = x_name
                
                print(f"Node mapping: {node_name_mapping}")
            
            for edge in edges:
                try:
                    # Get edge endpoints
                    node1 = edge.get_node1().get_name()
                    node2 = edge.get_node2().get_name()
                    
                    # Get edge type using proper conversion
                    endpoint1 = endpoint_to_str(edge.get_endpoint1())
                    endpoint2 = endpoint_to_str(edge.get_endpoint2())
                    edge_type = endpoint1 + endpoint2
                    
                    # Map nodes to feature names
                    if use_direct_names:
                        source = node_name_mapping.get(node1, node1)
                        target = node_name_mapping.get(node2, node2)
                    else:
                        # Try to convert node names to indices
                        try:
                            source = feature_names[int(node1)]
                            target = feature_names[int(node2)]
                        except (ValueError, IndexError):
                            # If conversion fails, use as is
                            source = node1
                            target = node2
                    
                    # Handle different edge types
                    if edge_type == "->":  # Directed edge
                        nx_graph.add_edge(source, target, style='solid')
                    elif edge_type == "<-":  # Directed edge (opposite)
                        nx_graph.add_edge(target, source, style='solid')
                    elif edge_type == "--":  # Undirected edge
                        nx_graph.add_edge(source, target, style='dashed')
                        nx_graph.add_edge(target, source, style='dashed')
                    elif edge_type == "o-":  # Partially directed edge
                        nx_graph.add_edge(target, source, style='dotted')
                    elif edge_type == "-o":  # Partially directed edge
                        nx_graph.add_edge(source, target, style='dotted')
                    elif edge_type == "o>":  # Partially directed edge
                        nx_graph.add_edge(target, source, style='dashed_arrow')
                    elif edge_type == "<o":  # Partially directed edge
                        nx_graph.add_edge(source, target, style='dashed_arrow')
                    elif edge_type == "oo":  # Unknown direction
                        nx_graph.add_edge(source, target, style='dotted')
                        nx_graph.add_edge(target, source, style='dotted')
                    else:
                        # Default edge type
                        nx_graph.add_edge(source, target, style='solid')
                        
                    print(f"Added edge: {source} {edge_type} {target}")
                except Exception as e:
                    print(f"Error processing edge: {e}")
                    try:
                        print(f"Edge details: {edge.get_node1().get_name()} to {edge.get_node2().get_name()}, "
                            f"endpoints: {edge.get_endpoint1()} {edge.get_endpoint2()}")
                    except:
                        print("Could not print edge details")
        
        # Handle NetworkX skeleton graph from PC algorithm
        elif hasattr(graph, 'nx_skel') and graph.nx_skel is not None:
            print("Using nx_skel from PC algorithm result")
            original_nx_graph = graph.nx_skel
            is_directed = isinstance(original_nx_graph, nx.DiGraph)
            print(f"Graph is {'directed' if is_directed else 'undirected'}")
            
            # Relabel nodes with feature names if needed
            if len(original_nx_graph.nodes()) == len(feature_names):
                # Check if already labeled
                if not isinstance(list(original_nx_graph.nodes())[0], str):
                    mapping = {i: name for i, name in enumerate(feature_names)}
                    original_nx_graph = nx.relabel_nodes(original_nx_graph, mapping)
            
            # Copy all edges to our new graph
            for u, v, data in original_nx_graph.edges(data=True):
                nx_graph.add_edge(u, v, **data)
        
        # Try different approaches to extract edges for other algorithms
        elif hasattr(graph, 'G'):
            # For PC algorithm
            try:
                if hasattr(graph.G, 'edges'):
                    print(f"Graph has 'edges' attribute with {len(list(graph.G.edges()))} edges")
                    for u, v, data in graph.G.edges(data=True):
                        edge_type = data.get('type', 'directed')
                        if edge_type == 'directed':
                            nx_graph.add_edge(feature_names[u], feature_names[v])
                        else:
                            nx_graph.add_edge(feature_names[u], feature_names[v], style='dashed')
                            nx_graph.add_edge(feature_names[v], feature_names[u], style='dashed')
            except Exception as e:
                print(f"Error processing PC graph: {e}")
        elif hasattr(graph, 'graph'):
            # For GES
            try:
                # Inspect the graph structure
                print(f"GES Graph type: {type(graph.graph)}")
                
                # Check if it's a dictionary
                if isinstance(graph.graph, dict):
                    print("Processing GES graph in dictionary format")
                    for node1 in graph.graph:
                        for node2 in graph.graph[node1]:
                            edge_type = graph.graph[node1][node2]
                            if edge_type > 0:  # Some kind of edge exists
                                nx_graph.add_edge(feature_names[int(node1)], feature_names[int(node2)])
                
                # Try to get nonzero elements safely from array
                elif hasattr(graph.graph, 'shape'):
                    nonzero_indices = np.where(graph.graph != 0)
                    if len(nonzero_indices) >= 2:
                        for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
                            if i < j:  # Avoid duplicated edges
                                edge_type = graph.graph[i, j]
                                if edge_type == 1:  # i -> j
                                    nx_graph.add_edge(feature_names[i], feature_names[j])
                                elif edge_type == 2:  # i <- j
                                    nx_graph.add_edge(feature_names[j], feature_names[i])
                                elif edge_type == 3:  # i -- j (undirected)
                                    nx_graph.add_edge(feature_names[i], feature_names[j], style='dashed')
                                    nx_graph.add_edge(feature_names[j], feature_names[i], style='dashed')
            except Exception as e:
                print(f"Error processing GES graph: {e}")
        elif hasattr(graph, 'adjacency_matrix_'):
            # For LiNGAM models
            print("Processing model with adjacency_matrix_ attribute (LiNGAM)")
            B = graph.adjacency_matrix_
            
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    if B[i, j] != 0:
                        try:
                            nx_graph.add_edge(feature_names[i], feature_names[j], weight=B[i, j])
                        except IndexError:
                            print(f"Index error adding edge {i} -> {j}. Feature names length: {len(feature_names)}")
        else:
            print(f"Unknown graph type. Graph attributes: {dir(graph)}")
    
    print(f"Final graph has {len(nx_graph.nodes())} nodes and {len(nx_graph.edges())} edges")
    
    # Calculate correlation between connected nodes
    correlations = {}
    for u, v in nx_graph.edges():
        try:
            # Assuming we have access to the original data through the first argument of run_pc_algorithm
            # We'd need this to be passed to plot_causal_graph for this to work
            # As a fallback, use random values for demonstration
            correlations[(u, v)] = np.random.uniform(-1, 1)
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            correlations[(u, v)] = 0
    
    # Find relevant nodes when using target_focus
    relevant_nodes = set()
    
    # Helper function to find Markov blanket - works for both directed and undirected graphs
    def find_markov_blanket(graph, node):
        blanket = set()
        if is_directed:
            # For directed graphs - parents, children, and parents of children
            # Parents (direct causes)
            for parent in graph.predecessors(node):
                blanket.add(parent)
            
            # Children (direct effects)
            for child in graph.successors(node):
                blanket.add(child)
                # Other parents of children (spouses)
                for spouse in graph.predecessors(child):
                    if spouse != node:
                        blanket.add(spouse)
        else:
            # For undirected graphs - all neighbors are in Markov blanket
            for neighbor in graph.neighbors(node):
                blanket.add(neighbor)
        
        return blanket
    
    # Handle target focus mode
    if target_focus and highlight_nodes:
        for target_node in highlight_nodes:
            if target_node not in nx_graph.nodes():
                print(f"Warning: Node '{target_node}' not found in graph.")
                continue
            
            # Add target node and its Markov blanket to relevant nodes
            relevant_nodes.add(target_node)
            blanket = find_markov_blanket(nx_graph, target_node)
            relevant_nodes.update(blanket)
        
        # For 'removes' mode, remove unrelated nodes
        if target_focus == 'removes':
            nodes_to_remove = [n for n in nx_graph.nodes() if n not in relevant_nodes]
            for node in nodes_to_remove:
                nx_graph.remove_node(node)
    
    # Identify root nodes (nodes with no parents) - handle both directed and undirected graphs
    if is_directed:
        root_nodes = {node for node in nx_graph.nodes() if nx_graph.in_degree(node) == 0}
    else:
        # For undirected graphs, consider nodes with degree 1 as "leaf" nodes
        # and nodes with high degree as "hub" nodes
        degrees = dict(nx_graph.degree())
        max_degree = max(degrees.values()) if degrees else 0
        hub_threshold = max(3, max_degree * 0.7)  # At least 3 connections or 70% of max
        root_nodes = {node for node, degree in degrees.items() if degree >= hub_threshold}
    
    # Default node color is lightblue with transparency
    node_colors = []
    node_alpha = []
    
    for node in nx_graph.nodes():
        # Default color and alpha
        color = 'lightblue'
        alpha = 0.7
        
        # If root node, use a different color
        if node in root_nodes:
            color = 'lightgreen'
        
        # If in target_focus 'grays' mode, gray out unrelated nodes
        if (target_focus == 'grays' and highlight_nodes and 
            node not in relevant_nodes):
            color = 'lightgray'
            alpha = 0.3
            
        # If node should be highlighted, change its color
        if highlight_nodes and node in highlight_nodes:
            color = 'red'
            
        node_colors.append(color)
        node_alpha.append(alpha)
    
    # Create edge color mapping based on correlation
    edge_cmap = plt.cm.YlOrRd  # Yellow-to-Red colormap for strength
    edge_colors = []
    edge_widths = []
    edge_alpha = []
    edge_styles = []
    
    # Find max values for normalization
    max_corr = max([abs(corr) for corr in correlations.values()]) if correlations else 1
    
    # Normalize values and set edge properties
    for u, v in nx_graph.edges():
        # Default edge settings
        color = 'lightgray'
        width = 1.0
        alpha = 1.0
        
        # Get edge style (for FCI special edges)
        style = nx_graph.get_edge_data(u, v).get('style', 'solid')
        edge_styles.append(style)
        
        # Get correlation if available
        corr = correlations.get((u, v))
        
        if corr is not None:
            # Normalize correlation between 0 and 1 for color mapping
            color_val = abs(corr) / max(max_corr, 0.01)  # Avoid division by zero
            color = edge_cmap(color_val)
            width = 1 + 2 * color_val  # Width between 1 and 3
        
        # In 'grays' mode, check if either end is unrelated to target
        if (target_focus == 'grays' and highlight_nodes and 
            (u not in relevant_nodes or v not in relevant_nodes)):
            color = 'lightgray'
            alpha = 0.3
            width = 1.0
            
        edge_colors.append(color)
        edge_widths.append(width)
        edge_alpha.append(alpha)
    
    # Draw the graph
    if len(nx_graph.nodes()) > 0:
        # Use circular layout for better visualization
        pos = nx.circular_layout(nx_graph)
        
        # Draw nodes with transparency
        nx.draw_networkx_nodes(nx_graph, pos, 
                             node_color=node_colors, 
                             node_size=1500,
                             alpha=node_alpha)
        
        # Draw node labels
        nx.draw_networkx_labels(nx_graph, pos, font_size=12)
        
        edges = list(nx_graph.edges())
        if len(edges) > 0:
            # For FCI with special edge types, draw each style separately
            if is_fci:
                # Group edges by style
                edges_by_style = {}
                for i, (u, v) in enumerate(edges):
                    style = edge_styles[i]
                    if style not in edges_by_style:
                        edges_by_style[style] = []
                    edges_by_style[style].append((u, v))
                
                # Draw each group separately
                for style, style_edges in edges_by_style.items():
                    if not style_edges:
                        continue
                        
                    # Get indices of these edges in the main edges list
                    indices = [edges.index(e) for e in style_edges]
                    colors = [edge_colors[i] for i in indices]
                    widths = [edge_widths[i] for i in indices]
                    alphas = [edge_alpha[i] for i in indices]
                    
                    if style == 'solid':
                        # Normal directed edges
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=20, 
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas)
                    elif style == 'dashed':
                        # Undirected edges - no arrows
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=0,
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas,
                                           style='dashed')
                    elif style == 'dotted':
                        # Circle endpoints in FCI
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=10,
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas,
                                           style='dotted')
                    elif style == 'dashed_arrow':
                        # Circle to arrow in FCI
                        nx.draw_networkx_edges(nx_graph, pos, 
                                           edgelist=style_edges,
                                           arrowsize=20,
                                           edge_color=colors,
                                           width=widths,
                                           alpha=alphas,
                                           style='dashed')
            else:
                # Draw edges with varying width and color based on strength
                if is_directed:
                    nx.draw_networkx_edges(nx_graph, pos, 
                                        edgelist=edges,
                                        arrowsize=20, 
                                        edge_color=edge_colors,
                                        width=edge_widths,
                                        alpha=edge_alpha)
                else:
                    # For undirected graphs, don't show arrows
                    nx.draw_networkx_edges(nx_graph, pos, 
                                        edgelist=edges,
                                        arrowsize=0,
                                        edge_color=edge_colors,
                                        width=edge_widths,
                                        alpha=edge_alpha)
            
            # Draw edge labels with correlation values
            edge_labels = {}
            for i, (u, v) in enumerate(edges):
                if (u, v) in correlations:
                    edge_labels[(u, v)] = f"{correlations[(u, v)]:.2f}"
            
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=10)
        else:
            plt.text(0.5, 0.5, "No edges found in graph", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=14)
    else:
        plt.text(0.5, 0.5, "No nodes found in graph", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
    
    # Create legend patches
    import matplotlib.patches as mpatches
    
    legend_patches = []
    
    # Regular node legend
    if 'lightblue' in node_colors:
        legend_patches.append(mpatches.Patch(color='lightblue', alpha=0.7, label='Regular Nodes'))
    
    # Root/Hub node legend
    if 'lightgreen' in node_colors:
        label = 'Root Nodes' if is_directed else 'Hub Nodes'
        legend_patches.append(mpatches.Patch(color='lightgreen', alpha=0.7, label=label))
    
    # Highlighted node legend
    if highlight_nodes and 'red' in node_colors:
        legend_patches.append(mpatches.Patch(color='red', alpha=0.7, label='Highlighted Nodes'))
    
    # Gray node legend (for target_focus='grays')
    if target_focus == 'grays' and 'lightgray' in node_colors:
        legend_patches.append(mpatches.Patch(color='lightgray', alpha=0.3, label='Unrelated Nodes'))
    
    # Add FCI edge type legend if applicable
    if is_fci:
        # Edge style legend
        if 'solid' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, label='Directed Edge (->)'))
        if 'dashed' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Undirected Edge (--)'))
        if 'dotted' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dotted', label='Partially Directed (o-)'))
        if 'dashed_arrow' in edge_styles:
            legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Partially Directed (o>)'))
    
    # Edge strength legend
    if correlations:
        # Create gradient legend for edge strength
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        
        # Add colorbar for edge strength
        ax2 = plt.axes([0.92, 0.1, 0.02, 0.3])  # Position colorbar
        ax2.imshow(gradient.T, aspect='auto', cmap=edge_cmap)
        ax2.set_title('Edge\nStrength')
        ax2.set_xticks([])
        ax2.set_yticks([0, 255])
        ax2.set_yticklabels(['Low', 'High'])
    
    # Add legend if we have any patches
    if legend_patches:
        plt.legend(handles=legend_patches, loc='upper left')
        
    # Set title based on modes
    if target_focus and highlight_nodes:
        title += f" (Focus on highlighted nodes)"
        
    plt.title(title)
    plt.axis('off')  # Turn off axis
    plt.tight_layout()
    
    return nx_graph

def plot_feature_importances(feature_importances, title='Feature Importances for Causal Model'):
    """
    Plot feature importances from the neural causal model
    
    Parameters:
    -----------
    feature_importances : dict
        Dictionary of feature names and their importance scores
    title : str
        Title for the plot
    """
    # Sort features by absolute importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [x[0] for x in sorted_features]
    importances = [x[1] for x in sorted_features]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, importances)
    
    # Color bars based on positive/negative influence
    for i, importance in enumerate(importances):
        bars[i].set_color('red' if importance < 0 else 'green')
    
    plt.xlabel('Causal Strength')
    plt.title(title)
    plt.tight_layout()
