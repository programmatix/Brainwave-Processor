
import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, PC
# from pgmpy.estimators import BicScore
# from pgmpy.estimators.scores import BicScore
from pgmpy.estimators import BDeu, K2, BIC
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pomegranate.bayesian_network import BayesianNetwork



def bayesian_network(df, scoring_method='bic', highlight_nodes=None, target_focus=None):

    """
    Create and visualize a Bayesian network from data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to learn the Bayesian network from
    scoring_method : str, default='bic'
        The scoring method to use ('bic', 'k2', or 'bdeu')
    highlight_nodes : list, default=None
        Optional list of node names to highlight in a different color
    target_focus : str, default=None
        Optional mode to focus on highlighted nodes: 'removes' or 'grays' nodes not connected to highlights
        
    Returns:
    --------
    tuple:
        - model: The learned Bayesian network model
        - connections_df: DataFrame with details about all connections found
    """
    # Option 1: Score-based learning with Hill Climbing

    # df = df.copy()
    # df = df.replace(0, np.nan)

    hc = HillClimbSearch(df)
    if scoring_method == 'bic':
        score = BIC(df)
    elif scoring_method == 'k2':
        score = K2(df)
    elif scoring_method == 'bdeu':
        score = BDeu(df)

    # set any 0 values to NaN

    # constraints = [('shower:last', 'LEP:datetime')]

    best_model = hc.estimate(scoring_method=score)

    # pc = PC(df)
    # best_model = pc.estimate(significance_level=0.05)

    
    # Identify root nodes (nodes with no parents)
    root_nodes = {node for node in best_model.nodes() if not list(best_model.predecessors(node))}
    
    # Calculate prior distributions for root nodes
    prior_distributions = {}
    for node in root_nodes:
        if node in df.columns:
            # For categorical variables
            if df[node].dtype == 'object' or df[node].dtype == 'category':
                prior_distributions[node] = df[node].value_counts(normalize=True).to_dict()
            # For numerical variables, compute basic statistics
            else:
                prior_distributions[node] = {
                    'mean': df[node].mean(),
                    'median': df[node].median(),
                    'std': df[node].std(),
                    'min': df[node].min(),
                    'max': df[node].max()
                }
    
    # Create a DataFrame of all connections
    connections = []
    
    for parent, child in best_model.edges():
        # Calculate edge strength metrics
        correlation = None
        mutual_info = None
        association_count = None
        counter_example = None
        
        # For numerical data, calculate correlation
        if (parent in df.columns and child in df.columns and
            df[parent].dtype in ['float64', 'int64'] and 
            df[child].dtype in ['float64', 'int64']):
            
            correlation = df[parent].corr(df[child])
            
            # Find counter-example for strong correlations (above 0.9)
            if abs(correlation) > 0.9:
                # Calculate z-scores for both variables
                parent_z = (df[parent] - df[parent].mean()) / df[parent].std()
                child_z = (df[child] - df[child].mean()) / df[child].std()
                
                # Look for cases where the signs differ (negative correlation)
                # or where one is high but the other isn't (positive correlation)
                if correlation > 0:
                    # For positive correlation, find cases where one is high but other isn't
                    diff = abs(parent_z - child_z)
                    counter_idx = diff.nlargest(1).index[0] if not diff.empty else None
                else:
                    # For negative correlation, find cases where they go in same direction
                    product = parent_z * child_z
                    counter_idx = product.nlargest(1).index[0] if not product.empty else None
                
                if counter_idx is not None:
                    counter_example = {
                        'index': counter_idx,
                        parent: df.loc[counter_idx, parent],
                        child: df.loc[counter_idx, child]
                    }
            
        # For categorical data, calculate association counts
        elif parent in df.columns and child in df.columns:
            counts = df.groupby([parent, child]).size().reset_index(name='count')
            association_count = counts['count'].sum()
        
        # Add to connections list
        connections.append({
            'from': parent,
            'to': child,
            'correlation': correlation,
            'association_count': association_count,
            'counter_example': counter_example
        })
    
    # Create DataFrame of connections
    connections_df = pd.DataFrame(connections)
    
    # Visualize the resulting network
    def plot_network(model, title):
        G = nx.DiGraph()
        G.add_edges_from(model.edges())
        
        # Find relevant nodes when using target_focus
        relevant_nodes = set()
        
        # Handle target focus mode
        if target_focus and highlight_nodes:
            for target_node in highlight_nodes:
                if target_node not in G.nodes():
                    print(f"Warning: Node '{target_node}' not found in graph.")
                    continue
                
                # Helper function to find all ancestors recursively
                def find_ancestors(node, visited=None):
                    if visited is None:
                        visited = set()
                    visited.add(node)
                    relevant_nodes.add(node)  # Add to relevant nodes
                    for pred in G.predecessors(node):
                        relevant_nodes.add(pred)  # Add predecessor
                        if pred not in visited:
                            find_ancestors(pred, visited)
                
                # Add target node and find its ancestors
                find_ancestors(target_node)
            
            # For 'removes' mode, remove unrelated nodes
            if target_focus == 'removes':
                nodes_to_remove = [n for n in G.nodes() if n not in relevant_nodes]
                for node in nodes_to_remove:
                    G.remove_node(node)
        
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(G)
        
        # Default node color is lightblue with transparency
        node_colors = []
        node_alpha = []
        
        for node in G.nodes():
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
        
        # Draw nodes with transparency
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=1500,
                              alpha=node_alpha)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Create edge color mapping based on correlation or association strength
        edge_cmap = plt.cm.YlOrRd  # Yellow-to-Red colormap for strength
        edge_colors = []
        edge_widths = []
        edge_alpha = []
        
        # Find max values for normalization
        max_corr = 0
        max_assoc = 0
        
        for i, (u, v) in enumerate(G.edges()):
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                if corr is not None and not pd.isna(corr) and abs(corr) > max_corr:
                    max_corr = abs(corr)
                    
                if assoc is not None and not pd.isna(assoc) and assoc > max_assoc:
                    max_assoc = assoc
        
        # Normalize values and set edge properties
        for i, (u, v) in enumerate(G.edges()):
            # Default edge settings
            color = 'lightgray'
            width = 1.0
            alpha = 1.0
            
            # Get edge data from connections DataFrame
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                # Use correlation or association count for color intensity
                if corr is not None and not pd.isna(corr):
                    # Normalize correlation between 0 and 1 for color mapping
                    color_val = abs(corr) / max(max_corr, 0.01)  # Avoid division by zero
                    color = edge_cmap(color_val)
                    width = 1 + 2 * color_val  # Width between 1 and 3
                elif assoc is not None and not pd.isna(assoc):
                    # Normalize association count
                    color_val = assoc / max(max_assoc, 1)  # Avoid division by zero
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
        
        # Draw edges with varying width and color based on strength
        nx.draw_networkx_edges(G, pos, 
                             arrowsize=20, 
                             edge_color=edge_colors,
                             width=edge_widths,
                             alpha=edge_alpha)
        
        # Calculate edge weights based on mutual information or correlation
        edge_labels = {}
        for u, v in G.edges():
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                if corr is not None and not pd.isna(corr):
                    edge_labels[(u, v)] = f"{corr:.2f}"
                elif assoc is not None and not pd.isna(assoc):
                    edge_labels[(u, v)] = f"{assoc}"
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Create legend patches
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        
        legend_patches = []
        
        # Regular node legend
        if 'lightblue' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightblue', alpha=0.7, label='Regular Nodes'))
        
        # Root node legend
        if 'lightgreen' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightgreen', alpha=0.7, label='Root Nodes'))
        
        # Highlighted node legend
        if highlight_nodes and 'red' in node_colors:
            legend_patches.append(mpatches.Patch(color='red', alpha=0.7, label='Highlighted Nodes'))
        
        # Gray node legend (for target_focus='grays')
        if target_focus == 'grays' and 'lightgray' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightgray', alpha=0.3, label='Unrelated Nodes'))
        
        # Edge strength legend
        if max_corr > 0 or max_assoc > 0:
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
        plt.show()

    plot_network(best_model, "Bayesian Network from Hill Climbing with " + scoring_method)

    # Analyze direct and indirect relationships
    lep_time_deps = [parent for parent, child in best_model.edges() if child == 'LEP_time']
    print("Direct dependencies of LEP_time:", lep_time_deps)

    # Find Markov Blanket of LEP_time
    def find_markov_blanket(model, node):
        blanket = set()
        for parent, child in model.edges():
            if child == node:
                blanket.add(parent)
            elif parent == node:
                blanket.add(child)
                # Add other parents of this child
                for p, c in model.edges():
                    if c == child and p != node:
                        blanket.add(p)
        return blanket

    # Show Markov blanket for each highlighted node
    if highlight_nodes:
        for node in highlight_nodes:
            if node in best_model.nodes():
                print(f"Markov Blanket of {node}:", find_markov_blanket(best_model, node))
    
    # Add Markov blanket information to the connections DataFrame
    if not connections_df.empty:
        # Add a column for highlight status
        connections_df['is_highlight_source'] = connections_df['from'].isin(highlight_nodes) if highlight_nodes else False
        connections_df['is_highlight_target'] = connections_df['to'].isin(highlight_nodes) if highlight_nodes else False
        
        # For each highlighted node, add columns indicating if the edge is in its Markov blanket
        if highlight_nodes:
            for node in highlight_nodes:
                if node in best_model.nodes():
                    blanket = find_markov_blanket(best_model, node)
                    connections_df[f'in_{node}_markov_blanket'] = connections_df.apply(
                        lambda row: row['from'] in blanket or row['to'] in blanket, 
                        axis=1
                    )
    
    # Add information about prior distributions for root nodes
    root_node_info = pd.DataFrame([
        {'node': node, 'prior_distribution': str(prior_distributions.get(node, {}))}
        for node in root_nodes
    ])
    
    return best_model, connections_df, root_node_info    


def bayesian_network_pomegranate(df, scoring_method='bic', highlight_nodes=None, target_focus=None):
    """
    Create and visualize a Bayesian network from data using pomegranate library.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to learn the Bayesian network from
    scoring_method : str, default='bic'
        The scoring method to use ('bic', 'k2', or 'bdeu') 
    highlight_nodes : list, default=None
        Optional list of node names to highlight in a different color
    target_focus : str, default=None
        Optional mode to focus on highlighted nodes: 'removes' or 'grays' nodes not connected to highlights
        
    Returns:
    --------
    tuple:
        - model: The learned Bayesian network model
        - connections_df: DataFrame with details about all connections found
        - root_node_info: DataFrame with prior distribution information for root nodes
    """
    
    # Convert categorical columns to numerical if needed
    df_processed = df.copy()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        df_processed[col] = df_processed[col].astype('category').cat.codes
    
    # Define scoring method for pomegranate
    if scoring_method == 'bic':
        pom_scoring = 'bic'
    elif scoring_method == 'k2':
        pom_scoring = 'k2'
    elif scoring_method == 'bdeu':
        pom_scoring = 'bdeu'
    else:
        pom_scoring = 'bic'  # Default
    
    # Learn Bayesian network structure using pomegranate
    model = BayesianNetwork(
        # df_processed.values, 
        # algorithm='greedy', 
        # scoring=pom_scoring,
        # column_names=list(df.columns)
    )
    model.fit(df_processed.to_numpy())
    return model
    
    # Get structure as networkx graph
    G = nx.DiGraph()
    
    # Extract edges from pomegranate model
    for i, parents in enumerate(model.structure):
        node = df.columns[i]
        for parent in parents:
            parent_node = df.columns[parent]
            G.add_edge(parent_node, node)
    
    # Identify root nodes (nodes with no parents)
    root_nodes = {node for node in G.nodes() if G.in_degree(node) == 0}
    
    # Calculate prior distributions for root nodes
    prior_distributions = {}
    for node in root_nodes:
        if node in df.columns:
            # For categorical variables
            if df[node].dtype == 'object' or df[node].dtype == 'category':
                prior_distributions[node] = df[node].value_counts(normalize=True).to_dict()
            # For numerical variables, compute basic statistics
            else:
                prior_distributions[node] = {
                    'mean': df[node].mean(),
                    'median': df[node].median(),
                    'std': df[node].std(),
                    'min': df[node].min(),
                    'max': df[node].max()
                }
    
    # Create a DataFrame of all connections
    connections = []
    
    for parent, child in G.edges():
        # Calculate edge strength metrics
        correlation = None
        association_count = None
        counter_example = None
        
        # For numerical data, calculate correlation
        if (parent in df.columns and child in df.columns and
            df[parent].dtype in ['float64', 'int64'] and 
            df[child].dtype in ['float64', 'int64']):
            
            correlation = df[parent].corr(df[child])
            
            # Find counter-example for strong correlations (above 0.9)
            if abs(correlation) > 0.9:
                # Calculate z-scores for both variables
                parent_z = (df[parent] - df[parent].mean()) / df[parent].std()
                child_z = (df[child] - df[child].mean()) / df[child].std()
                
                # Look for cases where the signs differ (negative correlation)
                # or where one is high but the other isn't (positive correlation)
                if correlation > 0:
                    # For positive correlation, find cases where one is high but other isn't
                    diff = abs(parent_z - child_z)
                    counter_idx = diff.nlargest(1).index[0] if not diff.empty else None
                else:
                    # For negative correlation, find cases where they go in same direction
                    product = parent_z * child_z
                    counter_idx = product.nlargest(1).index[0] if not product.empty else None
                
                if counter_idx is not None:
                    counter_example = {
                        'index': counter_idx,
                        parent: df.loc[counter_idx, parent],
                        child: df.loc[counter_idx, child]
                    }
            
        # For categorical data, calculate association counts
        elif parent in df.columns and child in df.columns:
            counts = df.groupby([parent, child]).size().reset_index(name='count')
            association_count = counts['count'].sum()
        
        # Add to connections list
        connections.append({
            'from': parent,
            'to': child,
            'correlation': correlation,
            'association_count': association_count,
            'counter_example': counter_example
        })
    
    # Create DataFrame of connections
    connections_df = pd.DataFrame(connections)
    
    # Visualize the resulting network
    def plot_network(graph, title):
        # Find relevant nodes when using target_focus
        relevant_nodes = set()
        
        # Handle target focus mode
        if target_focus and highlight_nodes:
            for target_node in highlight_nodes:
                if target_node not in graph.nodes():
                    print(f"Warning: Node '{target_node}' not found in graph.")
                    continue
                
                # Helper function to find all ancestors recursively
                def find_ancestors(node, visited=None):
                    if visited is None:
                        visited = set()
                    visited.add(node)
                    relevant_nodes.add(node)  # Add to relevant nodes
                    for pred in graph.predecessors(node):
                        relevant_nodes.add(pred)  # Add predecessor
                        if pred not in visited:
                            find_ancestors(pred, visited)
                
                # Add target node and find its ancestors
                find_ancestors(target_node)
            
            # For 'removes' mode, remove unrelated nodes
            if target_focus == 'removes':
                nodes_to_remove = [n for n in graph.nodes() if n not in relevant_nodes]
                for node in nodes_to_remove:
                    graph.remove_node(node)
        
        plt.figure(figsize=(20, 15))
        pos = nx.spring_layout(graph)
        
        # Default node color is lightblue with transparency
        node_colors = []
        node_alpha = []
        
        for node in graph.nodes():
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
        
        # Draw nodes with transparency
        nx.draw_networkx_nodes(graph, pos, 
                              node_color=node_colors, 
                              node_size=1500,
                              alpha=node_alpha)
        
        # Draw node labels
        nx.draw_networkx_labels(graph, pos, font_size=12)
        
        # Create edge color mapping based on correlation or association strength
        edge_cmap = plt.cm.YlOrRd  # Yellow-to-Red colormap for strength
        edge_colors = []
        edge_widths = []
        edge_alpha = []
        
        # Find max values for normalization
        max_corr = 0
        max_assoc = 0
        
        for i, (u, v) in enumerate(graph.edges()):
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                if corr is not None and not pd.isna(corr) and abs(corr) > max_corr:
                    max_corr = abs(corr)
                    
                if assoc is not None and not pd.isna(assoc) and assoc > max_assoc:
                    max_assoc = assoc
        
        # Normalize values and set edge properties
        for i, (u, v) in enumerate(graph.edges()):
            # Default edge settings
            color = 'lightgray'
            width = 1.0
            alpha = 1.0
            
            # Get edge data from connections DataFrame
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                # Use correlation or association count for color intensity
                if corr is not None and not pd.isna(corr):
                    # Normalize correlation between 0 and 1 for color mapping
                    color_val = abs(corr) / max(max_corr, 0.01)  # Avoid division by zero
                    color = edge_cmap(color_val)
                    width = 1 + 2 * color_val  # Width between 1 and 3
                elif assoc is not None and not pd.isna(assoc):
                    # Normalize association count
                    color_val = assoc / max(max_assoc, 1)  # Avoid division by zero
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
        
        # Draw edges with varying width and color based on strength
        nx.draw_networkx_edges(graph, pos, 
                             arrowsize=20, 
                             edge_color=edge_colors,
                             width=edge_widths,
                             alpha=edge_alpha)
        
        # Calculate edge weights based on mutual information or correlation
        edge_labels = {}
        for u, v in graph.edges():
            edge_data = connections_df[(connections_df['from'] == u) & (connections_df['to'] == v)]
            if not edge_data.empty:
                corr = edge_data['correlation'].iloc[0]
                assoc = edge_data['association_count'].iloc[0]
                
                if corr is not None and not pd.isna(corr):
                    edge_labels[(u, v)] = f"{corr:.2f}"
                elif assoc is not None and not pd.isna(assoc):
                    edge_labels[(u, v)] = f"{assoc}"
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
        
        # Create legend patches
        import matplotlib.patches as mpatches
        
        legend_patches = []
        
        # Regular node legend
        if 'lightblue' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightblue', alpha=0.7, label='Regular Nodes'))
        
        # Root node legend
        if 'lightgreen' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightgreen', alpha=0.7, label='Root Nodes'))
        
        # Highlighted node legend
        if highlight_nodes and 'red' in node_colors:
            legend_patches.append(mpatches.Patch(color='red', alpha=0.7, label='Highlighted Nodes'))
        
        # Gray node legend (for target_focus='grays')
        if target_focus == 'grays' and 'lightgray' in node_colors:
            legend_patches.append(mpatches.Patch(color='lightgray', alpha=0.3, label='Unrelated Nodes'))
        
        # Edge strength legend
        if max_corr > 0 or max_assoc > 0:
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
        plt.show()

    plot_network(G, f"Bayesian Network with Pomegranate using {scoring_method}")
    
    # Find Markov Blanket
    def find_markov_blanket(graph, node):
        blanket = set()
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
        
        return blanket
    
    # Show Markov blanket for each highlighted node
    if highlight_nodes:
        for node in highlight_nodes:
            if node in G.nodes():
                print(f"Markov Blanket of {node}:", find_markov_blanket(G, node))
    
    # Add Markov blanket information to the connections DataFrame
    if not connections_df.empty:
        # Add a column for highlight status
        connections_df['is_highlight_source'] = connections_df['from'].isin(highlight_nodes) if highlight_nodes else False
        connections_df['is_highlight_target'] = connections_df['to'].isin(highlight_nodes) if highlight_nodes else False
        
        # For each highlighted node, add columns indicating if the edge is in its Markov blanket
        if highlight_nodes:
            for node in highlight_nodes:
                if node in G.nodes():
                    blanket = find_markov_blanket(G, node)
                    connections_df[f'in_{node}_markov_blanket'] = connections_df.apply(
                        lambda row: row['from'] in blanket or row['to'] in blanket, 
                        axis=1
                    )
    
    # Add information about prior distributions for root nodes
    root_node_info = pd.DataFrame([
        {'node': node, 'prior_distribution': str(prior_distributions.get(node, {}))}
        for node in root_nodes
    ])
    
    # Find direct dependencies of specific nodes (similar to LEP_time in original function)
    if 'LEP_time' in df.columns and 'LEP_time' in G.nodes():
        lep_time_deps = list(G.predecessors('LEP_time'))
        print("Direct dependencies of LEP_time:", lep_time_deps)
    
    return G, connections_df, root_node_info