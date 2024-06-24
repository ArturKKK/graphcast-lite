import networkx as nx
import numpy as np
import plotly.graph_objs as go


def visualize_mesh_edges_3d(edges):
    # Create a graph
    G = nx.Graph()

    # Add edges to the graph
    for i in range(edges.shape[1]):
        G.add_edge(edges[0, i], edges[1, i])

    # Create 3D positions for the nodes using a spring layout
    pos = nx.spring_layout(G, dim=3)
    xyz = np.array([pos[v] for v in sorted(G)])

    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        trace = go.Scatter3d(
            x=[x0, x1, None], 
            y=[y0, y1, None], 
            z=[z0, z1, None],
            mode='lines',
            line=dict(color='gray', width=2),
        )
        edge_trace.append(trace)

    # Create node trace
    node_trace = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers+text',
        marker=dict(symbol='circle', size=5, color='lightblue'),  # Smaller size for better visibility
        text=[str(i) for i in range(len(pos))],
        textposition='top center'
    )

    # Create the layout with axis ticks removed
    layout = go.Layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        )
    )

    # Create the figure
    fig = go.Figure(data=edge_trace + [node_trace], layout=layout)

    # Show the figure
    fig.show()

if __name__ == '__main__':
    # Example usage
    edges = np.array([[ 0,  1,  0,  2,  0,  4,  0,  6,  0,  8,  1,  2,  1,  5,  1,  6,  1,  7,
                    2,  3,  2,  7,  2,  8,  3,  7,  3,  8,  3,  9,  3, 10,  4,  6,  4,  8,
                    4, 10,  4, 11,  5,  6,  5,  7,  5,  9,  5, 11,  6, 11,  7,  9,  8, 10,
                    9, 10,  9, 11, 10, 11],
                  [ 1,  0,  2,  0,  4,  0,  6,  0,  8,  0,  2,  1,  5,  1,  6,  1,  7,  1,
                    3,  2,  7,  2,  8,  2,  7,  3,  8,  3,  9,  3, 10,  3,  6,  4,  8,  4,
                   10,  4, 11,  4,  6,  5,  7,  5,  9,  5, 11,  5, 11,  6,  9,  7, 10,  8,
                   10,  9, 11,  9, 11, 10]])

    visualize_mesh_edges_3d(edges)
    print("here")