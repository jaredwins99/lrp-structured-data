import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

def plot_predictions(y_pred, y_obs):

    fig, ax = plt.subplots(1,1)

    y_pred_np = y_pred.to('cpu').detach().numpy().flatten()

    ax.plot(y_pred_np, y_obs, 'o')
    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False, labelsize=6, length=0, width=0)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.grid(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False, labelsize=6, length=0, width=0)

    return im

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_correlation_graph(correlation):

    # Round for visualization
    corr_matrix = np.round(100*correlation)/100

    # Create a graph from the numpy array
    G = nx.from_numpy_array(corr_matrix - np.diag(np.ones(corr_matrix.shape[0])), create_using=nx.Graph)

    # Partition edges by weight to distinguish larger weights
    e5 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.8]
    e4 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.6 and d["weight"] < 0.8]
    e3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.4 and d["weight"] < 0.6]
    e2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.2 and d["weight"] < 0.4]
    e1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.2]
    n5 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < -0.8]
    n4 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < -0.6 and d["weight"] >= -0.8]
    n3 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < -0.4 and d["weight"] >= -0.6]
    n2 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < -0.2 and d["weight"] >= -0.4]
    n1 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= -0.2]

    # Set seed
    pos = nx.spring_layout(G, seed=75)  # positions for all nodes - seed for reproducibility

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    # Colors the edges based on the partitions
    nx.draw_networkx_edges(G, pos, edgelist=e5, width=2, edge_color="#00ad5a")
    nx.draw_networkx_edges(G, pos, edgelist=e4, width=2, alpha=0.85, edge_color="#48df6d", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=e3, width=2, alpha=0.7, edge_color="#6ef181", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=e2, width=2, alpha=0.55, edge_color="#8ee397", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=e1, width=2, alpha=0.4, edge_color="#adf5ad", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=n5, width=2, edge_color="#a72525")
    nx.draw_networkx_edges(G, pos, edgelist=n4, width=2, alpha=0.85, edge_color="#bc583d", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=n3, width=2, alpha=0.7, edge_color="#cf815c", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=n2, width=2, alpha=0.55, edge_color="#e2a981", style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=n1, width=2, alpha=0.4, edge_color="#f5d0ad", style="dashed")

    # Edge labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, alpha=.7)

    # Define thresholds for positive weights
    positive_weight_thresholds = [0.8, 0.6, 0.4, 0.2, 0]

    # Define thresholds for negative weights
    negative_weight_thresholds = [0, -0.2, -0.4, -0.6, -0.8]

    def partition_edges_by_weight(G, weight_thresholds):
        partitioned_edges = {weight: [] for weight in weight_thresholds}
        for u, v, d in G.edges(data=True):
            for threshold, edges in partitioned_edges.items():
                if d['weight'] >= threshold:
                    edges.append((u, v))
                    break  # Stop checking other thresholds
        return partitioned_edges

    # Partition edges by positive and negative weights
    positive_edges = partition_edges_by_weight(G, positive_weight_thresholds)
    negative_edges = partition_edges_by_weight(G, negative_weight_thresholds)

    # Draw edge labels with different alpha values based on thresholds
    for i, threshold in enumerate(positive_weight_thresholds[:-1]):
        edge_list = positive_edges[threshold]
        edge_labels = {edge: f"{G.edges[edge]['weight']:.2f}" for edge in edge_list}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                    alpha=1 - 0.15 * i)
        
    for i, threshold in enumerate(negative_weight_thresholds[1:]):
        edge_list = negative_edges[threshold]
        edge_labels = {edge: f"{G.edges[edge]['weight']:.2f}" for edge in edge_list}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                    alpha=1 - 0.15 * (i + 1))



    # Figure
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("graph_visual_diag_1_Wr_1")
    plt.show()