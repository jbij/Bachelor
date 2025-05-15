# %%
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import backboning as bb

# %%
hashtags_to_remove = set([
    'hashtag', 'fyp', 'foryou', 'foryoupage', 'fypシ', 'viral', 'love',
    'trending', 'tiktok', 'funny', 'fypage', 'capcut', 'duet', 'news',
    'foryourpage', 'fy', 'fypシ゚viral', 'follow', 'viralvideo', 'like',
    'trend', 'stitch', 'video', 'lol', 'instagram', 'asmr', 'explorepage',
    'instagood', 'viraltiktok', 'youtube', 'share', 'new', '2023', 'reels',
    'followme', 'vlog', 'satisfying', 'viralvideos', 'wow', 'funnyvideos',
    'repost', 'relatable', 'followforfollowback', 'breakingnews', 'storytime',
    'tiktokfamous', 'greenscreenvideo', 'for', 'foru', 'tiktoktrend', 'goviral',
    'bhfyp', 'viralpost', 'f', 'tiktoker', 'fypp', 'fyppppppppppppppppppppppp',
    'tiktokviral', '4upage', 'forupage', '4you', 'xyzabc', 'xyzcba', '4u', 'xyzbca', 'trendy', 'oh', 'ohno', 'relatable', 'bhfyp', 'trending', '2023', 'follow', 'explorepage', 'like', 'viral', 'tiktok', 'fybシ', 'usa_tiktok',
    'foruyou', 'trends', 'fybpage', 'trendiing', 'forupage', 'fyb', 'foryourpage', 'foryoupage', 'viralvideo', 'fyou', 'foryou', '4u', '4you', 'pageforyou', 'fyp', 'series', 'fdseite', 'fypage',
    'fyoupage', 'fds', '4upage', 'tiktokfanpage', '4youpage', 'fürdich', 'fyoupagetiktok', 'viralllllll', 'dancetrends', 'dancetrend', 'duet', 'share'
])

# %%
def backbone_pipepline(filepath, threshold, is_directed, hashtags_to_remove = hashtags_to_remove):
    with open(filepath, "rb") as f:
        G = pickle.load(f)

    G.remove_nodes_from(hashtags_to_remove)

    edge_data = [
    {'src': u, 'trg': v, 'nij': d['weight']}
    for u, v, d in G.edges(data=True)]

    df = pd.DataFrame(edge_data)

    if is_directed:
        disparity_applied = bb.disparity_filter(df, undirected = False)
    else:
        disparity_applied = bb.disparity_filter(df, undirected = True)

    thresh_applied = bb.thresholding(disparity_applied, threshold=threshold).drop(columns=["score"])

    if is_directed:
        backbone = nx.from_pandas_edgelist(
        thresh_applied,
        source="src",
        target="trg",
        edge_attr="nij",
        create_using=nx.DiGraph())
    else:
        backbone = nx.from_pandas_edgelist(
        thresh_applied,
        source="src",
        target="trg",
        edge_attr="nij",
        create_using=nx.Graph())
    
    return backbone

# %%
def gini_coefficient(x):
    """Compute Gini coefficient of array of values."""
    x = np.array(x)
    if np.amin(x) < 0:
        raise ValueError("Gini coefficient is not defined for negative values.")
    if np.amin(x) == 0:
        # Avoid division by zero
        x = x + 1e-10
    
    x_sorted = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    
    return (2 * np.sum(index * x_sorted) / (n * np.sum(x_sorted))) - (n + 1) / n


# %%
def load_and_filter_graph(filepath, is_directed, hashtags_to_remove):
    with open(filepath, "rb") as f:
        G = pickle.load(f)
    
    G.remove_nodes_from(hashtags_to_remove)

    edge_data = [
        {'src': u, 'trg': v, 'nij': d['weight']}
        for u, v, d in G.edges(data=True)
    ]

    df = pd.DataFrame(edge_data)

    disparity_applied = bb.disparity_filter(df, undirected=not is_directed)

    return disparity_applied


def build_backbone_from_disparity(disparity_df, threshold, is_directed):
    thresh_applied = bb.thresholding(disparity_df, threshold=threshold).drop(columns=["score"])

    graph_type = nx.DiGraph() if is_directed else nx.Graph()
    backbone = nx.from_pandas_edgelist(
        thresh_applied,
        source="src",
        target="trg",
        edge_attr="nij",
        create_using=graph_type
    )

    return backbone


def choose_threshold_by_gcc_size(filepath, thresholds, is_directed=False, hashtags_to_remove=hashtags_to_remove, plot=True):
    disparity_applied = load_and_filter_graph(filepath, is_directed, hashtags_to_remove)

    gcc_sizes = []
    edge_counts = []
    original_size = None

    for t in thresholds:
        backbone = build_backbone_from_disparity(disparity_applied, threshold=t, is_directed=is_directed)

        components = (
            nx.weakly_connected_components(backbone) if is_directed else nx.connected_components(backbone)
        )
        largest_cc_size = len(max(components, key=len))
        num_edges = backbone.number_of_edges()

        gcc_sizes.append((t, largest_cc_size))
        edge_counts.append((t, num_edges))

        if original_size is None:
            original_size = backbone.number_of_nodes()


    gcc_sizes = np.array(gcc_sizes)
    edge_counts = np.array(edge_counts) 

    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel("Threshold", fontsize=14)
        ax1.set_ylabel("GCC Size", color="blue", fontsize=14)
        ax1.tick_params(axis='y', labelcolor="blue", labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        l1, = ax1.plot(gcc_sizes[:, 0], gcc_sizes[:, 1], marker='o', color="blue", label="GCC Size")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Edges", color="red", fontsize=14)
        ax2.tick_params(axis='y', labelcolor="red", labelsize=12)

        l2, = ax2.plot(edge_counts[:, 0], edge_counts[:, 1], marker='s', linestyle='--', color="red", label="Edge Count")

        lines = [l1, l2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper right", fontsize=12)

        plt.title("GCC Size & Edge Count vs Threshold", fontsize=16)
        plt.grid(True)
        fig.tight_layout()
        plt.show()


    return max(gcc_sizes, key=lambda x: x[1])  # Return the threshold with the largest GCC size

import os

def choose_threshold_by_gcc_size(
    filepath,
    thresholds,
    is_directed=False,
    hashtags_to_remove=hashtags_to_remove,
    plot=True,
    plot_dir="plots",
    filename_prefix="threshold_plot"
):
    os.makedirs(plot_dir, exist_ok=True)

    disparity_applied = load_and_filter_graph(filepath, is_directed, hashtags_to_remove)

    gcc_sizes = []
    edge_counts = []
    num_components_list = []
    original_size = None

    for t in thresholds:
        backbone = build_backbone_from_disparity(disparity_applied, threshold=t, is_directed=is_directed)

        if is_directed:
            components = list(nx.weakly_connected_components(backbone))
        else:
            components = list(nx.connected_components(backbone))

        largest_cc_size = len(max(components, key=len)) if components else 0
        num_edges = backbone.number_of_edges()
        num_components = len(components)

        gcc_sizes.append((t, largest_cc_size))
        edge_counts.append((t, num_edges))
        num_components_list.append((t, num_components))

        if original_size is None:
            original_size = backbone.number_of_nodes()

    gcc_sizes = np.array(gcc_sizes)
    edge_counts = np.array(edge_counts)
    num_components_arr = np.array(num_components_list)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.set_xlabel("Threshold", fontsize=14)
        ax1.set_ylabel("GCC Size", color="blue", fontsize=14)
        ax1.tick_params(axis='y', labelcolor="blue", labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.plot(gcc_sizes[:, 0], gcc_sizes[:, 1], marker='o', color="blue", label="GCC Size")

        ax1_2 = ax1.twinx()
        ax1_2.set_ylabel("Number of Edges", color="red", fontsize=14)
        ax1_2.tick_params(axis='y', labelcolor="red", labelsize=12)
        ax1_2.plot(edge_counts[:, 0], edge_counts[:, 1], marker='s', linestyle='--', color="red", label="Edge Count")

        ax1.set_title("GCC Size & Edge Count vs Threshold", fontsize=16)

        ax2.set_xlabel("Threshold", fontsize=14)
        ax2.set_ylabel("Number of Components", color="green", fontsize=14)
        ax2.tick_params(axis='y', labelcolor="green", labelsize=12)
        ax2.tick_params(axis='x', labelsize=12)
        ax2.plot(num_components_arr[:, 0], num_components_arr[:, 1], marker='^', linestyle=':', color="green", label="Component Count")

        ax2.set_title("Number of Components vs Threshold", fontsize=16)

        plt.tight_layout()

        plot_filename = f"{filename_prefix}_{'directed' if is_directed else 'undirected'}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return max(gcc_sizes, key=lambda x: x[1])

# %%
follow_path = '../../shared-folder-gald/data/follow_graph.pkl'

# %%
thresholds_to_test = np.linspace(0.8, 0.999, 25)

gcc_size = choose_threshold_by_gcc_size(
    filepath=follow_path,
    thresholds=thresholds_to_test,
    is_directed=True,
    plot=True,
    plot_dir="plots",
    filename_prefix="follow_gcc_plot"
)
# %%
hashtag_path = '../../shared-folder-gald/data/unipartite_og.pkl'

# %%
thresholds_to_test = np.linspace(0.8, 0.999, 25)

gcc_size = choose_threshold_by_gcc_size(
    filepath=hashtag_path,
    thresholds=thresholds_to_test,
    is_directed=False,
    plot=True,
    plot_dir="plots",
    filename_prefix="hashtag_gcc_plot"
)



