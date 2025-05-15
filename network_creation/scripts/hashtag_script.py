

# %%
import pandas as pd
import networkx as nx

# %%
videos = pd.read_json("../shared-folder-gald/data/video-creators.json")

# %%
videos.head()

# %%
hashtag_dict = videos.groupby('id')['hashtag_names'].apply(list).to_dict()

hashtag_set = [(video, hashtag) for video, hashtags in hashtag_dict.items() for hashtag in hashtags[0]]

# Create a DataFrame
hashtag_df = pd.DataFrame(hashtag_set, columns=['id', 'hashtag_names'])
hashtag_df


# %%
G = nx.Graph()

video_ids = set(hashtag_df['id'])
hashtags = set(hashtag_df['hashtag_names'])

G.add_nodes_from(video_ids, bipartite=0) 
G.add_nodes_from(hashtags, bipartite=1)  


for _, row in hashtag_df.iterrows():
    G.add_edge(row['id'], row['hashtag_names'])



# projecting to a unipartite graph of hashtags
HG = nx.bipartite.weighted_projected_graph(G, hashtags, ratio=False)
HG


# %%
nx.write_graphml(HG, "graph.graphml")

# %%
import pickle

with open("graph.pkl", "wb") as f:
    pickle.dump(HG, f)

# %%
import csv

edges_with_weights = HG.edges(data=True)

with open("edgelist_unipartite.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Source", "Target", "Weight"]) 
    
    for u, v, weight in edges_with_weights:
        writer.writerow([u, v, weight['weight']])


