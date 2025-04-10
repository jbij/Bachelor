
# ### Creating Network of hashtags based on followers relationships 
# 

# %% [markdown]
# Steps:
# 1. directed follower network from user to user 
# 2. get videos of each user 
# exclude if user has no hashtags in their videos 
# 3. use tf-idf on hashtags from all user videos and exctract the most significant ones 
# function = hashtags of user 
# 4. create directed network of hashtags 
# 5. direction from hashtags of the follower to hashtags of the folowed 
# 

# %%
import json
import pandas as pd
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import networkx as nx

# %%
follows = pd.read_csv('../shared-folder-gald/data/follow-link.csv')

# %%
videos = pd.read_json('../shared-folder-gald/data/video-creators.json')

# %%
import pickle

with open('follow_user_hashtags.csv', 'wb') as f:
    pickle.dump(user_hashtags, f)

# %%
with open('follow_user_hashtags.csv', 'rb') as f:  # 'rb' = read binary
    user_hashtags = pickle.load(f)

# %%
list(user_hashtags.values())[:5]

# %%
# Verify the number of relationships
print(f"Total number of relationships: {len(follows)}")

# %%
follows

# %%
edges = defaultdict(int)

# Build the network
for _, row in follows.iterrows():
    user1 = row['source']
    user2 = row['target']

    hashtags_user1 = user_hashtags.get(user1, [])
    hashtags_user2 = user_hashtags.get(user2, [])

    for h1 in hashtags_user1:
        for h2 in hashtags_user2:
            edges[(h1, h2)] += 1

# Create directed graph
G = nx.DiGraph()
for (h1, h2), weight in edges.items():
    G.add_edge(h1, h2, weight=weight)

# Print sample edges
print("Some edges in the directed network with weights:")
for edge in list(G.edges(data=True))[:10]:
    print(edge)

print(f"Total number of edges: {len(G.edges())}")

# %%
nx.write_graphml(G, "follow_graph.graphml")

# %%
import pickle

with open("follow_graph.pkl", "wb") as f:
    pickle.dump(G, f)

# %%
import csv

edges_with_weights = G.edges(data=True)

# Open a CSV file to write
with open("f_edgelist_unipartite.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Source", "Target", "Weight"])  # Column headers
    
    # Write the edges and weights
    for u, v, weight in edges_with_weights:
        writer.writerow([u, v, weight['weight']])


