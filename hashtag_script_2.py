# %% [markdown]
# ### Initial step: From data we need to create graph of cocuring hashtags.
# 

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
# Sample DataFrame (assuming it's already loaded)
# hashtag_df = pd.read_csv("your_data.csv")  # Load your data if needed

# Count occurrences of each hashtag
hashtag_counts = hashtag_df["hashtag_names"].value_counts()

# Define a threshold (e.g., keep only hashtags that appear at least 100 times)
threshold = 2  
frequent_hashtags = hashtag_counts[hashtag_counts >= threshold]

# Filter the original DataFrame
filtered_df = hashtag_df[hashtag_df["hashtag_names"].isin(frequent_hashtags.index)]

# Display results
print(f"Original unique hashtags: {hashtag_df['hashtag_names'].nunique()}")
print(f"Filtered unique hashtags: {filtered_df['hashtag_names'].nunique()}")

# Save or use the filtered data
# filtered_df.to_csv("filtered_hashtags.csv", index=False)


# %%
# # Save as CSV
# filtered_df.to_csv("edgelist_filtered.csv", index=False)


# %% [markdown]
# ### Creating a bipartitre graph for videos-hashtags

# %%
# UNFILTERED GRAPH
# G = nx.Graph()
# #for nodes

# video_ids = set(hashtag_df['id'])
# hashtags = set(hashtag_df['hashtag_names'])

# G.add_nodes_from(video_ids, bipartite=0)  # First set (videos)
# G.add_nodes_from(hashtags, bipartite=1)   # Second set (hashtags)

# #for edges

# for _, row in hashtag_df.iterrows():
#     G.add_edge(row['id'], row['hashtag_names'])

#FILTERED GRAPH
G = nx.Graph()
#for nodes

video_ids = set(filtered_df['id'])
hashtags = set(filtered_df['hashtag_names'])

G.add_nodes_from(video_ids, bipartite=0)  # First set (videos)
G.add_nodes_from(hashtags, bipartite=1)   # Second set (hashtags)

#for edges

for _, row in filtered_df.iterrows():
    G.add_edge(row['id'], row['hashtag_names'])


# %% [markdown]
# Note to self: create a bayesian graph for the report

# %% [markdown]
# ### Uniparted network of hashtags
# 
# Two hashtags are connected if they appeared in the same video.
# 
# The edge weight represents how many times they co-occurred.

# %%

# projecting to a unipartite graph of hashtags
HG = nx.bipartite.weighted_projected_graph(G, hashtags, ratio=False)
HG


# %%
# %%
nx.write_graphml(HG, "filtered_bipartite.graphml")

# %%
import pickle

with open("filtered_bipartite.pkl", "wb") as f:
    pickle.dump(HG, f)

# %%
import csv

edges_with_weights = HG.edges(data=True)

# Open a CSV file to write
with open("edgelist_unipartite_filtered.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Source", "Target", "Weight"])  # Column headers
    
    # Write the edges and weights
    for u, v, weight in edges_with_weights:
        writer.writerow([u, v, weight['weight']])



