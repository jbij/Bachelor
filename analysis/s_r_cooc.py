# %%
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # Louvain algorithm
import matplotlib.pyplot as plt
import community as community_louvain
from collections import defaultdict
import pandas as pd

# %%
#From Hugging Face
# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight and powerful

# %%
# Load the network
import pickle

# Load the pickle file
with open('../ready_networks/cooc_filtered.pkl', 'rb') as f:
    G = pickle.load(f)

# Get the hashtags (nodes)
hashtags = list(G.nodes)

# %%
# Embed hashtags
embeddings = model.encode(hashtags)
print(embeddings.shape) #480 hashtags / each is represented by a 384-dimension vector

# %%
# Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)

# %%
#cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# %%
#have to plot a histogram to see the curt-off
import matplotlib.pyplot as plt

# similarities is your 480 x 480 cosine similarity matrix (flattened)
plt.hist(similarities.flatten(), bins=100)
plt.title('Distribution of Cosine Similarities')
plt.show()

# %%
# Graph
G = nx.Graph()
for i in range(len(hashtags)):
    G.add_node(hashtags[i])
    for j in range(i+1, len(hashtags)):
        if similarity_matrix[i, j] > 0.2:  # Threshold to create an edge - look at histogram
            G.add_edge(hashtags[i], hashtags[j], weight=similarity_matrix[i, j])

# %%
# Cluster with Louvain
partition = community_louvain.best_partition(G, weight='weight')

# %%
# Save graph to a pickle file
with open('S_R_cooc.pkl', 'wb') as f:
    pickle.dump(G, f)


