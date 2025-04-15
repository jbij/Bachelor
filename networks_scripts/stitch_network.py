# %%
import json
import pandas as pd
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# %%
duet_stitch = pd.read_csv('../data/duet_stitch_uniques.csv')

# %%
duet_stitch.head()

# %%
videos = pd.read_json('../../shared-folder-gald/data/video-creators.json')

# %%
videos.head()

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
    'tiktokviral'
])

# Function to remove unwanted hashtags
def clean_hashtags(hashtags):
    if isinstance(hashtags, list):  # Ensure it's a list
        return [tag for tag in hashtags if tag not in hashtags_to_remove]
    return hashtags  # Return as-is if not a list

# Apply the cleaning function
videos['hashtag_names'] = videos['hashtag_names'].apply(clean_hashtags)

# Display the cleaned dataframe
print(videos[['hashtag_names']].head())

# %%
# videos dictionary 
user_videos = {}
for index, row in videos.iterrows():
    user_videos.setdefault(row['username'], []).append(row['id'])

# %% [markdown]
# filter the videos by users that exist in the stitch/duet database 
# 

# %%
usernames_unique = pd.unique(duet_stitch[['username', 'creator']].values.ravel()).tolist()
usernames_unique

# %%
# Initialize an empty dictionary to store user -> hashtags
user_hashtags = {}

for user, videos in videos.groupby('username'):  # No need for iterrows()
    all_hashtags = []
    
    for _, row in videos.iterrows():  # Now iterating rows correctly
        all_hashtags.extend(row['hashtag_names'])  # Add hashtags from each row
    
    # Store the concatenated hashtags in the dictionary
    user_hashtags[user] = all_hashtags


# %%
# Convert usernames_unique to a set for faster lookups (O(1) instead of O(n))
usernames_unique_set = set(usernames_unique)

# Filter dictionary: Keep only users in usernames_unique
filtered_user_hashtags = {user: videos for user, videos in user_hashtags.items() if user in usernames_unique_set}

# %%
#hashtags to  text format
user_hashtag_text = {user: " ".join(hashtags) for user, hashtags in filtered_user_hashtags.items()}
print(user_hashtag_text)

# %%
from collections import Counter

# Flatten all hashtags and count occurrences
hashtag_counts = Counter(tag for tags in user_hashtag_text.values() for tag in tags.split())

# Keep hashtags appearing at least 3 times
frequent_hashtags = {tag for tag, count in hashtag_counts.items() if count >= 3}

# Filter hashtags in user_hashtag_text
filtered_user_hashtag_text = {
    user: " ".join([tag for tag in hashtags.split() if tag in frequent_hashtags])
    for user, hashtags in user_hashtag_text.items()
}


# %%
# Step 2: Compute TF-IDF
vectorizer = TfidfVectorizer(max_features=100000)
tfidf_matrix = vectorizer.fit_transform(filtered_user_hashtag_text.values())

# %%
# Get feature names (hashtags)
feature_names = vectorizer.get_feature_names_out()

# Step 3: Extract Top Hashtags Per User
user_tfidf_scores = {}

for i, user in enumerate(filtered_user_hashtag_text.keys()):
    # Get TF-IDF scores for this user
    tfidf_scores = tfidf_matrix[i].toarray().flatten()
    
    # Rank hashtags by score
    top_indices = tfidf_scores.argsort()[::-1]  # Sort in descending order
    top_hashtags = [feature_names[idx] for idx in top_indices[:]]  # Get top 5 hashtags
    
    # Store in dictionary
    user_tfidf_scores[user] = top_hashtags


# %%
# Step 4: Append to Original Dictionary
user_hashtags = defaultdict(list)
for user, hashtags in user_tfidf_scores.items():
    user_hashtags[user].extend(hashtags)

# Convert to normal dictionary (optional)
user_hashtags = dict(user_hashtags)


# %%
# Verify the number of relationships
print(f"Total number of relationships: {len(duet_stitch)}")

# %% [markdown]
# creating pickle for duet_stitch hashtags called : duet_stitch_hashtags.csv

# %%
import pickle

with open('duet_stitch_hashtags.csv', 'wb') as f:
    pickle.dump(user_hashtags, f)

# %%
with open('duet_stitch_hashtags.csv', 'rb') as f:  # 'rb' = read binary
    user_hashtags = pickle.load(f)

# %%
edges = defaultdict(int)

# %%
# Build the network
for _, row in duet_stitch.iterrows():
    user1 = row['username']
    user2 = row['creator']

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
nx.write_graphml(G, "duet_stitch_graph.graphml")

# %%
import pickle

with open("duet_stitch_graph.pkl", "wb") as f:
    pickle.dump(G, f)

# %%
import csv

edges_with_weights = G.edges(data=True)

# Open a CSV file to write
with open("ds_edgelist_unipartite.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Username", "Creator", "Weight"])  # Column headers
    
    # Write the edges and weights
    for u, v, weight in edges_with_weights:
        writer.writerow([u, v, weight['weight']])


