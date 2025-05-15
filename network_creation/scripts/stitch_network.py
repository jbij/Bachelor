# %%
import json
import pandas as pd
import requests
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import networkx as nx

# %%
duet_stitch = pd.read_csv('duet_stitch_uniques.csv')

# %%
duet_stitch.head()

# %%
videos = pd.read_json('data/video-creators.json')

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

def clean_hashtags(hashtags):
    if isinstance(hashtags, list):  
        return [tag for tag in hashtags if tag not in hashtags_to_remove]
    return hashtags 

videos['hashtag_names'] = videos['hashtag_names'].apply(clean_hashtags)

print(videos[['hashtag_names']].head())

# %%
user_videos = {}
for index, row in videos.iterrows():
    user_videos.setdefault(row['username'], []).append(row['id'])


# %%
usernames_unique = pd.unique(duet_stitch[['username', 'creator']].values.ravel()).tolist()
usernames_unique

# %%
user_hashtags = {}

for user, videos in videos.groupby('username'):  
    all_hashtags = []
    
    for _, row in videos.iterrows():  
        all_hashtags.extend(row['hashtag_names']) 
    
    user_hashtags[user] = all_hashtags


# %%
usernames_unique_set = set(usernames_unique)

filtered_user_hashtags = {user: videos for user, videos in user_hashtags.items() if user in usernames_unique_set}

# %%
user_hashtag_text = {user: " ".join(hashtags) for user, hashtags in filtered_user_hashtags.items()}
print(user_hashtag_text)

# %%
from collections import Counter

hashtag_counts = Counter(tag for tags in user_hashtag_text.values() for tag in tags.split())

frequent_hashtags = {tag for tag, count in hashtag_counts.items() if count >= 3}

filtered_user_hashtag_text = {
    user: " ".join([tag for tag in hashtags.split() if tag in frequent_hashtags])
    for user, hashtags in user_hashtag_text.items()
}

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(filtered_user_hashtag_text.values())

# %%
feature_names = vectorizer.get_feature_names_out()

user_tfidf_scores = {}

for i, user in enumerate(filtered_user_hashtag_text.keys()):
    tfidf_scores = tfidf_matrix[i].toarray().flatten()
    
    top_indices = tfidf_scores.argsort()[::-1]  
    top_hashtags = [feature_names[idx] for idx in top_indices[:10]] 
    
    user_tfidf_scores[user] = top_hashtags


user_hashtags = defaultdict(list)
for user, hashtags in user_tfidf_scores.items():
    user_hashtags[user].extend(hashtags)

user_hashtags = dict(user_hashtags)


print(f"Total number of relationships: {len(duet_stitch)}")


# %%
import pickle

with open('duet_stitch_hashtags.csv', 'wb') as f:
    pickle.dump(user_hashtags, f)

# %%
with open('duet_stitch_hashtags.csv', 'rb') as f:  
    user_hashtags = pickle.load(f)

# %%
edges = defaultdict(int)

for _, row in duet_stitch.iterrows():
    user1 = row['username']
    user2 = row['creator']

    hashtags_user1 = user_hashtags.get(user1, [])
    hashtags_user2 = user_hashtags.get(user2, [])

    for h1 in hashtags_user1:
        for h2 in hashtags_user2:
            edges[(h1, h2)] += 1

G = nx.DiGraph()
for (h1, h2), weight in edges.items():
    G.add_edge(h1, h2, weight=weight)

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

with open("ds_edgelist_unipartite.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Username", "Creator", "Weight"])  
    
    for u, v, weight in edges_with_weights:
        writer.writerow([u, v, weight['weight']])


