# %% [markdown]
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

# %%
follows = pd.read_csv('../shared-folder-gald/data/follow-link.csv')

# %%
videos = pd.read_json('../shared-folder-gald/data/video-creators.json')

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
user_videos = {}
for index, row in videos.iterrows():
    user_videos.setdefault(row['username'], []).append(row['id'])

# %% [markdown]
# filter the videos by users that exist in the follower database 
# 

# %%
usernames_unique = pd.unique(follows[['source', 'target']].values.ravel()).tolist()
usernames_unique

# %% [markdown]
# Get dictionary of hahtags per user

# %% [markdown]
# Using TF-IDF (Term Frequency-Inverse Document Frequency) to determine the most important hashtags for each user based on their videos.

# %% [markdown]
# Plan:
# 1. Convert user_videos into a format suitable for TfidfVectorizer.
# 2. Compute TF-IDF scores for hashtags within each user’s videos.
# Compute TF-IDF: Treat each user as a "document" and their hashtags as "terms".
# 3. Select top hashtags per user based on importance.
# 4. Append those hashtags to your dictionary.

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
#number of creators is smaller after filtering by the follow list
print(len(list(filtered_user_hashtags.keys())))

# %%
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

# Output Example
print(user_hashtags)

# %%
import pickle

with open('user_hashtags.csv', 'wb') as f:
    pickle.dump(user_hashtags, f)


