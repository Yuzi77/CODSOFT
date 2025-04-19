import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
df = pd.read_csv('ratings.csv')
print(df.head())
pf = pd.read_csv('movies.csv')
print(pf.head())

# Create user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Compute similarity matrix (cosine similarity)
from sklearn.metrics.pairwise import cosine_similarity
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame for easier handling
user_sim_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Recommend function
def recommend_movies(user_id, num_recommendations=3):
    if user_id not in user_item_matrix.index:
        print("User ID not found.")
        return []

    # Get similar users
    sim_scores = user_sim_df[user_id].sort_values(ascending=False)
    sim_scores = sim_scores.drop(user_id)

    # Weighted sum of ratings from similar users
    weighted_ratings = np.dot(sim_scores.values, user_item_matrix.loc[sim_scores.index])
    user_rated_items = user_item_matrix.loc[user_id] > 0

    recommendations = pd.Series(weighted_ratings, index=user_item_matrix.columns)
    recommendations = recommendations[~user_rated_items]  # Filter already rated
    top_items = recommendations.sort_values(ascending=False).head(num_recommendations)

    # Merge with movie titles
    top_items_df = top_items.reset_index()
    top_items_df.columns = ['item_id', 'score']
    merged = top_items_df.merge(pf, on='item_id')

    return merged[['title', 'score']]

# Example usage
user_id = int(input("enter user id"))
print(f"\nTop recommendations for User {user_id}:")
print(recommend_movies(user_id))
