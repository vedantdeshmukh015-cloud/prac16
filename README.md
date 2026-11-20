# prac16

# Dataset: MovieLens 100K (Synthetic representation for standalone execution)

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Synthetic User-Item Rating Matrix
ratings = pd.DataFrame({
    'Toy Story': [5, 4, 1, 0, 0],
    'Jumanji': [1, 0, 5, 4, 2],
    'Grumpier Old Men': [4, 5, 0, 0, 1],
    'Heat': [0, 4, 5, 1, 5],
    'Sabrina': [5, 0, 0, 4, 1]
}, index=['User1', 'User2', 'User3', 'User4', 'User5'])

item_sim = cosine_similarity(ratings.T.values)
item_sim_df = pd.DataFrame(item_sim, index=ratings.columns,
                           columns=ratings.columns)

target_user = 'User1'
unrated_movies = ratings.columns[ratings.loc[target_user] == 0].tolist()
rated_movies = ratings.columns[ratings.loc[target_user] > 0].tolist()
recommendations = {}

for movie in unrated_movies:
    weights = item_sim_df.loc[movie, rated_movies]
    user_ratings = ratings.loc[target_user, rated_movies]
    if weights.sum() > 0:
        recommendations[movie] = (weights * user_ratings).sum() / weights.sum()

print(f"Item-based recommendations for {target_user}:")
for m, score in sorted(recommendations.items(), key=lambda x: x[1],
                       reverse=True):
    print(f"{m}: {score:.2f}")
