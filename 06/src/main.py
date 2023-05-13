import os
import pickle

import pandas as pd
import numpy as np
import sys
import warnings


TOP_N = 5
OUT_FILE = '../results/results.txt'


def printR(string):
    sys.stdout.write('\r' + string)
    sys.stdout.flush()


def appendToFile(string):
    print(string)
    with open(OUT_FILE, 'a') as f:
        print(string, file=f)


def extract_all_genres(movies):
    result = []
    genres_column = movies['genres'].values
    for genres in genres_column:
        for g in genres.split("|"):
            if g not in result:
                result.append(g)
    result.sort()
    return result[1:]  # remove (no genres listed)


def get_genre_vector(movies: pd.DataFrame, genres: list) -> pd.DataFrame:
    for genre in genres:
        movies[genre] = 0
    for i, row in movies.iterrows():
        for genre in genres:
            if genre in row['genres'].split("|"):
                movies.at[i, genre] = 1
    movies.pop('genres')
    return movies


def get_user_profiles(ratings: pd.DataFrame, movies: pd.DataFrame, genres: list):
    user_profiles = pd.DataFrame(columns=['userId'] + genres)
    ratings_filtered = ratings.query('rating >= 2.5')
    ratings_filtered = ratings_filtered.reset_index()
    rated_movies = dict()
    for i, rating_row in ratings_filtered.iterrows():
        printR(f'Creating user profiles... {i+1} / {ratings_filtered.shape[0]}')
        genre_vector = movies.loc[movies['movieId'] == rating_row['movieId']].values.flatten().tolist()[2:]
        if rating_row['userId'] in user_profiles['userId'].values:
            user_row = user_profiles.loc[user_profiles['userId'] == rating_row['userId']].values.flatten().tolist()
            for i in range(len(genre_vector)):
                user_row[i+1] += genre_vector[i]
            user_profiles.loc[user_profiles['userId'] == rating_row['userId']] = user_row
            rated_movies[int(rating_row['userId'])].append({
                'movieId': int(rating_row['movieId']),
                'rating': rating_row['rating']
            })
        else:
            user_row = [int(rating_row['userId'])] + genre_vector
            user_profiles.loc[len(user_profiles)] = user_row
            rated_movies[int(rating_row['userId'])] = [{
                'movieId': int(rating_row['movieId']),
                'rating': rating_row['rating']
            }]
    return user_profiles, rated_movies


def cosine_sim(A: list, B: list):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    AxB = [i * j for i, j in zip(A, B)]
    divident = sum(AxB)
    A_squares = [i ** 2 for i in A]
    B_squares = [i ** 2 for i in B]
    divisor = np.sqrt(sum(A_squares)) * np.sqrt(sum(B_squares))
    return divident / divisor


def content_based(user_profiles: pd.DataFrame, movies: pd.DataFrame):
    results = {}
    for i, user_row in user_profiles.iterrows():
        printR(f'Content based... {i+1} / {len(user_profiles)}')
        top_movies = []
        user_id = int(user_row['userId'])
        user_preferences = user_row.values.tolist()[1:]
        for j, movie_row in movies.iterrows():
            movie_id = movie_row['movieId']
            movie_title = movie_row['title']
            movie_categories = movie_row.values.tolist()[2:]
            user_movie_cos_similarity = cosine_sim(user_preferences, movie_categories)
            top_movies.append({
                'movieId': movie_id,
                'title': movie_title,
                'similarity': user_movie_cos_similarity,
            })
        results[user_id] = top_movies
    return results


def get_best_movies_from_users(similar_users: list, movies: pd.DataFrame, rated_movies: dict):
    top_movies = []
    top_n = 50
    top_n_users = similar_users[:top_n]
    top_n_userIds = [u['userId'] for u in top_n_users]

    for userId in top_n_userIds:
        for rated_movie in rated_movies[userId]:
            movie_row = movies.loc[movies['movieId'] == rated_movie['movieId']]
            movie_id = int(movie_row['movieId'])
            movie_name = movie_row['title'].values[0]
            top_movies_ids = [mId['movieId'] for mId in top_movies]
            if movie_id not in top_movies_ids:
                top_movies.append({
                    'movieId': movie_id,
                    'title': movie_name,
                    'similarity': rated_movie['rating'],
                })
            else:
                for m in top_movies:
                    if int(m['movieId']) == movie_id:
                        m['similarity'] += rated_movie['rating']
                        break
    # average out the rating and map it to value 0-1
    for top_movie in top_movies:
        top_movie['similarity'] /= top_n
        top_movie['similarity'] = np.interp(top_movie['similarity'], [0, 5], [0, 1])
    return top_movies


def collaborative_filtering(user_profiles: pd.DataFrame, movies: pd.DataFrame, rated_movies: dict):
    user_similarities = {}
    for i, user_row_i in user_profiles.iterrows():
        printR(f'Collaborative filtering... {i+1} / {len(user_profiles)}')
        top_similar_users = []
        user_preferences_i = user_row_i.values.tolist()[1:]
        for j, user_row_j in user_profiles.iterrows():
            if user_row_i['userId'] == user_row_j['userId']:
                continue
            user_preferences_j = user_row_j.values.tolist()[1:]
            user_user_cos_sim = cosine_sim(user_preferences_i, user_preferences_j)
            top_similar_users.append({
                'userId': int(user_row_j['userId']),
                'user_user_cos_sim': user_user_cos_sim,
            })
        sorted_top_similar_users = sorted(top_similar_users, key=lambda x: x['user_user_cos_sim'])[::-1]
        user_similarities[int(user_row_i['userId'])] = sorted_top_similar_users
    results = {}
    for i, user_row in user_profiles.iterrows():
        printR(f'Collaborative filtering, gathering top movies... {i+1} / {len(user_profiles)}')
        results[user_row['userId']] = get_best_movies_from_users(user_similarities[int(user_row['userId'])], movies, rated_movies)
    return results


def hybrid(content_based_sim, cbs_weight, collaborative_filtering_sim, cfs_weight):
    hybrid_res = {}
    for user_id in range(1, len(content_based_sim) + 1):
        printR(f'Hybrid... {user_id} / {len(content_based_sim)}')
        user_cbs = content_based_sim[user_id]
        user_cfs = collaborative_filtering_sim[user_id]
        user_cbs_sort = sorted(user_cbs, key=lambda x: x['movieId'])
        user_cfs_sort = sorted(user_cfs, key=lambda x: x['movieId'])
        hybrid_movies = []
        for cbs_movie, cfs_movie in zip(user_cbs_sort, user_cfs_sort):
            hybrid_sim = cbs_movie['similarity'] * cbs_weight + cfs_movie['similarity'] * cfs_weight
            hybrid_movies.append({
                'movieId': cbs_movie['movieId'],
                'title': cbs_movie['title'],
                'similarity':  hybrid_sim,
            })
        hybrid_res[user_id] = hybrid_movies
    return hybrid_res


def print_results(content_based_sim, collaborative_filtering_sim, hybrid_sim, ratings):
    for user_id in range(1, len(content_based_sim) + 1):
        # Movies which user already rated
        user_rated_movies = [m['movieId'] for m in ratings[user_id]]
        # Get list of similar movies
        user_cbs_movies = content_based_sim[user_id]
        user_cfs_movies = collaborative_filtering_sim[user_id]
        user_hyb_movies = hybrid_sim[user_id]
        # Filter out the movies which were already rated
        user_cbs_movies_filtered = [movie for movie in user_cbs_movies if movie['movieId'] not in user_rated_movies]
        user_cfs_movies_filtered = [movie for movie in user_cfs_movies if movie['movieId'] not in user_rated_movies]
        user_hyb_movies_filtered = [movie for movie in user_hyb_movies if movie['movieId'] not in user_rated_movies]
        # Sort by similarity and pick top n
        cbs_movies = sorted(user_cbs_movies_filtered, key=lambda x: x['similarity'], reverse=True)[:TOP_N]
        cfs_movies = sorted(user_cfs_movies_filtered, key=lambda x: x['similarity'], reverse=True)[:TOP_N]
        hyb_movies = sorted(user_hyb_movies_filtered, key=lambda x: x['similarity'], reverse=True)[:TOP_N]
        
        appendToFile(f'User {user_id}')
        # Content based
        appendToFile(f'Top {TOP_N} content based:')
        for i, cbs_m in zip(range(1, len(cbs_movies) + 1), cbs_movies):
            appendToFile(f"    {i}: {cbs_m['similarity']:.03f} {str(cbs_m['movieId']).rjust(6, ' ')} - {cbs_m['title']} ")
        # Collaborative filtering
        appendToFile(f'Top {TOP_N} collaborative filtering:')
        for i, cfs_m in zip(range(1, len(cfs_movies) + 1), cfs_movies):
            appendToFile(f"    {i}: {cfs_m['similarity']:.03f} {str(cfs_m['movieId']).rjust(6, ' ')} - {cfs_m['title']} ")
        # Hybrid
        appendToFile(f'Top {TOP_N} hybrid:')
        for i, hyb_m in zip(range(1, len(hyb_movies) + 1), hyb_movies):
            appendToFile(f"    {i}: {hyb_m['similarity']:.03f} {str(hyb_m['movieId']).rjust(6, ' ')} - {hyb_m['title']} ")
        appendToFile('=' * 60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    genres = extract_all_genres(movies)
    movies = get_genre_vector(movies, genres)

    # User profiles and ratings ########################################################################################
    user_profiles_path = '../results/user_profiles.csv'
    ratings_path = '../results/ratings.pickle'
    if not os.path.isfile(user_profiles_path) or not os.path.isfile(ratings_path):
        user_profiles, rated_movies = get_user_profiles(ratings, movies, genres)
        user_profiles.to_csv('../results/user_profiles.csv', index=False)
        with open('../results/ratings.pickle', 'wb') as ratings_f:
            pickle.dump(rated_movies, ratings_f)
    else:
        user_profiles = pd.read_csv(user_profiles_path)
        with open(ratings_path, 'rb') as ratings_f:
            rated_movies = pickle.load(ratings_f)

    # Content based ####################################################################################################
    content_based_path = '../results/content_based.pickle'
    if not os.path.isfile(content_based_path):
        content_based_similar_movies = content_based(user_profiles, movies)
        with open(content_based_path, 'wb') as content_based_f:
            pickle.dump(content_based_similar_movies, content_based_f)
    else:
        with open(content_based_path, 'rb') as content_based_f:
            content_based_similar_movies = pickle.load(content_based_f)

    # Collaborative filtering ##########################################################################################
    collaborative_filtering_path = '../results/collaborative_filtering.pickle'
    if not os.path.isfile(collaborative_filtering_path):
        collaborative_filtering_similar_movies = collaborative_filtering(user_profiles, movies, rated_movies)
        with open(collaborative_filtering_path, 'wb') as collaborative_filtering_f:
            pickle.dump(collaborative_filtering_similar_movies, collaborative_filtering_f)
    else:
        with open(collaborative_filtering_path, 'rb') as collaborative_filtering_f:
            collaborative_filtering_similar_movies = pickle.load(collaborative_filtering_f)

    # Hybrid ###########################################################################################################
    hybrid_path = '../results/hybrid.pickle'
    if not os.path.isfile(hybrid_path):
        hybrid_similar_movies = hybrid(content_based_similar_movies, 0.6, collaborative_filtering_similar_movies, 0.4)
        with open(hybrid_path, 'wb') as hybrid_f:
            pickle.dump(hybrid_similar_movies, hybrid_f)
    else:
        with open(hybrid_path, 'rb') as hybrid_f:
            hybrid_similar_movies = pickle.load(hybrid_f)

    # Results ##########################################################################################################
    printR("")
    print_results(content_based_similar_movies, collaborative_filtering_similar_movies, hybrid_similar_movies, rated_movies)
    exit(0)
