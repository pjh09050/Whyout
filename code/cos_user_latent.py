from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def place_user_latent_cos(user_id, data, idx):
    drop_user_place_index = int(idx[idx['idx'] == user_id].iloc[:,3])

    cosine_sim_matrix = cosine_similarity(data)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=data.index, columns=data.index)
    
    # user_id의 score를 가져옴
    user_similarities = cosine_sim_df.loc[drop_user_place_index]
    
    # user_id를 선택하지 않도록 -1을 해줌
    user_similarities[drop_user_place_index] = -1
    
    # user_id와 가장 유사한 유저 선택
    most_similar_user_id = user_similarities.idxmax()
    highest_similarity_score = user_similarities.max()

    # new_user_idx 찾기
    new_user_id = int(idx[idx.iloc[:,3] == most_similar_user_id].iloc[:,0])
    print(f'The user most similar to user {user_id} is user {new_user_id} with a similarity score of {highest_similarity_score}')

    return new_user_id

def product_user_latent_cos(user_id, data, idx):
    drop_user_product_index = int(idx[idx['idx'] == user_id].iloc[:,3])

    cosine_sim_matrix = cosine_similarity(data)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=data.index, columns=data.index)
    
    # user_id의 score를 가져옴
    user_similarities = cosine_sim_df.loc[drop_user_product_index]
    
    # user_id를 선택하지 않도록 -1을 해줌
    user_similarities[drop_user_product_index] = -1

    # user_id와 가장 유사한 유저 선택
    most_similar_user_id = user_similarities.idxmax()
    highest_similarity_score = user_similarities.max()

    # new_user_idx 찾기
    new_user_id = int(idx[idx.iloc[:,3] == most_similar_user_id].iloc[:,0])
    print(f'The user most similar to user {user_id} is user {new_user_id} with a similarity score of {highest_similarity_score}')

    return new_user_id

def video_user_latent_cos(user_id, data, idx):
    drop_user_video_index = int(idx[idx['idx'] == user_id].iloc[:,3])

    cosine_sim_matrix = cosine_similarity(data)
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=data.index, columns=data.index)
    
    # user_id의 score를 가져옴
    user_similarities = cosine_sim_df.loc[drop_user_video_index]
    
    # user_id를 선택하지 않도록 -1을 해줌
    user_similarities[drop_user_video_index] = -1
    
    # user_id와 가장 유사한 유저 선택
    most_similar_user_id = user_similarities.idxmax()
    highest_similarity_score = user_similarities.max()

    # new_user_idx 찾기
    new_user_id = int(idx[idx.iloc[:,3] == most_similar_user_id].iloc[:,0])
    print(f'The user most similar to user {user_id} is user {new_user_id} with a similarity score of {highest_similarity_score}')

    return new_user_id