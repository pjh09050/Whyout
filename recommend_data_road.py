import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

def recommend_step1(item, sgd_preds, user_id, item_df, ratings_df, ratings_df_idx, num_recommendations):
    drop_user_index = int(ratings_df_idx[ratings_df_idx['idx'] == user_id].iloc[:,3])
    # 원본 행동 데이터에서 user_id에 해당하는 행을 가져옴
    user_data = ratings_df.loc[drop_user_index]
    # 유저가 평가하지 않은 아이템의 index를 가져옴
    user_history_non_indices = [int(i) for i in user_data[user_data <= 0].index.tolist()]
    # user_id에 해당하는 SGD 결과값을 가져온 후, 유저가 평가하지 않은 아이템의 결과값만 뽑아옴
    user_predictions = sgd_preds.loc[drop_user_index]
    user_predictions_filtered = user_predictions.iloc[user_history_non_indices]
    # SGD 결과값이 높은 순으로 정렬
    sorted_predictions = user_predictions_filtered.sort_values(ascending=False)
    # 상위 N개만큼 뽑아옴
    top_recommendations = sorted_predictions.index.tolist()[:num_recommendations]
    # 아이템 idx 매핑
    recommendations_result = item_df.iloc[top_recommendations]['idx'].tolist()
    #print(f"user {user_id}에게 추천해줄 {10}개 {item} idx : {recommendations_result}")
    return recommendations_result

def item_user_latent_cos(user_id, original_item, item_list, dict):
    print('유사도 선택시 item_list:', item_list)
    if user_id in dict[item_list[0]][3]['idx'].values:
        item = item_list[0]
        print(f'user {user_id}는 {item}에 대한 행동이 존재함')
    elif user_id in dict[item_list[1]][3]['idx'].values:
        item = item_list[1]
        print(f'user {user_id}는 {item}에 대한 행동이 존재함')
    else:
        """
        모든 아이템에 대한 행동이 없는 유저에게 추천하는 함수 추가 
        """
        print(f'user {user_id}는 모든 아이템에 대한 행동이 없음')

    print(f'2. {item} user latent에서 user {user_id}과 유사한 user 찾기')
    drop_user_place_index = int(dict[item][3][dict[item][3]['idx'] == user_id].iloc[:,3])
    cosine_sim_matrix = cosine_similarity(dict[item][4])
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=dict[item][4].index, columns=dict[item][4].index)
    # user_id의 score를 가져옴
    user_similarities = cosine_sim_df.loc[drop_user_place_index]
    # user_id를 선택하지 않도록 -1을 해줌
    user_similarities[drop_user_place_index] = -1
    # 유사도가 높은 순으로 정렬
    sorted_user_similarities = user_similarities.sort_values(ascending=False)
    # 유사도가 높은 유저를 순서대로 기존의 추천하려는 아이템에 행동이 있는지 확인
    for i in range(len(sorted_user_similarities)):
        most_similar_user_id = sorted_user_similarities.index[i]
        new_user_id = int(dict[item][3][dict[item][3].iloc[:,3] == most_similar_user_id].iloc[:,0])
        highest_similarity_score = sorted_user_similarities.iloc[i]
        
        if new_user_id in dict[original_item][3]['idx'].values:
            # new_user_idx 찾기
            new_user_id = int(dict[item][3][dict[item][3].iloc[:,3] == most_similar_user_id].iloc[:,0])
            print('3. update new_user_id:', new_user_id)
            break
        else:
            print(f'{new_user_id}가 {original_item}에 대한 행동이 없음')
    print(f'4. user {user_id}과 가장 유사한 user : {new_user_id}, cos : {highest_similarity_score}')
    return new_user_id


def recommendation_system(user_id, item, item_list, dict, num_recommendations):
    if user_id in dict[item][3]['idx'].values:
        recomm_list = recommend_step1(item, dict[item][0], user_id, dict[item][1], dict[item][2], dict[item][3], num_recommendations)
        # print(f"1. user {user_id}에게 추천해줄 {10}개 {item} idx : {recomm_list}")
        return recomm_list
    else:
        print(f'1. user {user_id}는 {item}에 대한 행동내역이 없음')
        # 아이템 리스트에서 행동이 없는 아이템 제거
        item_list.remove(item)
        new_user_id = item_user_latent_cos(user_id, item, item_list, dict)
        user_id = new_user_id
        print(f'5. user {new_user_id}에게 {item} recommend_step2 시작')
        recom_list2 = recommend_step1(item, dict[item][0], user_id, dict[item][1], dict[item][2], dict[item][3], num_recommendations)
        #print(f"6. user {user_id}에게 추천해줄 {num_recommendations}개 {item} idx : {recom_list2}")
        return recom_list2
    
# 모든 데이터 place, product, video 추천
def recommend_all(total_sgd_preds, user_id, total_df, ratings_df, idx, num_recommendations):
    if user_id in idx['idx'].values:
        print(f'{user_id}번 유저의 행동이 있습니다.')
        user_index = int(idx[idx['idx'] == user_id].iloc[:,2])
        #print('user_index:', user_index)

        # 원본 평점 데이터에서 user_id에 해당하는 행을 DataFrame으로 가져온다.
        user_data = ratings_df.loc[user_index]

        # 사용자가 이미 평가한 상품의 인덱스를 추출
        user_history_indices = [int(i) for i in user_data[user_data > 0].index.tolist()]
        user_history_non_indices = [int(i) for i in user_data[user_data <= 0].index.tolist()]
        #print(f'이미 평가한 아이템 길이: {len(user_history_indices)}')
        #print(len(user_history_non_indices),user_history_non_indices)
        non_recommendations = total_df.iloc[user_history_indices]['idx'].tolist()
        recommendations = total_df.iloc[user_history_non_indices]['idx'].tolist()
        #print("이미 평가한 아이템 길이, idx:", len(non_recommendations),non_recommendations)
        #print("평가 안한 아이템 길이, idx:", len(recommendations), recommendations)

        # SGD를 통해 예측된 사용자의 평점을 기반으로 데이터 정렬
        user_predictions = total_sgd_preds.loc[user_index]
        user_predictions_filtered = user_predictions.iloc[user_history_non_indices]
        sorted_predictions = user_predictions_filtered.sort_values(ascending=False)
        top_recommendations = sorted_predictions.index.tolist()[:num_recommendations]
        recommendations_result = total_df.iloc[top_recommendations]['idx'].tolist()
        print(f"user {user_id}에게 추천해줄 {10}개 아이템 idx : {recommendations_result}")
        return recommendations_result
    else:
        print(f'{user_id}번 유저의 행동이 없습니다.')
