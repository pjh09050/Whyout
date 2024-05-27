import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
import warnings
warnings.filterwarnings("ignore")

def recommend_step(item, user_id, case2_dict, num_recommendations):
    """
    case2_dict[item][0] : sgd 결과
    case2_dict[item][1] : meta data
    case2_dict[item][2] : 유저의 행동데이터
    case2_dict[item][3] : 행동데이터의 idx
    """
    drop_user_index = int(case2_dict[item][3][case2_dict[item][3]['idx'] == user_id].iloc[:,3]) # user_id에 해당하는 index 값 가져오기
    user_data = case2_dict[item][2].loc[drop_user_index] # 원본 행동 데이터에서 user_id에 해당하는 행을 가져옴
    user_history_non_indices = [int(i) for i in user_data[user_data <= 0].index.tolist()] # 유저가 평가하지 않은 아이템의 index를 가져옴
    user_predictions = case2_dict[item][0].loc[drop_user_index] # user_id에 해당하는 SGD 결과값을 가져온 후
    user_predictions_filtered = user_predictions.iloc[user_history_non_indices] # 유저가 평가하지 않은 아이템의 결과값만 뽑아옴
    sorted_predictions = user_predictions_filtered.sort_values(ascending=False) # SGD 결과값이 높은 순으로 정렬
    top_recommendations = sorted_predictions.index.tolist()[:num_recommendations] # 상위 N개만큼 뽑아옴
    recommendations_result = case2_dict[item][1].iloc[top_recommendations]['idx'].tolist() # 아이템 idx 매핑
    return recommendations_result

def user_latent_cos(user_id, original_item, item_list, case2_dict):
    """
    original_item : 추천하려는 item
    """
    exist_action = False
    item = ''
    for item_category in item_list:
        if user_id in case2_dict[item_category][3]['idx'].values:
            exist_action = True
            item = item_category
            print(f'user {user_id}는 {item_category}에 대한 행동이 존재함')
            break
    if exist_action is False:
        """ 모든 아이템에 대한 행동이 없는 유저에게 추천하는 함수 추가? """
        print(f'user {user_id}는 모든 아이템에 대한 행동이 없음')

    print(f'{item} user latent에서 user {user_id}과 유사한 user 찾기')
    drop_user_place_index = int(case2_dict[item][3][case2_dict[item][3]['idx'] == user_id].iloc[:,3]) # user_id에 대한 index 번호 추출
    cosine_sim_matrix = cosine_similarity(case2_dict[item][4]) # user_latent에 대한 코사인 유사도 계산
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=case2_dict[item][4].index, columns=case2_dict[item][4].index)
    user_similarities = cosine_sim_df.loc[drop_user_place_index] # user_id의 코사인 유사도 값을 가져옴
    user_similarities[drop_user_place_index] = -1 # user_id를 선택하지 않도록 -1을 해줌
    sorted_user_similarities = user_similarities.sort_values(ascending=False) # 유사도가 높은 순으로 정렬
    # 유사도가 높은 유저를 순서대로 기존의 추천하려는 아이템에 행동이 있는지 확인
    for i in range(len(sorted_user_similarities)):
        most_similar_user_id = sorted_user_similarities.index[i]
        new_user_id = int(case2_dict[item][3][case2_dict[item][3].iloc[:,3] == most_similar_user_id].iloc[:,0]) # new_user_idx 찾기
        highest_similarity_score = sorted_user_similarities.iloc[i]
        if new_user_id in case2_dict[original_item][3]['idx'].values: # 기존에 추천하려는 아이템에 행동이 있다면 new_user_id로 선택됨
            new_user_id = new_user_id
            break
        else:
            print(f'{new_user_id}가 {original_item}에 대한 행동이 없음')
    print(f'user {user_id}과 가장 유사한 user : {new_user_id}, cos : {highest_similarity_score}')
    return new_user_id

def get_recommended_items(user_id, item, item_list, case2_dict, num_recommendations):
    if user_id in case2_dict[item][3]['idx'].values:
        recomm_list = recommend_step(item, user_id, case2_dict, num_recommendations)
        return recomm_list
    else:
        print(f'user {user_id}는 {item}에 대한 행동내역이 없음')
        item_list.remove(item) # 아이템 리스트에서 행동이 없는 아이템 제거
        new_user_id = user_latent_cos(user_id, item, item_list, case2_dict) # user_id와 가장 유사도가 높은 user_id를 탐색
        recom_list2 = recommend_step(item, new_user_id, case2_dict, num_recommendations) # new_user_id에 대해 아이템 추천
        return recom_list2
    
# 모든 데이터 place, product, video 추천
def recommend_all(total_sgd_preds, user_id, total_df, ratings_df, idx, num_recommendations):
    if user_id in idx['idx'].values:
        print(f'{user_id}번 유저의 행동이 있습니다.')
        user_index = int(idx[idx['idx'] == user_id].iloc[:,2])

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

def item_latent_cos(user_id, item, case2_dict, num_recommendations):
    if user_id in case2_dict[item][3]['idx'].values:
        print(f'{item} item latent에서 user {user_id}이 행동했던 item과 유사한 item 찾기')
        drop_user_place_index = int(case2_dict[item][3][case2_dict[item][3]['idx'] == user_id].iloc[:,3]) # user_id에 대한 index 번호 추출
        all_item = case2_dict[item][2].loc[drop_user_place_index]
        non_zero_columns = all_item[all_item != 0].index.tolist()
        select_item = int(random.choice(non_zero_columns))
        cosine_sim_matrix = cosine_similarity(case2_dict[item][5]) # user_latent에 대한 코사인 유사도 계산
        cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=case2_dict[item][5].index, columns=case2_dict[item][5].index)
        user_similarities = cosine_sim_df.loc[select_item] # user_id의 코사인 유사도 값을 가져옴
        user_similarities[select_item] = -1 # user_id를 선택하지 않도록 -1을 해줌
        sorted_user_similarities = user_similarities.sort_values(ascending=False) # 유사도가 높은 순으로 정렬
        top_recommendations = sorted_user_similarities.index.tolist()[:num_recommendations] # 상위 N개만큼 뽑아옴
        recommendations_result = case2_dict[item][1].iloc[top_recommendations]['idx'].tolist() # 아이템 idx 매핑
        print(f'user {user_id}과 사용했던 아이템 {select_item} 와 유사도가 높은 idx: {recommendations_result}')
        return recommendations_result
    else:
        print(f'user {user_id}는 {item}에 대한 행동이 없습니다')