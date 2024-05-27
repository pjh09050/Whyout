from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def item_latent_cos(user_id, original_item, item_list, case2_dict):
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
        new_user_id = int(case2_dict[item][3][case2_dict[item][3].iloc[:,3] == most_similar_user_id].iloc[:,0])
        highest_similarity_score = sorted_user_similarities.iloc[i]
        if new_user_id in case2_dict[original_item][3]['idx'].values: # 기존에 추천하려는 아이템에 행동이 있다면 new_user_id의 idx(key)값을 가져옴
            new_user_id = int(case2_dict[item][3][case2_dict[item][3].iloc[:,3] == most_similar_user_id].iloc[:,0]) # new_user_idx 찾기
            break
        else:
            print(f'{new_user_id}가 {original_item}에 대한 행동이 없음')
    print(f'user {user_id}과 가장 유사한 user : {new_user_id}, cos : {highest_similarity_score}')
    return new_user_id