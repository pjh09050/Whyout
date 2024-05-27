import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


def vectorize_outdoor(interests):
    if isinstance(interests, list):
        return interests
    else:
        return [0] * 10
def vectorize_item(interests):
    if isinstance(interests, list):
        return interests
    else:
        return [0] * 3

def make_user_interest(user_interest):
    df_filled = user_interest.fillna(0) # NaN 값을 0으로 대체

    # 관심 아이템, 아웃도어 벡터화
    df_filled['관심 아이템'] = df_filled['관심 아이템'].apply(vectorize_item)
    df_filled['관심 아웃도어'] = df_filled['관심 아웃도어'].apply(vectorize_outdoor)

    df_features = df_filled.drop(columns=['idx', '나이']) # 불필요한 컬럼 제거
    df_features['Combined_Interest'] = df_features['관심 아이템'] + df_features['관심 아웃도어'] # 관심 아이템 + 관심 아웃도어를 하나의 차원으로 변경
    df_features = pd.DataFrame(df_features['Combined_Interest']) # 유사도 측정을 위한 데이터프레임 변환
    return df_features

def interest_similarity(item, case2_dict, user_interest, item_interest, outdoor_interest):
    df_features = make_user_interest(user_interest) # 기존 유저의 관심 아이템 + 관심 아웃도어의 전처리 작업
    new_data = item_interest + outdoor_interest # 새로운 유저의 관심 아이템 + 관심 아웃도어를 하나의 차원으로 변경
    
    # 신규 유저와 기존 유저의 관심 항목이 같은지 확인
    exact_match_indices = []
    for index, row in df_features.iterrows():
        if row['Combined_Interest'] == new_data:
            exact_match_indices.append(index)

    # 만약 기존 유저와 관심 항목이 일치하는게 없다면 if, 있으면 else
    if not exact_match_indices:
        combined_interest_matrix = np.array(df_features['Combined_Interest'].tolist()) # 코사인 유사도를 계산하기 위해 데이터프레임 변환
        cosine_sim_matrix = cosine_similarity([new_data], combined_interest_matrix) # 새로운 데이터와의 코사인 유사도 계산
        sorted_indices = np.argsort(-cosine_sim_matrix[0])
        sorted_similarity_scores = cosine_sim_matrix[0][sorted_indices]

        # 기존 아이템에 행동이 존재하는지 확인
        user_id, similarity_score = None, None
        for i, s in list(zip(sorted_indices, sorted_similarity_scores)): # (user_id의 index, 코사인 유사도 값)
            user_index = int(user_interest[user_interest.index == i].iloc[:,0]) # user_id의 idx(key)값 가져오기
            if user_index in case2_dict[item][3]['idx'].values: # key값이 원래 추천하려는 아이템에 존재한다면 if
                user_id = user_index
                similarity_score = s
                break
            else:
                print(f'모든 유저가 {item}에 대한 행동이 없습니다.')
        print(f"{item}에 행동이 존재하며 유사도가 가장 높은 인덱스는 {user_id}이며, 유사도 점수는 {similarity_score}입니다.")
    else:
        # 관심항목이 정확히 일치하는 user_id에서 기존에 추천하려는 item에 행동이 있는지 확인
        match_idx = []
        for i in exact_match_indices: 
            i = int(user_interest[user_interest.index == i].iloc[:,0]) # user_id의 idx(key)값을 가져옴
            if i in case2_dict[item][3]['idx'].values: # 기존에 추천하려는 item에 행동이 있는지 확인
                match_idx.append(i) 
        print(f'{item}에 대한 행동이 있는 user index : {match_idx}')
        # 신규 유저와 관심항목이 동일한 기존 유저들 중 item에 대한 행동이 하나라도 존재하면
        if match_idx:
            user_actions = case2_dict[item][6].loc[match_idx].apply(lambda row : (row != 0).sum(), axis=1) # 행동이 있는 부분을 count함
            user_id = int(user_interest[user_interest.index == user_actions.idxmax()].iloc[:,0]) # 가장 행동이 많은 user_id를 뽑음
            print(f'{item}에 대한 행동이 가장 많은 유저 : user {user_id}')
        else:
            print(f'모든 유저가 {item}에 대한 행동이 없습니다')
    return user_id