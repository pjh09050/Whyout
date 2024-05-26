import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# 관심 아이템, 아웃도어 벡터화
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

def make_new_user_df(final_user_interest):
    # NaN 값을 0으로 대체
    df_filled = final_user_interest.fillna(0)
    # 관심 아이템, 아웃도어 컬럼의 문자열을 리스트로 변환
    df_filled['관심 아이템'] = df_filled['관심 아이템'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df_filled['관심 아웃도어'] = df_filled['관심 아웃도어'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
    df_filled['관심 아이템'] = df_filled['관심 아이템'].apply(vectorize_item)
    df_filled['관심 아웃도어'] = df_filled['관심 아웃도어'].apply(vectorize_outdoor)
    # 불필요한 컬럼 제거
    df_features = df_filled.drop(columns=['idx', '나이'])
    # 관심 아이템 + 관심 아웃도어를 하나의 차원으로 변경
    df_features['Combined_Interest'] = df_features['관심 아이템'] + df_features['관심 아웃도어']
    # 유사도 측정을 위한 데이터프레임 변환
    df_features = pd.DataFrame(df_features['Combined_Interest'])
    return df_features

def find_similar_index(final_user_interest, item_interest, outdoor_interest):
    # NaN 값을 0으로 채우고, 유사도 측정을 위한 데이터프레임 변환
    df_features = make_new_user_df(final_user_interest)
    # 새로운 유저의 관심 아이템 + 관심 아웃도어를 하나의 차원으로 변경
    new_data = item_interest + outdoor_interest
    # 새로운 데이터가 기존 데이터프레임에 존재하는지 확인
    exact_match_indices = []
    for index, row in df_features.iterrows():
        if row['Combined_Interest'] == new_data:
            exact_match_indices.append(index)
    if not exact_match_indices:
        # 코사인 유사도를 계산하기 위해 데이터프레임 변환
        combined_interest_matrix = np.array(df_features['Combined_Interest'].tolist())
        # 새로운 데이터와의 코사인 유사도 계산
        similarities = cosine_similarity([new_data], combined_interest_matrix)
        # 가장 유사한 인덱스 찾기
        most_similar_index = np.argmax(similarities)
        similarity_score = similarities[0, most_similar_index]
        result = most_similar_index
        print(f"유사도가 가장 높은 인덱스는 {most_similar_index}이며, 유사도 점수는 {similarity_score}입니다.")
    else:
        result = exact_match_indices
        print(f"새로운 데이터와 정확히 일치하는 인덱스는 {exact_match_indices}입니다.")
    return result