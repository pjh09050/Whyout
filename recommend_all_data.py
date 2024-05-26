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
        print(f'이미 평가한 아이템 길이: {len(user_history_indices)}')
        #print(len(user_history_non_indices),user_history_non_indices)
        non_recommendations = total_df.iloc[user_history_indices]['idx'].tolist()
        recommendations = total_df.iloc[user_history_non_indices]['idx'].tolist()
        print("이미 평가한 아이템 idx:", non_recommendations)
        print("평가 안한 아이템 idx:", recommendations)

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
