from interest_similarity import interest_similarity
from get_recommended_items import get_recommended_items
from load_data import *
import warnings
warnings.filterwarnings("ignore")

user_id = 38000
item_interest = [1,1,1]
outdoor_interest = [0,1,0,0,1,0,0,0,0,1]
num_recom = 10
case2_dict = { 'place' : [case2_sgd_rating_place, place, case2_user_place, case2_user_place_idx, 
                          case2_user_latent_place, case2_item_latent_place, user_place],
         'video' : [case2_sgd_rating_video, video, case2_user_video, case2_user_video_idx, 
                    case2_user_latent_video, case2_item_latent_video, user_video],
         'product' : [case2_sgd_rating_product, product, case2_user_product, case2_user_product_idx, 
                      case2_user_latent_product, case2_item_latent_product, user_product]}
item = 'place'
item_list = list(case2_dict.keys())

# Case 2만 구현
# if : 기존 유저, else : 신규 유저 구분하기
if user_id in user_interest['idx'].values:
    recomm_list = get_recommended_items(user_id, item, item_list, case2_dict, num_recom)
    if recomm_list == -1:
        print(f'user {user_id}은 모든 item에 행동이 없습니다.')
    else:
        print(f"user {user_id}에게 추천해줄 {num_recom}개 {item} idx : {recomm_list}")
else:
    new_user_id = interest_similarity(item, case2_dict, user_interest, item_interest, outdoor_interest)
    recomm_list = get_recommended_items(new_user_id, item, item_list, case2_dict, num_recom)
    print(f"user {user_id}에게 추천해줄 {num_recom}개 {item} idx : {recomm_list}")