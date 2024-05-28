import pandas as pd
# user = pd.read_csv('../Data/whyout_data/user.csv') # (31177,3)

# user, place, product, video 메타데이터
user_interest = pd.read_csv('Data/whyout_data/user_interest.csv') # (31177,4)
place = pd.read_csv('Data/whyout_data/place.csv') # shape(4697,10)
product = pd.read_csv('Data/whyout_data/product.csv') # shape(5821,11)
video = pd.read_csv('Data/whyout_data/video.csv') # shape(3250, 9)

# 유저의 전체 행동데이터 
user_place = pd.read_csv('Data/whyout_data/user_place.csv') # (31177,4697)
user_product = pd.read_csv('Data/whyout_data/user_product.csv') # (31177,5821)
user_video = pd.read_csv('Data/whyout_data/user_video.csv') # (31177, 3250)

# [Case 2] 유저의 행동데이터 (행동 데이터가 없는 유저를 삭제한 데이터)
case2_user_place = pd.read_csv('Data/whyout_data/case2_user_place.csv') # (22420,4697) 
case2_user_product = pd.read_csv('Data/whyout_data/case2_user_product.csv') # (2994,5821)
case2_user_video = pd.read_csv('Data/whyout_data/case2_user_video.csv') # (11067, 3250)

# [Case 2] 유저의 행동데이터 idx (행동 데이터가 없는 유저를 삭제한 데이터의 idx)
case2_user_place_idx = pd.read_csv('Data/whyout_data/case2_user_place_idx.csv') # (22420,4)
case2_user_product_idx = pd.read_csv('Data/whyout_data/case2_user_product_idx.csv') # (2294,4)
case2_user_video_idx = pd.read_csv('Data/whyout_data/case2_user_video_idx.csv') # (11067, 4)

# [Case 2] SGD 결과(R)
case2_sgd_rating_place = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_sgd_rating_place.csv')
case2_sgd_rating_product = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_sgd_rating_product.csv')
case2_sgd_rating_video = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_sgd_rating_video.csv')

# [Case 2] SGD 결과의 user_latent(U)
case2_user_latent_place = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_user_latent_place.csv')
case2_user_latent_product = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_user_latent_product.csv')
case2_user_latent_video = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_user_latent_video.csv')

# [Case 2] SGD 결과의 item_latent(V)
case2_item_latent_place = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_item_latent_place.csv')
case2_item_latent_product = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_item_latent_product.csv')
case2_item_latent_video = pd.read_csv('Data/whyout_data/sgd_result/del_data/case2_item_latent_video.csv')