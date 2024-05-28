import pandas as pd
import csv

# 엑셀 파일 읽기
xls_file = 'Data/whyout_data/raw_data/와이아웃 ai용 raw data_0523.xlsx'
xls = pd.ExcelFile(xls_file)
sheet_name_list = ['place', 'video', 'product', 'user']
txt_name_list = ['poi', 'video', 'product']
c = 0

# 시트별로 CSV로 내보내기
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name, index_col=0, header=1)
    sheet_name = sheet_name_list[c]
    csv_file = f'{sheet_name}.csv'
    df.to_csv(csv_file, encoding='utf-8-sig')
    print(f'{csv_file}로 내보내기 완료!')
    c += 1

    
for j in range(len(txt_name_list)):
    # 행동데이터 txt 파일을 csv 파일로 변환
    text_file_path = f"Data/whyout_data/raw_data/user_{txt_name_list[j]}_output_total.txt"
    csv_file_path = f"user_{sheet_name_list[j]}.csv"

    with open(text_file_path, 'r') as text_file, open(csv_file_path, 'w', newline='') as csv_file:
        # CSV writer 객체 생성
        csv_writer = csv.writer(csv_file)
        
        # 첫 줄을 읽어서 컬럼 수를 파악하고, 컬럼명을 생성
        first_line = text_file.readline().strip().split(',')
        column_names = [str(i) for i in range(len(first_line))]
        csv_writer.writerow(column_names)
        
        # 첫 줄 데이터를 CSV 파일에 쓰기
        first_line = [int(item) for item in first_line]
        csv_writer.writerow(first_line)
        
        # 나머지 줄을 읽어서 CSV 파일에 쓰기
        for line in text_file:
            # 줄에서 필요한 데이터 추출
            data = line.strip().split(',')
            data = [int(item) for item in data]
            # CSV 파일에 데이터 쓰기
            csv_writer.writerow(data)

    print(f"텍스트 파일 '{text_file_path}'을 CSV 파일 '{csv_file_path}'로 변환했습니다.")

user_place = pd.read_csv('user_place.csv')
user_product = pd.read_csv('user_product.csv')
user_video = pd.read_csv('user_video.csv')
user = pd.read_csv('user.csv')

# 사용자의 평가 유무 확인
def find_zero_indices(df):
    return df.index[df.eq(0).all(axis=1)].tolist()

place_zero_indices = find_zero_indices(user_place)
product_zero_indices = find_zero_indices(user_product)
video_zero_indices = find_zero_indices(user_video)

print('장소를 평가하지 않은 사용자 수:',len(place_zero_indices))
print('상품을 평가하지 않은 사용자 수:',len(product_zero_indices))
print('영상을 평가하지 않은 사용자 수:',len(video_zero_indices))

case2_user_place = user_place.drop(place_zero_indices)
case2_user_product = user_product.drop(product_zero_indices)
case2_user_video = user_video.drop(video_zero_indices)
case2_user_place.to_csv('case2_user_place.csv', index=False)
case2_user_product.to_csv('case2_user_product.csv', index=False)
case2_user_video.to_csv('case2_user_video.csv', index=False)

case2_user_place_idx = user.drop(place_zero_indices)
case2_user_product_idx = user.drop(product_zero_indices)
case2_user_video_idx = user.drop(video_zero_indices)
case2_user_place_idx.to_csv('case2_user_place_idx.csv', index=False)
case2_user_product_idx.to_csv('case2_user_product_idx.csv', index=False)
case2_user_video_idx.to_csv('case2_user_video_idx.csv', index=False)