import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

# COCO 데이터셋 클래스 인덱스와 클래스 이름 매핑
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'boy', 'hair', 'cat'
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 객체 태깅 함수 수정
def tag_objects(image_path):
    # 이미지 전처리
    image = transform_image(image_path)
    image = image.unsqueeze(0)
    
    # 추론 수행
    with torch.no_grad():
        predictions = model(image)
    
    # 예측 결과에서 객체 클래스 가져오기
    labels = predictions[0]['labels']
    labels = labels.numpy()
    
    # 객체 클래스 출력
    for label in labels:
        class_name = map_label_to_class_name(label.item())
        print("Detected Object:", class_name)

# 이미지 전처리를 위한 변환 함수
def transform_image(image_path):
    # 이미지를 PIL 이미지로 로드
    image = Image.open(image_path).convert("RGB")
    
    # 이미지를 텐서로 변환
    image = F.to_tensor(image)
    
    # 텐서를 정규화
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return image

# 객체 레이블을 클래스 이름으로 매핑하는 함수
def map_label_to_class_name(label):
    return COCO_CLASSES[label]

# 사전 훈련된 Faster R-CNN 모델 불러오기
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 이미지 파일 경로
image_path = "example.jpg"

# 객체 태깅 함수 호출
tag_objects(image_path)
