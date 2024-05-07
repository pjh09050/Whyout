import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 사전 훈련된 Faster R-CNN 모델 불러오기
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO 데이터셋 클래스 인덱스와 클래스 이름 매핑
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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

# 객체 태깅 함수
def tag_objects(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 이미지를 텐서로 변환하고 정규화하기
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    # 추론 수행
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 예측 결과에서 객체 클래스 가져오기
    labels = predictions[0]['labels']
    labels = labels.numpy()
    
    # 객체 클래스 출력
    for label in labels:
        class_name = COCO_CLASSES[label]
        print("Detected Object:", class_name)

# 이미지 파일 경로
image_path = "example.jpg"

# 객체 태깅 함수 호출
tag_objects(image_path)
