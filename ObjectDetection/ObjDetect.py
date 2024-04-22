import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


COCO_INSTANCE_CATEGORY_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow"
]

def get_prediction(img_path, threshold):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    
    
    predictions = model([img_tensor])
    
    
    boxes = predictions[0]['boxes'][predictions[0]['scores'] > threshold].detach().numpy()
    labels = predictions[0]['labels'][predictions[0]['scores'] > threshold].detach().numpy()
    
    return boxes, labels

def draw_boxes(img, boxes, labels):
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        color = (0, 255, 0)  
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(img, COCO_INSTANCE_CATEGORY_NAMES[label], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img

def main():
    img_path = 'image.png'  
    threshold = 0.9
    
    
    boxes, labels = get_prediction(img_path, threshold)
    
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    result_img = draw_boxes(img, boxes, labels)
    
    
    plt.figure(figsize=(10, 7))
    plt.imshow(result_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
