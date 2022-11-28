from pycocotools.coco import COCO
import torch
import json
from torchvision import models
import numpy as np
import cv2

annotation_file = "./coco_minitrain_25k/annotations/instances_val2017.json"
coco = COCO(annotation_file)

# get all image ids
imgIds = coco.getImgIds(catIds=[1])

# all images of type person
imgs = coco.loadImgs(imgIds)

model = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
model.eval()

bounding_boxes = []

count = 1
for img in imgs:
    image = cv2.imread(f"./coco_minitrain_25k/images/val2017/{img['file_name']}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.to("cpu")
    pred = model(image)
    boxes = pred[0]["boxes"]
    scores = pred[0]["scores"]
    labels = pred[0]["labels"]
    # Save the bounding boxes in a list
    for i in range(len(boxes)):
        if scores[i] > 0.75 and labels[i] == 1:
            box = boxes[i].detach().numpy()
            box = box.astype("int")
            bounding_boxes.append({"bbox": box.tolist(), "image": img["file_name"]})
            
    print("Done with image", count, "/", len(imgs))
    count += 1

# Save the bounding boxes in a json file
with open("bounding_boxes.json", "w") as f:
    json.dump(bounding_boxes, f)
