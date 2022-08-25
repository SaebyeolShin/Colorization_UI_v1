######## lib ##########

import os
from os.path import join, isfile, isdir
from os import listdir
from argparse import ArgumentParser
import sys
import numpy as np
import cv2
from skimage import color
import skimage.io

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
setup_logger()

import torch
from tqdm import tqdm

import shutil
import glob
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


from PIL import Image

class args:
    test_img_dir = "./examplar/sample_videos/clips/taxi" # input image 경로 (crop 할 이미지 원본 경로)
    filter_no_obj = False
    test_img = "41.jpg"
    cropped_dir = "./examplar/sample_videos/frames" 
    segmented = True
    taxi = True
    
    
def read_to_pil(img_path):
    '''
    return: pillow image object HxWx3
    '''
    out_img = Image.open(img_path)
    if len(np.asarray(out_img).shape) == 2:
        out_img = np.stack([np.asarray(out_img), np.asarray(out_img), np.asarray(out_img)], 2)
        out_img = Image.fromarray(out_img)
    return out_img

input_dir = args.test_img_dir
image_list = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

for image_path in tqdm(image_list):
    
    img = cv2.imread(join(input_dir, image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    l_stack = np.stack([l_channel, l_channel, l_channel], axis=2)
    outputs = predictor(l_stack)
    
    if args.taxi:
        outputs["instances"] = outputs["instances"][outputs["instances"].pred_classes == 2] ### CAR only
    # print(outputs['instances'].pred_classes)
    # print(outputs["instances"].scores)
    pred_bbox = outputs["instances"].pred_boxes.to(torch.device('cpu')).tensor.numpy()
    pred_scores = outputs["instances"].scores.cpu().data.numpy()
    
    
    ## 1단계 전체 이미지 segmentation
    mask = outputs["instances"].pred_masks
    
    if args.segmented:
        mask1 = mask.detach().cpu().numpy().astype(int)
        mask2 = np.where(mask1.sum(axis=0)>0, 1, 0)
        mask3 = np.stack([mask2, mask2, mask2], axis=2) # mask 3개 모두 사용 

        mask4 = np.where(mask3==0, 1, 0)

        original_img = img * mask4 # original에서 segmented부분 뺀 사진
        seg_img = img * mask3 # 검정색 배경 segmented
        seg_img_2 = np.where(seg_img==0, 255, seg_img) # 흰색 배경
        black_mask = np.where(seg_img==0, 255, 0)
        
        # plt.imshow(seg_img_2)
        skimage.io.imsave("./visualize/ori_white_mask.png",seg_img_2) # ori_white_mask
        skimage.io.imsave("./visualize/black_mask.png",black_mask) # black_mask
        skimage.io.imsave(f"./{args.cropped_dir}/target_0.png",seg_img) # ori_black_mask -> colorize할 것
        skimage.io.imsave(f"./visualize/original_img.png",original_img) # ori_black_mask -> colorize할 것



os.chdir("./examplar")
os.system(f"python test.py --clip_path ./sample_videos/frames \
   --ref_path ./sample_videos/ref/taxi \
   --output_path ./sample_videos/output")


# 합치기
img = Image.open(f"../visualize/original_img.png")
img2 = Image.open(f"./sample_videos/output/frames_ref/target_0.png")

im = img.resize((768, 432))

original = np.array(im)
seg = np.array(img2)
converted = original+seg

skimage.io.imsave("../visualize/converted.png",converted) # original_img 저장