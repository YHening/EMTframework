#%% md

# Training and Evaluation of the Multitask Learning Model

#%% md

## Imports

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7" # 0, 1, 2, ...
import pandas as pd
import ipywidgets as widgets
from PIL import Image, ImageOps, ImageDraw, ImageFont


from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
from matplotlib.pyplot import imshow
import matplotlib.pyplot as pltb

import detectron2
import torchvision
from detectron2.utils.visualizer import ColorMode
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor #, default_argument_parser, launch
from detectron2 import model_zoo
from detectron2.engine import HookBase

import torch
import torchvision
from src.utils_tumor import get_data_fr_paths, get_cocos_from_data_fr,get_advanced_dis_data_fr,format_seg_names
import src.utils_detectron as ud
import src.detec_helper as dh     #读取文件有问题，主要是csv文件格式不清楚。
#from src.utils_detectron import F_KEY, CLASS_KEY, ENTITY_KEY
from src.utils_detectron import F_KEY, CLASS_KEY, ENTITY_KEY
from src.categories import make_cat_advanced, cat_mapping_new, cat_naming_new, reverse_cat_list
from itertools import cycle

# import some common libraries
import numpy as np
import random

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
setup_logger()
cfg = get_cfg()
print(detectron2.__version__)

cfg.OUTPUT_DIR = "./model_2020_T1/model1/"


# General definitions:
mask_it = True                 # Apply Segmentations #不知道这个的意思是不是json文件中（或者数据应该标个框）
simple = True                 # Extinguish between benign and malignant only
advanced = True                # Advanced division between train test and valid set
train = False        # Define whether to train or load a model


# load dataframe and paths
df, paths = get_data_fr_paths()
#df_ex, paths_ex = get_data_fr_paths(mode=True)#格外数据集，先删除，2021/10/5 yang delete
# prepare all datasets to coco format
if train:
    print('remake coco')
    #cocos = get_cocos_from_data_fr(df, paths, save=True, simple=simple)
model_str = dict()
scores = dict()
targets = dict()
# dis = get_advanced_dis_data_fr(df,1)
for folder_index in ['1','2','3','4','5']:
    #cocos,save_file = testt_get_cocos_from_data_fr(df, paths,folder_index, save=True, simple=simple)
    if train:
        print('remake coco')
        #cocos = get_cocos_from_data_fr(df, paths, save=True, simple=simple)

    df, paths = get_data_fr_paths(folder_index)
    dis = get_advanced_dis_data_fr(df, folder_index)
    df.head()
    # Register datasets
    path = os.path.join(os.getcwd())
    pic_path = path+'/PNG' 
    pic_path_external = os.path.join(path, "PNG_external")
    #注册自己的数据集
    print(f'training {folder_index} folder')
    train_json_name= f'train{folder_index}t1.json' 
    valid_json_name= f'valid{folder_index}t1.json'
    test_json_name= f'test{folder_index}t1.json'
    register_coco_instances(      #coco-json格式
        f"my_dataset_train{folder_index}", {}, os.path.join(path, train_json_name), pic_path
    )
    register_coco_instances(
        f"my_dataset_valid{folder_index}", {}, os.path.join(path, valid_json_name), pic_path
    )
    register_coco_instances(
        f"my_dataset_test{folder_index}", {}, os.path.join(
            path, test_json_name), pic_path
    )

    # select the right network
    if mask_it:
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"   ##model_zoo/configs/coco-instances
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )

    cfg.DATALOADER.NUM_WORKERS = 8
    #cfg.INPUT.FORMAT = "F"
    # training parameters
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.GAMMA = 0.1
    # #cfg.SOLVER.IMS_PER_BATCH = 4
    # # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (12000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 800
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

    #cfg.SOLVER.WARMUP_ITERS = 3000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    # pick a good LR
    # cfg.INPUT.MAX_SIZE_TRAIN = 256
    # roi and classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     #可能要改，猜测：改成256

    # set to "RepeatFactorTrainingSampler" in order to allow balanced sampling
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.3
    cfg.SOLVER.MAX_ITER = 30000  # Maximum number of iterations for training

    if advanced:
        _, cat_list = make_cat_advanced(simple=simple)#可能跟几分类有关
    else:
        _, cat_list = make_cat_advanced(simple=simple)

    #num_classes = len(list(cat_list)) if simple else len(list(cat_list.keys()))

    num_classes=2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
       2 if simple else num_classes  # former 5
    )  # we have the malign and benign class / all 5 classes

    # Select Trainer
    if train:
          os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
          trainer = ud.CocoTrainer(cfg)
          trainer.resume_or_load(resume=False)
          trainer.train()

    cfg.OUTPUT_DIR = "./model_2020_T1/"

    path = "./model_2020_T1/"
    cfg.OUTPUT_DIR = path+'model'+folder_index+"/"
    print(f'testing {folder_index} folder')
  
    model_str[1] = "model_0012499.pth"
    model_str[2] = "model_0027499.pth"
    model_str[3] = "model_0010499.pth"
    model_str[4] = "model_0014499.pth"
    model_str[5] = "model_0012499.pth"
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_str[int(folder_index)])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    # cfg.DATASETS.TEST = (f"my_dataset_test{folder_index}",)
    predictor = DefaultPredictor(cfg)
    res = ud.yang_personal_score(predictor, df, folder_index, simple=simple, mode='test', external=False)
    #ud.plot_confusion_matrix(res['conf'], ['ganran', 'tumor'] if simple else reverse_cat_list)
    scores[int(folder_index)] = res["scores_probaility"]
    dh.folder_generate_all_images(predictor,folder_index, external=False)
    iou_mask, dice_mask, iou_bb, dice_bb = dh.get_iou_masks(predictor, folder_index, external=False)



