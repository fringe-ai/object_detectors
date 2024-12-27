import os
import sys
import json 
import cv2
import glob
import subprocess

import torch
from torch import Tensor, nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.instantiate import dump_dataclass, instantiate
from detectron2.export import dump_torchscript_IR, scripting_with_instances
from detectron2.export.flatten import TracingAdapter, flatten_to_tuple
from detectron2.export.torchscript_patch import patch_builtin_len
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead
from detectron2.structures import Boxes, Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.testing import (
    assert_instances_allclose,
    convert_scripted_instances,
    get_sample_coco_image,
    random_boxes,
    reload_script_model,
    skipIfOnCPUCI,
)
import logging
import numpy as np


PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))
sys.path.append(os.path.join(ROOT, 'object_detectors'))

from detectron2_lmi.model import Detectron2Model

with open('tests/assets/images/coco/class_names.txt','r') as f:
    classnames = f.readlines()

# create a class map "0":"class1", "1":"class2", ...
class_map = {str(i): classnames[i].strip() for i in range(len(classnames))}    

with open('tests/assets/models/od/detectron2/class_map.json','w') as f:
    json.dump(class_map, f)

MASKRCNN_MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
COCO_CLASSMAP = 'tests/assets/models/od/detectron2/class_map.json'
MODEL_PATH = 'tests/assets/models/od/detectron2/model.pt'
SAMPLE_IMAGE = 'tests/assets/images/coco/sample_image.jpg'
OUT_DIR = 'tests/assets/validation'

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestDetectron2ModelPT:

    def test_model(self):
        og_model = model_zoo.get(MASKRCNN_MODEL_CONFIG, trained=True)
        og_model.eval()
        # image = cv2.imread('tests/assets/images/coco/sample_image.jpg')
        img = get_sample_coco_image()
        inputs = [{"image": img}]
        with torch.no_grad():
            orginal_preds = og_model.inference(inputs, do_postprocess=False)[0]
            
        confs = {
           v:0.00 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        preds = model.predict(image, confs=confs, process_masks=False)
        assert orginal_preds.pred_boxes.tensor.shape == preds.get('boxes')[0].shape
        assert orginal_preds.pred_classes.shape == preds.get('classes')[0].shape
        assert orginal_preds.scores.shape == preds.get('scores')[0].shape
        assert orginal_preds.pred_masks.shape == preds.get('masks')[0].shape
        
        # check if the scores are all close
        assert np.allclose(orginal_preds.scores.cpu().numpy(), preds.get('scores'))
        assert np.allclose(orginal_preds.pred_masks.cpu().numpy(), preds.get('masks'))
        
    def test_annotations(self):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        
        annotated_image = model.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
        
    def test_operators(self):
        confs = {
           v:0.95 for k,v in class_map.items()
        }
        model = Detectron2Model(MODEL_PATH, class_map)
        image = cv2.imread(SAMPLE_IMAGE)
        image = cv2.resize(image, (512, 512))
        operators = [{'resize': [1024,1024,512,512]}]
        outputs = model.predict(image, confs=confs, return_segments=True, process_masks=True, operators=operators)
        outputs['boxes'] = outputs['boxes'][0]
        outputs['classes'] = outputs['classes'][0]
        outputs['scores'] = outputs['scores'][0]
        outputs['masks'] = outputs['masks'][0]
        outputs['segments'] = outputs['segments'][0]
        logger.debug(f"Boxes: {outputs['masks'].shape}")
        assert len(outputs['boxes']) == len(outputs['classes']) == len(outputs['scores']) == len(outputs['masks']) == len(outputs['segments'])
        assert outputs['masks'].shape[1] == 512
        assert outputs['masks'].shape[2] == 512
        
        annotated_image = model.annotate_image(
           outputs, image, show_segments=True
        )
        cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(SAMPLE_IMAGE)), annotated_image)
        
        
        