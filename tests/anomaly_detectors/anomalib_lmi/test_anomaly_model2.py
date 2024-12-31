import pytest
import logging
from collections.abc import Sequence
import sys
import os
import tempfile
import glob
import cv2
import numpy as np
import torch
import subprocess
from anomalib.deploy.inferencers.torch_inferencer import TorchInferencer
from anomalib.data.utils import read_image

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))
sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))


from anomalib_lmi.anomaly_model2 import AnomalyModel2
from anomaly_detector import AnomalyDetector


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v1.pt'
OUTPUT_PATH = 'tests/assets/validation/ad_v1'



def test_compare_results_with_anomalib():
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH)
    model2 = AnomalyModel2(MODEL_PATH)
    paths = glob.glob(os.path.join(DATA_PATH, '*.png'))
    for p in paths:
        # using anomalib code
        tensor = read_image(p,as_tensor=True)
        pred = model1.forward(model1.pre_process(tensor))
        if isinstance(pred, dict):
            pred = pred['anomaly_map']
        elif isinstance(pred, Sequence):
            pred = pred[1]
        elif isinstance(pred, torch.Tensor):
            pass
        else:
            raise Exception(f'Not supported output: {type(pred)}')
        pred = pred.cpu().numpy().squeeze()
        
        # using AIS code
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pred2 = model2.predict(rgb)
        
        assert np.array_equal(pred, pred2)

def test_compare_results_with_anomalib_api():
    """
    compare prediction results between current implementation and anomalib
    """
    model1 = TorchInferencer(MODEL_PATH)
    model2 = AnomalyDetector(dict(framework='anomalib', model_name='padim', task='seg', version='v1'), MODEL_PATH)
    paths = glob.glob(os.path.join(DATA_PATH, '*.png'))
    for p in paths:
        # using anomalib code
        tensor = read_image(p,as_tensor=True)
        pred = model1.forward(model1.pre_process(tensor))
        if isinstance(pred, dict):
            pred = pred['anomaly_map']
        elif isinstance(pred, Sequence):
            pred = pred[1]
        elif isinstance(pred, torch.Tensor):
            pass
        else:
            raise Exception(f'Not supported output: {type(pred)}')
        pred = pred.cpu().numpy().squeeze()
        
        # using AIS code
        im = cv2.imread(p)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        pred2 = model2.predict(rgb)
        
        assert np.array_equal(pred, pred2)
        
        
def test_warmup():
    ad = AnomalyModel2(MODEL_PATH,224,112)
    ad.warmup()
    ad.warmup([672,640])
    
    ad = AnomalyModel2(MODEL_PATH)
    ad.warmup()
    ad.warmup([256,224])

def test_warmup_api():
    ad = AnomalyDetector(dict(framework='anomalib', model_name='padim', task='seg', version='v1'), MODEL_PATH,224,112)
    ad.warmup()
    ad.warmup([672,640])
    
    ad = AnomalyDetector(dict(framework='anomalib', model_name='padim', task='seg', version='v1'), MODEL_PATH,224,112)
    ad.warmup()
    ad.warmup([256,224])
    

def test_model():
    ad = AnomalyModel2(MODEL_PATH,224,224,'resize')
    ad.test(DATA_PATH, OUTPUT_PATH)
    
    ad = AnomalyModel2(MODEL_PATH)
    ad.test(DATA_PATH, OUTPUT_PATH)
    
def test_model_api():
    ad = AnomalyDetector(dict(framework='anomalib', model_name='padim', version='v1'), MODEL_PATH,224,224,'resize')
    ad.test(DATA_PATH, OUTPUT_PATH)
    
    ad = AnomalyDetector(dict(framework='anomalib', model_name='padim', version='v1'), MODEL_PATH)
    ad.test(DATA_PATH, OUTPUT_PATH)
    
    
def test_cmds():
    """test model inference and model to tensorrt conversion
    """
    with tempfile.TemporaryDirectory() as t:
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = f'$PYTHONPATH:{ROOT}/lmi_utils:{ROOT}/anomaly_detectors'
        cmd = f'python -m anomalib_lmi.anomaly_model2 test -i {MODEL_PATH} -d {DATA_PATH} -o {str(t)} -g -p --tile 224 224 --stride 224 224 --resize'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
        
        l1 = glob.glob(os.path.join(DATA_PATH, '*.png'))
        l2 = glob.glob(os.path.join(t, '*_annot.png'))
        assert len(l1) == len(l2)
        
        t2 = os.path.join(t,'recon')
        cmd = f'python -m anomalib_lmi.anomaly_model2 convert -i {MODEL_PATH} -o {t2} --hw 1120 1120 --tile 224 224 --stride 224 224'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
        
        assert os.path.isfile(os.path.join(t2, 'model.engine'))
        