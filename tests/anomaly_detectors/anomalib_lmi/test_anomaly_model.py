import logging
import sys
import os
import tempfile
import subprocess
from pathlib import Path
import cv2

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))
sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))


# from anomalib_lmi.anomaly_model1 import AnomalyModel
from anomalib_lmi.anomaly_model import AnomalyModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v1.pt'
OUTPUT_PATH = 'tests/assets/validation/ad_v1'



def test_load_pt_model_v0():
    ad = AnomalyModel(MODEL_PATH, version='v0')

    ad.test(DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True)
        
        
def test_cmds():
    with tempfile.TemporaryDirectory() as t:
        my_env = os.environ.copy()
        my_env['PYTHONPATH'] = f'$PYTHONPATH:{ROOT}/lmi_utils:{ROOT}/anomaly_detectors'
        cmd = f'python -m anomalib_lmi.anomaly_model -i {MODEL_PATH} -d {DATA_PATH} -o {str(t)} -g -p'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
        
        cmd = f'python -m anomalib_lmi.anomaly_model -a convert -i {MODEL_PATH} -e {str(t)}'
        logger.info(f'running cmd: {cmd}')
        result = subprocess.run(cmd,shell=True,env=my_env,capture_output=True,text=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    