import logging
import sys
import os
import tempfile

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))
sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))


import gadget_utils.pipeline_utils as pipeline_utils
from anomalib_lmi.anomaly_model2 import AnomalyModel2


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v1.pt'
OUTPUT_PATH = 'tests/assets/validation'



def test_model():
    AnomalyModel2.test(
        MODEL_PATH, DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True,
    )
    
    
def test_convert():
    with tempfile.TemporaryDirectory() as t:
        AnomalyModel2.convert(MODEL_PATH,t,fp16=True)
        AnomalyModel2.test(
            os.path.join(t,'model.engine'), DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True,
        )
    