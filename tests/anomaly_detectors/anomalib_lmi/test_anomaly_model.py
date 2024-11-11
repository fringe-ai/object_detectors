import logging
import sys
import os
import tempfile

# add path to the repo
PATH = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PATH))))
sys.path.append(os.path.join(ROOT, 'lmi_utils'))
sys.path.append(os.path.join(ROOT, 'anomaly_detectors'))


from anomalib_lmi.anomaly_model import AnomalyModel


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DATA_PATH = 'tests/assets/images/nvtec-ad'
MODEL_PATH = 'tests/assets/models/ad/model_v0.pt'
OUTPUT_PATH = 'tests/assets/validation/ad_v0'



def test_model():
    ad = AnomalyModel(MODEL_PATH)
    ad.test(DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True)
    
    
def test_convert():
    with tempfile.TemporaryDirectory() as t:
        ad = AnomalyModel(MODEL_PATH)
        ad.convert(MODEL_PATH,t)
        
        ad2 = AnomalyModel(os.path.join(t,'model.engine'))
        ad2.test(DATA_PATH, OUTPUT_PATH, generate_stats=True,annotate_inputs=True)
    