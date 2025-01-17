import subprocess
from datetime import date
import logging
import yaml
import os
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# mounted locations in the docker container
HYP_YAML = '/app/config/hyp.yaml'

# default configs
DATA_YAML = '/app/config/dataset.yaml'
TRAIN_FOLDER = '/app/training'
VAL_FOLDER = '/app/validation'
PREDICT_FOLDER = '/app/prediction'
MODEL_PATH = '/app/trained-inference-models'
MODEL_NAMES = ['best.engine','best.pt']
SOURCE_PATH = '/app/data'


def check_path_exist(path, is_file:bool):
    """fail the program if the path does not exist

    Args:
        path (str): the input path
        is_file (bool): True if it's a file, False otherwise
    """
    if is_file and not os.path.isfile(path):
        raise Exception(f'Not found file: {path}')
    if not is_file and not os.path.isdir(path):
        raise Exception(f'Not found path: {path}')
    
    
def sanity_check(final_configs:dict, check_keys:dict):
    """check if the value to the check_keys exists. If not, throw exception.

    Args:
        final_configs (dict): the input configs
        check_keys (dict): < key_to_be_checked : True if is_file else False >
    """
    for k,v in check_keys.items():
        check_path_exist(final_configs[k],v)
    
    
def get_model_path(path, mode):
    # if export mode, use 'best.pt'. 
    # otherwise:
    #   use 'best.engine' if it exists. otherwise use 'best.pt'
    # return None if not found any model weights
    names = MODEL_NAMES[1:] if mode=='export' else MODEL_NAMES
    for fname in names:
        p = os.path.join(path, fname)
        if os.path.isfile(p):
            logger.info(f'Use the model weights: {p}')
            return p
    return None


def add_configs(final_configs:dict, configs:dict):
    """add to configs only if the configs do NOT exist. Modify the final_configs in-place.

    Args:
        final_configs (dict): the output configs
        configs (dict): the configs to be added
    """
    for k,v in configs.items():
        if k not in final_configs:
            logger.info(f'Not found the config: {k}. Use the default: {v}')
            final_configs[k] = v



if __name__=='__main__':
    # check if files exist
    check_path_exist(HYP_YAML, True)
        
    # load hyp yaml file
    with open(HYP_YAML) as f:
        hyp = yaml.safe_load(f)
       
    # use today's date as the default output folder name
    today = date.today().strftime("%Y-%m-%d")
    defaults = {}
    
    # add other default configs
    check_keys = {} # map < key : True if is_file else False >
    path_wts = get_model_path(MODEL_PATH, hyp['mode'])
    if hyp['mode'] == 'train':
        os.system("pkill -f tensorboard")
        pid = os.fork()
        if pid == 0:
            os.setsid()
            os.system(f"tensorboard --logdir {TRAIN_FOLDER} --port 6006")
            sys.exit(0)
        else:
            logger.info(f"Tensorboard started with PID {pid}")
        
        tmp = {'data':DATA_YAML, 'project':TRAIN_FOLDER}
        check_keys['data'] = True
    elif hyp['mode'] == 'export':
        tmp = {'model':path_wts}
        check_keys['model'] = True
    elif hyp['mode'] == 'predict':
        tmp = {'model':path_wts, 'source':SOURCE_PATH, 'project':PREDICT_FOLDER}
        check_keys['source'] = False
        check_keys['model'] = True
    elif hyp['mode'] == 'val':
        tmp = {'data':DATA_YAML, 'model':path_wts, 'project':VAL_FOLDER}
        check_keys['data'] = True
        check_keys['model'] = True
    else:
        raise Exception(f"Not support the mode: {hyp['mode']}. All supported modes are: train, val, predict, export.")
    defaults.update(tmp)
    add_configs(hyp, defaults)
    
    # error checking
    sanity_check(hyp, check_keys)
    
    # get final command
    final_cmd = ['yolo'] + [f'{k}={v}' for k, v in hyp.items()] + [f'name="{today}"']
    logger.info(f'cmd: {final_cmd}')
    
    # run final command
    subprocess.run(final_cmd, check=True)
    
