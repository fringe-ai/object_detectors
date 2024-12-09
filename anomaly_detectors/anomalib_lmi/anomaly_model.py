import os
import logging
from collections import OrderedDict, namedtuple
import tensorrt as trt
import torch
import numpy as np
import albumentations as A
from torchvision.transforms import v2
import torch.nn.functional as F
from .base import Anomalib_Base, to_list
import gadget_utils.pipeline_utils as pipeline_utils
from collections.abc import Sequence
from image_utils.tiler import Tiler, ScaleMode

Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
         
class AnomalyModelTRT(Anomalib_Base):
    
    def __init__(self, model_path, tile=None, stride=None, tile_mode='padding',version='v1'):
        self.version = version
        
        if not os.path.isfile(model_path):
            raise Exception(f'Cannot find the model file: {model_path}')

        self.logger.info(f"Loading model: {model_path}")
        with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        for i in range(self.model.num_bindings):
            name = self.model.get_tensor_name(i)
            dtype = trt.nptype(self.model.get_tensor_dtype(name))
            shape = tuple(self.context.get_tensor_shape(name))
            self.logger.info(f'binding {name} ({dtype}) with shape {shape}')
            if self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_shape = shape
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)
            im = self.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.model_shape=list(input_shape[-2:])
        self.batch_size = input_shape[0]
        self.inference_mode='TRT'

        if tile is not None:
            if stride is None:
                raise Exception('Must provide stride using tiling')
            
            tile = to_list(tile)
            if self.model_shape != tile:
                raise Exception(f'tile shape {tile} mismatch with model expected shape: {self.model_shape}')
            
            self.tiler = Tiler(tile,stride)
            self.tile_mode = ScaleMode.PADDING if tile_mode=='padding' else ScaleMode.INTERPOLATION
            self.logger.info(f'init tiler with tile={tile}, stride={stride}, mode={self.tile_mode}')

    @torch.inference_mode()
    def predict(self, image):
        '''
        Desc: Model prediction
        Args: image: numpy array [H,W,Ch]

        Note: predict calls the preprocess method
        returns:
            - output: resized output to match training data's size
        '''
        input_batch = self.preprocess(image)
        self.binding_addrs['input'] = int(input_batch.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        outputs = {x: self.bindings[x].data for x in self.output_names}
        output = outputs['output']

        if self.tiler is not None:
            output = self.tiler.untile(output, self.tile_mode)

        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        output = np.squeeze(output)
        return output


class AnomalyModelPT(Anomalib_Base):

    def __init__(self,  model_path, tile=None, stride=None, tile_mode='padding', version='v1'):
        if not os.path.isfile(model_path):
            raise Exception(f'Cannot find the model file: {model_path}')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
        
        self.version = version
        self.loaded = False
        try:
            model = torch.load(model_path,map_location=self.device)["model"]
            model.eval()
            self.pt_model=model.to(self.device)
            self.pt_metadata = torch.load(model_path, map_location=self.device)["metadata"] if model_path else {}
            self.logger.info(f"Model metadata: {self.pt_metadata}")
            if "transform" in self.pt_metadata:
                self.pt_transform=A.from_dict(self.pt_metadata["transform"])
                self.version = 'v0'
            else:
                self.pt_transform = None
                self.version = 'v1'
            
            if self.version != version:
                self.logger.warning(f"Model version updated {self.version}")

            if self.version == 'v0':
                for d in self.pt_metadata['transform']['transform']['transforms']:
                    if d['__class_fullname__']=='Resize':
                        self.model_shape = [d['height'], d['width']]
            elif self.version == 'v1':
                for d in self.pt_model.transform.transforms:
                    if isinstance(d, v2.Resize):
                        self.model_shape = to_list(d.size)
            self.inference_mode='PT'
            self.loaded = True
        except Exception as e:
            self.logger.exception(
                f"Unable to load pt model using v0 and v1 {e}")
        
        try:
            if not self.loaded:
                self.pt_model = torch.jit.load(model_path)
                self.model_shape = [224, 224]
                self.inference_mode='TS'
        except Exception as e:
            self.logger.exception(
                f"Unable to load pt model using torchscript {e}")
            self.loaded = False
        
        if tile is not None:
            if stride is None:
                raise Exception('Must provide stride using tiling')
            
            tile = to_list(tile)
            if self.model_shape != tile:
                raise Exception(f'tile shape {tile} mismatch with model expected shape: {self.model_shape}')
            
            self.tiler = Tiler(tile,stride)
            self.tile_mode = ScaleMode.PADDING if tile_mode=='padding' else ScaleMode.INTERPOLATION
            self.logger.info(f'init tiler with tile={tile}, stride={stride}, mode={self.tile_mode}')
    
    @torch.inference_mode()
    def predict(self, image):
        '''
        Desc: Model prediction 
        Args: image: numpy array [H,W,Ch]
        
        Note: predict calls the preprocess method
        returns:
            - output: resized output to match training data's size
        '''
        input_batch = self.preprocess(image)
        if self.version == 'v0':
            output=self.pt_model(input_batch)[0].cpu().numpy()
        else:
            preds = self.pt_model(input_batch)
            if isinstance(preds, torch.Tensor):
                output = preds
            elif isinstance(preds, dict):
                output = preds['anomaly_map']
            elif isinstance(preds, Sequence):
                output = preds[1]
            else:
                raise Exception(f'Unknown prediction type: {type(preds)}')

            if self.tiler is not None:
                output = self.tiler.untile(output,self.tile_mode)
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
        output = np.squeeze(output)
        return output


class AnomalyModel(Anomalib_Base):
    def __new__(cls,model_path, tile=None, stride=None, tile_mode='padding',version='v1'):
        if not os.path.isfile(model_path):
            raise Exception(f'Cannot find the model file: {model_path}')
        
        _,ext=os.path.splitext(model_path)
        if ext=='.engine':
            return AnomalyModelTRT(model_path, tile=tile, stride=stride, tile_mode=tile_mode,version=version)
            
        else:
            return AnomalyModelPT(model_path, tile=tile, stride=stride, tile_mode=tile_mode, version=version)

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-a','--action', default="test", nargs='?', choices=['convert','test'], help='Action modes')
    ap.add_argument('-i','--model_path', default="/app/model/model.pt", help='Input model file path.')
    ap.add_argument('-e','--export_dir', default="/app/export")
    ap.add_argument('-d','--data_dir', default="/app/data", help='Data file directory.')
    ap.add_argument('-o','--annot_dir', default="/app/annotation_results", help='Annot file directory.')
    ap.add_argument('-g','--generate_stats', action='store_true',help='generate the data stats')
    ap.add_argument('-p','--plot',action='store_true', help='plot the annotated images')
    ap.add_argument('-t','--ad_threshold',type=float,default=None,help='AD patch threshold.')
    ap.add_argument('-m','--ad_max',type=float,default=None,help='AD patch max anomaly.')
    ap.add_argument('-v','--version',type=str,default=None,help='Anomalib version v0 or v1.')

    args = vars(ap.parse_args())
    action=args['action']
    model_path = args['model_path']
    export_dir = args['export_dir']
    
    ad = AnomalyModel(model_path)
    
    if action=='convert':
        os.makedirs(export_dir,exist_ok=True)
        ad.convert(model_path,export_dir)

    if action=='test':
        os.makedirs(args['annot_dir'],exist_ok=True)
        ad.test(args['data_dir'],args['annot_dir'],args['generate_stats'],
                args['plot'],args['ad_threshold'],args['ad_max'])