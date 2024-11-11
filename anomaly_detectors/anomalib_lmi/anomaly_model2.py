import os
import logging 
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
import tensorrt as trt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from .base import Anomalib_Base
from image_utils.tiler import Tiler
import gadget_utils.pipeline_utils as pipeline_utils


logging.basicConfig()


MINIMUM_QUANT=1e-12
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))


class AnomalyModel2(Anomalib_Base):
    '''
    Desc: Class used for AD model inference.
    '''
    logger = logging.getLogger('AnomalyModel v2')
    logger.setLevel(logging.INFO)
    
    def __init__(self, model_path, input_hw=None, tile=None, stride=None):
        """_summary_

        Args:
            model_path (str): the path to the model file, either a pt or trt engine file
            input_hw(int | list, optional), the input image shape [h,w]. Must provide if using tiling.
            tile (int | list, optional): tile size [h,w]. Must provide if using tiling.
            stride (int | list, optional): stride size [h,w]. Must provide if using tiling.
            
        attributes:
            - self.device: device to run model on
            - self.fp16: flag for half precision
            - self.model_shape: model input shape (h,w)
            - self.inference_mode: model inference mode (TRT or PT)
            - self.tiler: tiling object
        """
        if not os.path.isfile(model_path):
            raise Exception(f'Cannot find the model file: {model_path}')
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.logger.warning('GPU device unavailable. Use CPU instead.')
            self.device = torch.device('cpu')
            
        _,ext = os.path.splitext(model_path)
        self.fp16 = False
        self.logger.info(f"Loading model: {model_path}")
        if ext=='.engine':
            with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            self.context = model.create_execution_context()
            self.bindings = OrderedDict()
            self.output_names = []
            for i in range(model.num_bindings):
                name = model.get_tensor_name(i)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                shape = tuple(self.context.get_tensor_shape(name))
                self.logger.info(f'binding {name} ({dtype}) with shape {shape}')
                if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                im = self.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            self.model_shape=list(shape[-2:])
            self.inference_mode='TRT'
        elif ext=='.pt':  
            checkpoint = torch.load(model_path,map_location=self.device)
            self.pt_model = checkpoint['model']
            self.pt_model.eval()
            self.pt_metadata = checkpoint["metadata"]
            self.logger.info(f"Model metadata: {self.pt_metadata}")
            for d in self.pt_model.transform.transforms:
                if isinstance(d, v2.Resize):
                    self.model_shape = d.size
            self.inference_mode='PT'
        else:
            raise Exception(f'Unknown model format: {ext}')
        
        # init tiler
        self.tiler = None
        self.input_hw = None
        if tile is not None and stride is not None:
            if input_hw is None:
                raise Exception('Must provide input_hw using tiling.')
            self.logger.info(f'input hw: {input_hw}')
            self.logger.info(f'init tiler with tile={tile}, stride={stride}')
            self.tiler = Tiler(tile,stride)
            if isinstance(input_hw,int):
                input_hw = [input_hw]*2
            self.input_hw = input_hw
            
            
    
    @torch.inference_mode()
    def preprocess(self, image):
        '''
        Desc: Preprocess input image.
        args:
            - image: numpy array [H,W,Ch]
        '''
        img = self.from_numpy(image).float()
        
        # resize to self.input_hw for tiling
        if self.input_hw is not None:
            h,w = self.input_hw
            img = pipeline_utils.resize_image(img,H=h,W=w)
        
        # grayscale to rgb
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1,1,3)
            
        img = img.permute((2, 0, 1)).unsqueeze(0)
        img = img / 255.0
        
        if self.tiler is not None:
            img = self.tiler.tile(img)
        
        # resize baked into the pt model
        if self.inference_mode=='TRT':
            h,w =  self.model_shape
            img = F.interpolate(img, size=(h,w), mode='bilinear')
        
        img = img.contiguous()
        return img.half() if self.fp16 else img
        
        
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
        if self.inference_mode=='TRT':
            self.binding_addrs['input'] = int(input_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            outputs = {x:self.bindings[x].data for x in self.output_names}
            output = outputs['output']
        elif self.inference_mode=='PT':
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
            output = self.tiler.untile(output)
        
        if isinstance(output, torch.Tensor):
            output = output.cpu().numpy()
        output = np.squeeze(output)
        return output
        

    def warmup(self):
        '''
        Desc: 
            Warm up model using a np zeros array with shape matching model input size.
        Args: 
            input_hw(list, optional): if using tiling, must provide input h,w.
        '''
        if self.tiler is not None:
            zeros = np.zeros(self.input_hw+[3,])
        else:
            zeros = np.zeros(self.model_shape+[3,])
        self.predict(zeros)



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
    ap.add_argument('--hw',type=int,nargs=2,default=None,help='input image shape (h,w). Muse be provided if using tiling')
    ap.add_argument('--tile',type=int,nargs=2,default=None,help='tile size (h,w)')
    ap.add_argument('--stride',type=int,nargs=2,default=None,help='stride size (h,w)')

    args = vars(ap.parse_args())
    action=args['action']
    model_path = args['model_path']
    export_dir = args['export_dir']
    
    ad = AnomalyModel2(model_path,args['hw'],args['tile'],args['stride'])
    
    if action=='convert':
        os.makedirs(export_dir, exist_ok=True)
        ad.convert(model_path,export_dir)
    elif action=='test':
        os.makedirs(args['annot_dir'], exist_ok=True)
        ad.test(args['data_dir'],args['annot_dir'],args['generate_stats'],
                args['plot'],args['ad_threshold'],args['ad_max'])
