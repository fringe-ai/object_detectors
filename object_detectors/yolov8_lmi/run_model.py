import cv2
import logging
import os
import random
import numpy as np
import torch

from yolov8_lmi.model import Yolov8
from gadget_utils.pipeline_utils import plot_one_box, get_img_path_batches


BATCH_SIZE = 1


if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wts_file', required=True, help='the path to the model weights file. The type of supported files are: ".pt" or ".engine"')
    parser.add_argument('-i','--path_imgs', required=True, help='the path to the testing images')
    parser.add_argument('-o','--path_out' , required=True, help='the path to the output folder')
    parser.add_argument('--sz', required=True, nargs=2, type=int, help='the model input size, two numbers: h w')
    parser.add_argument('-c','--confidence',default=0.25,type=float,help='[optional] the confidence for all classes, default=0.25')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.NOTSET)
    
    model = Yolov8(args.wts_file)
    logger = logging.getLogger(__name__)
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)
        
    # warm up
    t1 = time.time()
    model.warmup(args.sz)
    t2 = time.time()
    logger.info(f'warmup input shape: {args.sz}')
    logger.info(f'warmup proc time -> {t2-t1:.4f}')
        
    batches = get_img_path_batches(batch_size=BATCH_SIZE, img_dir=args.path_imgs)
    logger.info(f'loaded {len(batches)} with a batch size of {BATCH_SIZE}')
    for batch in batches:
        for p in batch:
            t1 = time.time()
            
            # inference
            im,im0 = model.load_with_preprocess(p)
            preds = model.forward(im)
            results = model.postprocess(preds,im,im0,args.confidence)

            #annotation
            fname = os.path.basename(p)
            save_path = os.path.join(args.path_out,fname)
            im_out = np.copy(im0)
            for i in range(len(results['boxes'])): # loop through batch
                # uppack results
                boxes,scores,classes = results['boxes'][i],results['scores'][i],results['classes'][i]
                masks = results['masks'][i] if 'masks' in results else None
                segments = results['segments'][i] if 'segments' in results else []
                for j in range(len(boxes)-1,-1,-1): # loop through each box
                    mask = masks[j] if masks is not None else None
                    plot_one_box(boxes[j],im_out,mask,label=f'{classes[j]}: {scores[j]:.2f}')
                    if segments and len(segments[j]):
                        seg = np.array(segments[j]).reshape((-1,1,2)).astype(np.int32)
                        cv2.drawContours(im_out, [seg], -1, (0, 255, 0), 1)
                
            # save output image from RGB to BGR
            cv2.imwrite(save_path,im_out[:,:,::-1])
                
            t2 = time.time()
            logger.info(f'proc time of {fname} -> {t2-t1:.4f}')