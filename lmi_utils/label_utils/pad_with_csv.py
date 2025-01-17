import cv2
import os
import argparse
import numpy as np
import logging

#LMI packages
from label_utils.shapes import Rect, Mask, Keypoint, Brush
from label_utils.csv_utils import load_csv, write_to_csv
from gadget_utils.pipeline_utils import fit_array_to_size
from system_utils.path_utils import get_relative_paths


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pad_image_with_csv(input_path, csv_path, output_path, output_imsize, save_bg_images, append_to_csv, recursive):
    """
    pad/crop the image to the size [W,H] and modify its annotations accordingly
    arguments:
        input_path(str): the input image path
        csv_path(str): the path to the csv annotation file
        output_imsize(list): the width and height of the output image
    """
    if not os.path.isdir(input_path):
        raise Exception(f'input image folder does not exist: {input_path}')
    fname_to_shape, _ = load_csv(csv_path, input_path)

    W,H = output_imsize
    cnt_bg = 0
    cnt_warnings = 0
    output_shapes = {}
    
    files = get_relative_paths(input_path, recursive)
    for f in files:
        p = os.path.join(input_path, f)
        im_name = os.path.basename(f)
        im = cv2.imread(p)
        h,w = im.shape[:2]
        
        # found bg image
        if im_name not in fname_to_shape:
            if not save_bg_images:
                continue
            cnt_bg += 1
            logger.info(f'Input file: {im_name} with size of [{w},{h}] has no labels')
        
        # pad image
        im_out,pad_l,_,pad_t,_ = fit_array_to_size(im,W,H)

        #create output fname and save it
        out_name = os.path.splitext(im_name)[0] + f'_pad_{W}x{H}' + '.png'
        output_file=os.path.join(output_path, out_name)
        logger.info(f'write to: {output_file}')
        cv2.imwrite(output_file,im_out)

        #pad shapes
        shapes = fname_to_shape[im_name]
        fit_shapes_to_size(shapes, pad_l, pad_t)
        shapes,is_warning = clip_shapes(shapes, W, H)
        if is_warning:
            cnt_warnings += 1

        #modify the image name 
        for shape in shapes:
            shape.im_name = out_name
        output_shapes[out_name] = shapes
    if cnt_bg:
        logger.info(f'found {cnt_bg} images with no labels. These images will be used as background training data for YOLO.')
    if cnt_warnings:
        logger.warning(f'found {cnt_warnings} images with labels that is either removed entirely, or chopped to fit the new size')
    output_csv = os.path.join(output_path, "labels.csv")
    write_to_csv(output_shapes, output_csv, overwrite=not append_to_csv)
    

def clip_shapes(shapes, W, H):
    """
    description:
        clip the shapes so that they are fit in the target size [W,H]
    """
    to_del = []
    is_warning = False
    shapes = np.array(shapes)
    for i,shape in enumerate(shapes):
        is_del = 0
        if isinstance(shape, Rect):
            box = np.array(shape.up_left+shape.bottom_right)
            new_box = np.clip(box, a_min=0, a_max=[W,H,W,H])
            shapes[i].up_left = new_box[:2]
            shapes[i].bottom_right = new_box[2:]
            if np.all(new_box==0) or new_box[0]==new_box[2] or new_box[1]==new_box[3]:
                is_del = 1
                logger.warning(f'bbox {box} is outside of the size [{W},{H}]')
            elif (np.any(new_box==W) and np.all(box!=W)) or (np.any(new_box==H) and np.all(box!=H)) \
                    or (np.any(new_box==0) and np.all(box!=0)):
                logger.warning(f'bbox {box} is chopped to fit the size [{W}, {H}]')
                is_warning = True
        elif isinstance(shape, (Mask,Brush)):
            X,Y = np.array(shape.X), np.array(shape.Y)
            new_X = np.clip(X, a_min=0, a_max=W)
            new_Y = np.clip(Y, a_min=0, a_max=H)
            shapes[i].X,shapes[i].Y = new_X,new_Y
            if np.all(new_X==W) or np.all(new_Y==H) or np.all(new_X==0) or np.all(new_Y==0):
                is_del = 1
                logger.warning(f'polygon {[(x,y) for x,y in zip(new_X,new_Y)]} is outside of the size [{W},{H}]')
            elif (np.any(new_X==W) and np.all(X!=W)) or (np.any(new_Y==H) and np.all(Y!=H)) \
                or (np.any(new_X==0) and np.all(X!=0)) or (np.any(new_Y==0) and np.all(Y!=0)):
                logger.warning(f'polygon {[(x,y) for x,y in zip(new_X,new_Y)]} is chopped to fit the size [{W}, {H}]')
                is_warning = True
        elif isinstance(shape, Keypoint):
            x,y = shape.x, shape.y
            if x<0 or x>W or y<0 or y>H:
                is_del = 1
                logger.warning(f'keypoint ({x},{y}) is outside of the size [{W},{H}]')
        if is_del:
            is_warning = True
            to_del.append(i)
            
    new_shapes = np.delete(shapes,to_del,axis=0)
    return new_shapes.tolist(), is_warning
    

def fit_shapes_to_size(shapes, pad_l, pad_t):
    """
    description:
        add the left and top paddings to the shapes, modify in-place
    arguments:
        shapes(list): a list of Shape objects (Rect or Mask)
        pad_l(int): the left paddings
        pad_t(int): the top paddings 
    """
    for shape in shapes:
        if isinstance(shape, Rect):
            shape.up_left[0] += pad_l
            shape.up_left[1] += pad_t
            shape.bottom_right[0] += pad_l
            shape.bottom_right[1] += pad_t
        elif isinstance(shape, (Mask,Brush)):
            shape.X = [v+pad_l for v in shape.X]
            shape.Y = [v+pad_t for v in shape.Y]
        elif isinstance(shape, Keypoint):
            shape.x += pad_l
            shape.y += pad_t
    


if __name__=="__main__":
    ap = argparse.ArgumentParser(description='Pad or crop images with csv to output size.')
    ap.add_argument('--path_imgs', '-i', required=True, help='the path to the images')
    ap.add_argument('--path_csv', default='labels.csv', help='[optinal] the path of a csv file that corresponds to path_imgs, default="labels.csv" in path_imgs')
    ap.add_argument('--path_out','-o', required=True, help='the output path')
    ap.add_argument('--wh', required=True, help='the output image size [w,h], w and h are separated by a comma')
    ap.add_argument('--bg', action='store_true', help='save background images with no labels')
    ap.add_argument('--append', action='store_true', help='append to the existing output csv file')
    ap.add_argument('--recursive', action='store_true', help='search images recursively')
    args = vars(ap.parse_args())

    path_imgs = args['path_imgs']
    path_csv = args['path_csv'] if args['path_csv']!='labels.csv' else os.path.join(path_imgs, args['path_csv'])
    if not os.path.isfile(path_csv):
        raise Exception(f'Not found file: {path_csv}. Please create an empty csv file, if there are no labels.')
    output_path=args['path_out']
    output_imsize = list(map(int,args['wh'].split(',')))

    logger.info(f'output image size: {output_imsize}')
    assert len(output_imsize)==2, 'the output image size must be two ints'
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    pad_image_with_csv(path_imgs, path_csv, output_path, output_imsize, args['bg'], args['append'], args['recursive'])
