import argparse
import logging
from pathlib import Path
import os
import collections
import torch
import torchvision

from image_utils.tiler import Tiler, ScaleMode


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


IM_TYPES = ('png','jpg','jpeg','tiff')


@torch.inference_mode()
def __to_tiles(source:Path, dest:Path, tile_hw:list, stride_hw:list, mode=ScaleMode.PADDING):
    img = torchvision.io.read_image(source.as_posix()).unsqueeze(0) # [b,c,h,w]
    
    tiler = Tiler(tile_hw,stride_hw)
    tiles = tiler.tile(img,mode)
    
    # write tile images
    os.makedirs(dest, exist_ok=True)
    for i,tile in enumerate(tiles):
        torchvision.io.write_png(tile, str(dest/(source.stem+f'-t{i}.png')))
    
    # write metadata for image reconstruction
    tiler.write_metadata(dest/(source.stem+'-metadata.pt'))


def to_tiles(source:str, dest:str, tile_hw:list, stride_hw:list, mode=ScaleMode.PADDING, recursive=False):
    
    src_path = Path(source)
    dest_path = Path(dest)
    
    if src_path.is_file():
        __to_tiles(src_path, dest_path, tile_hw, stride_hw, mode)
    elif src_path.is_dir():
        for t in IM_TYPES:
            for file in src_path.rglob(f'*.{t}') if recursive else src_path.glob(f'*.{t}'):
                print(file)
                __to_tiles(file, dest_path, tile_hw, stride_hw, mode)


@torch.inference_mode()
def to_images(source, dest, mode=ScaleMode.PADDING):
    src_path = Path(source)
    dest_path = Path(dest)
    dest_path.mkdir(parents=True,exist_ok=True)
    
    meta_map = {}
    for p in src_path.glob('*.pt'):
        ls = p.stem.split('-')
        if len(ls)==2 and ls[1]=='metadata':
            meta_map[ls[0]] = p
            
    tile_map = collections.defaultdict(list)
    for p in src_path.glob('*.png'):
        ls = p.stem.split('-t')
        if len(ls)==2:
            tile_map[ls[0]] += [(int(ls[1]),p)]
    
    if tile_map.keys() != meta_map.keys():
        raise Exception('tile fnames must equal to metadata fnames')
    
    for fname,ps in tile_map.items():
        # init tiler through loading a metadata.pt
        tiler = Tiler.from_pt(meta_map[fname])
        
        # load tiles
        tiles = torch.zeros(len(ps),tiler.num_channel,*tiler.tile_size,dtype=torch.uint8)
        for i,p in ps:
            im = torchvision.io.read_image(p.as_posix())
            if im.dtype != tiles.dtype:
                raise Exception('Current implement only support uint8 images')
            tiles[i] = im
        
        # save image
        im = tiler.untile(tiles,mode).squeeze()
        torchvision.io.write_png(im, str(dest_path/(fname+'.png')))



if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--option', required=True)
    ap.add_argument('-i','--src', required=True)
    ap.add_argument('-o','--dest', required=True)
    ap.add_argument('--tile_hw', type=list, default=[224,224], help='tile hight and width')
    ap.add_argument('--stride_hw', type=list, default=[224,224], help='stride hight and width')
    ap.add_argument('--recursive', action='store_true', help='load images recursively')
    ap.add_argument('--resize', action='store_true', help='interpolate if it needs to resize images, otherwise pad zeros')
    
    args=ap.parse_args()
    
    mode = ScaleMode.INTERPOLATION if args.resize else ScaleMode.PADDING
    if args.option == 'tile':
        to_tiles(args.src,args.dest,args.tile_hw,args.stride_hw,mode=mode,recursive=args.recursive)
    elif args.option == 'untile':
        to_images(args.src,args.dest,mode)
        