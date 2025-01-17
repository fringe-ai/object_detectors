#built-in packages
import csv
import collections
from logging import warning
import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#LMI packages
from label_utils.shapes import Rect, Mask, Keypoint, Brush


def load_csv(fname:str, path_img:str='', class_map:dict=None, zero_index:bool=True):
    """
    load csv file into a dictionary mapping <image_name, a list of mask objects>
    Arguments:
        fname(str): the input csv file name
        path_img(str): the path to the image folder where its images should be listed in the csv file, default is ''
        class_map(dict): map <class, class ID>, default is None
	    zero_index(bool): it's used when class_map is None. whether the class ID is 0 or 1 indexed, default is True
    Return:
        shapes(dict): a dictionary maps <image_name, a list of Mask or Rect objects>
	    class_map(dict): <classname, ID> where IDs are 0-indexed if zero_index is true else 1-indexed
    """
    shapes = collections.defaultdict(list)
    if class_map is None:
        new_map = True
        class_map = {}
        idx = 0 if zero_index else 1
    else:
        new_map = False
        idx = max(class_map.values())+1

    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            im_name = row[0]
            category = row[1]
            try:
                # expect to find confidence level
                confidence = float(row[2])
                shape_type = row[3]
                coord_type = row[4]
                coordinates = row[5:]
            except Exception:
                # in case cannot find confidence level, set it to 1.0
                confidence = 1.0
                shape_type = row[2]
                coord_type = row[3]
                coordinates = row[4:]
            
            if category not in class_map:
                if not new_map:
                    warning(f'found new class in the {fname}: {category}, skip')
                    continue
                else:
                    #warning(f'found new class in the {fname}: {category}, add to class_map')
                    class_map[category] = idx
                    idx += 1
            fullpath = os.path.join(path_img,im_name) if path_img else ''
            if shape_type=='polygon':
                if coord_type=='x values':
                    M = Mask(im_name=im_name, fullpath=fullpath, category=category, confidence=confidence)
                    M.X = list(map(float,coordinates))
                elif coord_type=='y values':
                    assert(im_name==M.im_name)
                    M.Y = list(map(float,coordinates))
                    shapes[im_name].append(M)
                else:
                    raise Exception(f"invalid keywords: {coord_type}")
            elif shape_type=='brush':
                if coord_type=='x values':
                    B = Brush(im_name=im_name, fullpath=fullpath, category=category, confidence=confidence)
                    B.X = list(map(float,coordinates))
                elif coord_type=='y values':
                    assert(im_name==B.im_name)
                    B.Y = list(map(float,coordinates))
                    shapes[im_name].append(B)
            elif shape_type=='rect':
                if coord_type=='upper left':
                    R = Rect(im_name=im_name, fullpath=fullpath, category=category, confidence=confidence)
                    R.up_left = list(map(float,coordinates[:2]))
                    # handle angle if it exists
                    if len(coordinates)==4:
                        R.angle = float(coordinates[-1])
                elif coord_type=='lower right':
                    assert(im_name==R.im_name)
                    R.bottom_right = list(map(float,coordinates[:2]))
                    shapes[im_name].append(R)
                else:
                    raise Exception(f"invalid keywords: {coord_type}")
            elif shape_type=='keypoint':
                if coord_type=='x value':
                    K = Keypoint(im_name=im_name, fullpath=fullpath, category=category, confidence=confidence)
                    K.x = float(coordinates[0])
                elif coord_type=='y value':
                    assert(im_name==K.im_name)
                    K.y = float(coordinates[0])
                    shapes[im_name].append(K)
                else:
                    raise Exception(f"invalid keywords: {coord_type}")
    return shapes, class_map


def write_to_csv(shapes:dict, filename:str, overwrite=True):
    """
    write a dictionary of list of shapes into a csv file
    Arguments:
        shape(dict): a dictionary maps the filename to a list of Mask or Rect objects, i.e., <filename, list of Mask or Rect>
        filename(str): the output csv filename
        overwrite(bool): whether to overwrite the file if it exists, default is True
    """
    with open(filename, 'a+' if not overwrite else 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        for im_name in shapes:
            for shape in shapes[im_name]:
                if not isinstance(shape, (Rect, Mask, Keypoint, Brush)):
                    raise Exception(f"Found not supported class: {type(shape)}. Supported classes are Mask, Rect, Keypoint")
                # round the coordinates and convert to list
                shape.round()
                if isinstance(shape, Rect):
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'rect', 'upper left'] + shape.up_left + ['angle', f'{shape.angle:.2f}'])
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'rect', 'lower right'] + shape.bottom_right+ ['angle', f'{shape.angle:.2f}'])
                elif isinstance(shape, Mask):
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'polygon', 'x values'] + shape.X)
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'polygon', 'y values'] + shape.Y)
                elif isinstance(shape, Keypoint):
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'keypoint', 'x value', f'{shape.x}'])
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'keypoint', 'y value', f'{shape.y}'])
                elif isinstance(shape, Brush):
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'brush', 'x values'] + shape.X)
                    writer.writerow([shape.im_name, shape.category, f'{shape.confidence:.4f}', 'brush', 'y values'] + shape.Y)
                    

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='input csv file')
    ap.add_argument('-o', '--output', required=True, help='output csv file')
    args = ap.parse_args()

    shapes, class_map = load_csv(args.input)
    write_to_csv(shapes, args.output)
    logger.info(f'loaded {len(shapes)} shapes and saved to {args.output}')
