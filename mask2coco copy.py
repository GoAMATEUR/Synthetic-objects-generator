import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from glob import glob
import cv2
import shutil
from configs import CATEGORIES

IMAGE_SRC = r'./dataset/raw_images/' # Path of raw images
ROOT_DIR = r'./dataset' # Output path
IMAGE_DIR = os.path.join(ROOT_DIR, "images") # Output path of images
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations") # output path of annotations

# Dataset Info
INFO = {
    "description": "Synthetic Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "1.0.0",
    "year": 2023,
    "contributor": "Siyuan_Huang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# Define categories


# id2superclass = {}
# superclass2id = dict(zip(id2superclass.values(), id2superclass.keys())) # (key, value) = (id, name)
id2subclass = {}
subclass2id = dict(zip(id2subclass.values(), id2subclass.keys())) # (key, value) = (id, name)

# def getCategories():
#     image_files = glob(IMAGE_SRC + "*.jpg")
#     subClassList = []
#     temp = []
#     for image in image_files:
#         image_name = os.path.basename(image).split('.')[0]
#         try:
#             _, superClass, subClass, imageNumber = image_name.split('-')
#         except:
#             print("Invalid image name", image_name)
#             continue

#         if not os.path.exists(IMAGE_DIR+"/"+str(imageNumber)+".jpg"):
#             shutil.copy(image, IMAGE_DIR+"/"+str(imageNumber)+".jpg") # Copy image to output path
#         if subClass not in subClassList:
#             subClassList.append(subClass)
#             item = {'id':int(subNumber),  # 强转int类型，很重要！！
#                     'name':subClass,
#                     'supercategory':superClass
#             }
#             temp.append(item)
#     global CATEGORIES
#     CATEGORIES = sorted(temp,key=lambda x: x["id"])

    
 
# def filter_for_jpeg(root, files):
#     file_types = ['*.jpeg', '*.jpg', '*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     return files
 
 
# def filter_for_annotations(root, files, image_filename):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
#     file_name_prefix = basename_no_extension + '_.*'   # 用于匹配对应的二值mask
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
#     return files
 
 
def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }


 
    image_id = 1
    segmentation_id = 1
 
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
 
        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                    image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
 
            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
 
                # go through each associated annotation
                for annotation_filename in annotation_files:
 
                    
                    # class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    class_id = [x['id'] for x in CATEGORIES 
                                if x['name'].upper() == annotation_filename.split('_')[-2].upper()][0]  # 精确匹配类型名

                    print(annotation_filename+" "+str(class_id))
 
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)
 
                    annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_id, image_id, category_info, binary_mask,
                            image.size, tolerance=2)
 
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
 
                    segmentation_id = segmentation_id + 1
 
            image_id = image_id + 1
 
    with open(ROOT_DIR+'/instances_'+part+'2017.json', 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
 
 
if __name__ == "__main__":
    main()
