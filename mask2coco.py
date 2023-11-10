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
 
part = 'test'
IMAGE_SRC = 'C:/Users/awei/Desktop/rgb2mask/Image_'+part+'/'
ROOT_DIR = 'C:/Users/awei/Desktop/rgb2mask/modify_'+part
IMAGE_DIR = os.path.join(ROOT_DIR, "image")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")
 
INFO = {
    "description": "Leaf Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2017,
    "contributor": "Francis_Liu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# 根据自己的需要添加种类
CATEGORIES = [
    {
        'id': 1,  # 是数字1，不是字符串
        'name': 'leaf',
        'supercategory': 'leaf',
    }
]

# Camouflaged:
# COD10K-CAM-SuperNumber-SuperClass-SubNumber-SubClass-ImageNumber

# Non-Camouflaged:
# COD10K-NonCAM-SuperNumber-SuperClass-SubNumber-SubClass-ImageNumber
# Super_Class_Dictionary = {'1':'Aquatic', '2':'Terrestrial', '3':'Flying', '4':'Amphibian', '5':'Other'}
# Sub_Class_Dictionary = {'1':'batFish','2':'clownFish','3':'crab','4':'crocodile','5':'crocodileFish','6':'fish','7':'flounder',
#              '8':'frogFish','9':'ghostPipefish','10':'leafySeaDragon','11':'octopus','12':'pagurian','13':'pipefish',
#               '14':'scorpionFish','15':'seaHorse','16':'shrimp','17':'slug','18':'starFish','19':'stingaree',
#               '20':'turtle','21':'ant','22':'bug','23':'cat','24':'caterpillar','25':'centipede','26':'chameleon',
#               '27':'cheetah','28':'deer','29':'dog','30':'duck','31':'gecko','32':'giraffe','33':'grouse','34':'human',
#               '35':'kangaroo','36':'leopard','37':'lion','38':'lizard','39':'monkey','40':'rabbit','41':'reccoon',
#               '42':'sciuridae','43':'sheep','44':'snake','45':'spider','46':'stickInsect','47':'tiger','48':'wolf',
#               '49':'worm','50':'bat','51':'bee','52':'beetle','53':'bird','54':'bittern','55':'butterfly','56':'cicada',
#               '57':'dragonfly','58':'frogmouth','59':'grasshopper','60':'heron','61':'katydid','62':'mantis',
#               '63':'mockingbird','64':'moth','65':'owl','66':'owlfly','67':'frog','68':'toad','69':'other'}

def getCategories():
    image_files = glob(IMAGE_SRC + "*.jpg")
    subClassList = []
    temp = []
    for image in image_files:
        
        image_name = os.path.basename(image).split('.')[0]
        try:
            _,type,superNumer,superClass,subNumber,subClass,imageNumber = image_name.split('-')
        except:
            print("NonCAM")
            continue


        if not type=="CAM":
            continue


        if not os.path.exists(IMAGE_DIR+"/"+str(imageNumber)+".jpg"):
            shutil.copy(image, IMAGE_DIR+"/"+str(imageNumber)+".jpg")
        if subClass not in subClassList:
            subClassList.append(subClass)
            item = {'id':int(subNumber),  # 强转int类型，很重要！！
                    'name':subClass,
                    'supercategory':superClass
            }
            temp.append(item)
    global CATEGORIES
    CATEGORIES = sorted(temp,key=lambda x: x["id"])

    
 
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files
 
 
def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_.*'   # 用于匹配对应的二值mask
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
 
 
def main():
    getCategories()
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
                    class_id = [x['id'] for x in CATEGORIES if x['name'].upper() == annotation_filename.split('_')[-2].upper()][0]  # 精确匹配类型名

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
