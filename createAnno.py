import torch.utils.data
import numpy as np

CLASS_LABEL_MAPPING = dict()

class Augmentation:
    def __init__(self) -> None:
        pass

class SynDatasetParser:
    def __init__(self) -> None:
        self.weight = None
        self.height = None
        self.classes = CLASS_LABEL_MAPPING
        self.aug = Augmentation()
    
    def saveAnnoJson(self, mask:np.ndarray, seg: np.ndarray, object_labels: list, classes: dict):
        """_summary_

        Args:
            mask (np.ndarray): _description_
            seg (np.ndarray): _description_
            objects (list): [object_label_1, object2, object_label_3, ...]
            classes (dict): _description_
        """
        h, w = mask.shape[:2]
        for i in range(len(object_labels)):
            if i not in seg:
                continue # object out of frame
            object_label = object_labels[i]
            object_index = classes[object_label]
            
                
        ## TODO: generate COCO mask annotation coordinates for mask area
        
        
        
        return
        
    
    