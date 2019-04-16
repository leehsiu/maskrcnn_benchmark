# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import pickle
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
import numpy as np

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def _get_uv_anno_status(anno):
    uv_status = ['dp_x' in obj for obj in anno]
    return uv_status

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # containing at least min_keypoints_per_image
    # uv_status = _get_uv_anno_status(anno)
    # if any(uv_status):
    #     return True
    return True

class DensePoseCOCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file,root, remove_images_without_annotations,proposal_file=None,transforms=None
    ):
        super(DensePoseCOCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        if proposal_file is not None:
            with open(proposal_file,'rb') as fio:
                proposals = pickle.load(fio,encoding='latin1')
                proposals_ids = proposals['ids']
                proposals_boxes = proposals['boxes']
                order = np.argsort(proposals_ids)
                self.proposals_ids = [proposals_ids[i] for i in order]
                self.proposals_boxes = [proposals_boxes[i] for i in order]
            self.has_proposal = True
        else:
            self.proposals_ids = []
            self.proposals_boxes = []
            self.has_proposal = False
        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        print('datasets loaded with {} samples and {} proposals'.format(len(self.ids),len(self.proposals_ids)))
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(DensePoseCOCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        #* Notice that here we only add densepose annotations
        #anno = [obj for obj in anno if obj["iscrowd"] == 0 and 'dp_x' in obj]
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)


        keypoints = [obj["keypoints"] for obj in anno]
        keypoints = PersonKeypoints(keypoints, img.size)
        target.add_field("keypoints", keypoints)

        if self.has_proposal:
            c_ids = self.ids[idx]
            proposal_idx = self.proposals_ids.index(c_ids)
            proposal_box = self.proposals_boxes[proposal_idx]
            proposal_box = torch.as_tensor(proposal_box).reshape(-1, 4)  # guard against no boxes
            #proposal_bbox = BoxList(proposal_box, img.size, mode="xyxy").convert("xyxy")
            proposal_bbox = BoxList(proposal_box, img.size, mode="xyxy")
            proposal_bbox.clip_to_image(remove_empty=True)
        else:
            proposal_bbox = None

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img,target,proposal_bbox = self.transforms(img,target,proposal_bbox)
        return img,target,proposal_bbox,idx
        
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
