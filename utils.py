import cv2
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from mmcv.ops import batched_nms

# 放缩尺度
def scale_down_detections(pred_instances, scale_factor, original_shape):
    """Scale down detections to the original image size."""
    pred_instances.bboxes /= scale_factor
    if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
        scaled_masks = []
        for mask in pred_instances.masks:
            mask_tensor = mask.float().unsqueeze(0).unsqueeze(0)
            scaled_mask = F.interpolate(mask_tensor, size=original_shape[:2], mode='bilinear', align_corners=False)
            scaled_masks.append((scaled_mask.squeeze() > 0.5))
        pred_instances.masks = scaled_masks

    return pred_instances

class DetectionResults:
    def __init__(self, bboxes, labels, scores, masks=None):
        self.bboxes = bboxes
        self.labels = labels
        self.scores = scores
        self.masks = masks

# 合并所有的tta的推理的结果
def merge_tta_results(tta_results,iou_threshold=0.2):
    # Prepare to collect all bboxes, scores, labels, and optionally masks
    all_bboxes = []
    all_scores = []
    all_labels = []
    all_masks = []  # If masks are used, they will be handled similarly

    # Collect data from all TTA results
    for result in tta_results:
        # Ensure the bboxes are in tensor format properly
        if isinstance(result['bboxes'], torch.Tensor):
            all_bboxes.append(result['bboxes'])
        else:
            all_bboxes.append(torch.tensor(result['bboxes'], dtype=torch.float32))
        
        all_scores.append(torch.tensor(result['scores'], dtype=torch.float32) if not isinstance(result['scores'], torch.Tensor) else result['scores'])
        all_labels.append(torch.tensor(result['labels'], dtype=torch.int64) if not isinstance(result['labels'], torch.Tensor) else result['labels'])
        
        if 'masks' in result and result['masks'] is not None:
            all_masks.extend(result['masks'])  # Assumes masks are already tensors

    # Concatenate all results into tensors
    all_bboxes = torch.cat(all_bboxes, dim=0) if len(all_bboxes) else torch.tensor([])
    all_scores = torch.cat(all_scores, dim=0) if len(all_scores) else torch.tensor([])
    all_labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.tensor([])

    # Prepare NMS config
    nms_cfg = {'iou_threshold': iou_threshold}

    # Apply NMS using batched_nms from MMCV
    _, keep_indices = batched_nms(all_bboxes, all_scores, all_labels, nms_cfg)

    # Filter results based on NMS
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]
    final_masks = [all_masks[i] for i in keep_indices] if all_masks else []

    results = DetectionResults(
        bboxes=final_bboxes,
        labels=final_labels,
        scores=final_scores,
        masks=final_masks if final_masks else None
    )

    return results

# 旋转系列库

def rotate_image(img, angle):
    """ 使用 OpenCV 旋转图像 """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    return rotated_img

def rotate_bbox(bbox,rotation_degree, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox

    if rotation_degree == 90 or rotation_degree == -270:
        new_xmin = ymin
        new_ymin = img_width - xmax
        new_xmax = ymax
        new_ymax = img_width - xmin
    elif rotation_degree == 180 or rotation_degree == -180:
        new_xmin = img_width - xmax
        new_ymin = img_height - ymax
        new_xmax = img_width - xmin
        new_ymax = img_height - ymin
    elif rotation_degree == 270 or rotation_degree == -90:
        new_xmin = img_height - ymax
        new_ymin = xmin
        new_xmax = img_height - ymin
        new_ymax = xmax
    else:
        return bbox

    return torch.tensor([new_xmin, new_ymin, new_xmax, new_ymax], device=bbox.device)

def rotate_mask(mask, angle):
    """ Rotate the mask using PyTorch. """
    if mask.dim() == 2:  # Add channel dimension if not present
        mask = mask.unsqueeze(0)
    return TF.rotate(mask, angle).squeeze(0)

def rotate_results_back(results, angle, img_width, img_height):
    """Rotate back detection results to original orientation using PyTorch."""
    rotated_bboxes = torch.stack([rotate_bbox(bbox, angle, img_width, img_height) for bbox in results['bboxes']], dim=0)
    rotated_masks = [rotate_mask(mask, angle) for mask in results['masks']]
    
    results['bboxes'] = rotated_bboxes
    results['masks'] = rotated_masks
    return results

def filter_by_score(pred_instances, score_threshold):
    """Filter instances by score."""
    keep = pred_instances.scores > score_threshold
    pred_instances = pred_instances[keep]
    return pred_instances
