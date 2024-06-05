import cv2
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from mmcv.ops import batched_nms

# # 放缩尺度
# def scale_down_detections(pred_instances, scale_factor, original_shape):
#     """Scale down detections to the original image size."""
#     pred_instances.bboxes /= scale_factor
#     if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
#         scaled_masks = []
#         for mask in pred_instances.masks:
#             mask_tensor = mask.float().unsqueeze(0).unsqueeze(0)
#             scaled_mask = F.interpolate(mask_tensor, size=original_shape[:2], mode='bilinear', align_corners=False)
#             scaled_masks.append((scaled_mask.squeeze() > 0.5))
#         pred_instances.masks = scaled_masks

#     return pred_instances


# class DetectionResults:
#     def __init__(self, bboxes, labels, scores, masks=None):
#         self.bboxes = bboxes
#         self.labels = labels
#         self.scores = scores
#         self.masks = masks

# 合并所有的tta的推理的结果
# def merge_tta_results(tta_results,iou_threshold=0.2,top_n=None):
#     # Prepare to collect all bboxes, scores, labels, and optionally masks
#     all_bboxes = []
#     all_scores = []
#     all_labels = []
#     all_masks = []  # If masks are used, they will be handled similarly

#     # Collect data from all TTA results
#     for result in tta_results:
#         # Ensure the bboxes are in tensor format properly
#         if isinstance(result['bboxes'], torch.Tensor):
#             all_bboxes.append(result['bboxes'])
#         else:
#             all_bboxes.append(torch.tensor(result['bboxes'], dtype=torch.float32))

#         all_scores.append(torch.tensor(result['scores'], dtype=torch.float32) if not isinstance(result['scores'], torch.Tensor) else result['scores'])
#         all_labels.append(torch.tensor(result['labels'], dtype=torch.int64) if not isinstance(result['labels'], torch.Tensor) else result['labels'])

#         if 'masks' in result and result['masks'] is not None:
#             all_masks.extend(result['masks'])  # Assumes masks are already tensors

#     # Concatenate all results into tensors
#     all_bboxes = torch.cat(all_bboxes, dim=0) if len(all_bboxes) else torch.tensor([])
#     all_scores = torch.cat(all_scores, dim=0) if len(all_scores) else torch.tensor([])
#     all_labels = torch.cat(all_labels, dim=0) if len(all_labels) else torch.tensor([])

#     # Prepare NMS config
#     nms_cfg = {'iou_threshold': iou_threshold}

#     # Apply NMS using batched_nms from MMCV
#     _, keep_indices = batched_nms(all_bboxes, all_scores, all_labels, nms_cfg)

#     # Filter results based on NMS
#     final_bboxes = all_bboxes[keep_indices]
#     final_scores = all_scores[keep_indices]
#     final_labels = all_labels[keep_indices]
#     final_masks = [all_masks[i] for i in keep_indices]

#     results = DetectionResults(
#         bboxes=final_bboxes,
#         labels=final_labels,
#         scores=final_scores,
#         masks=final_masks if final_masks else None
#     )

#     return results

from mmengine.structures import InstanceData

def merge_tta_results(tta_results, iou_threshold=0.2):
    merged_instances = InstanceData.cat(tta_results)

    # Prepare NMS config
    nms_cfg = {'iou_threshold': iou_threshold}

    _, keeps = batched_nms(
        boxes=merged_instances.bboxes,
        scores=merged_instances.scores,
        idxs=merged_instances.labels,
        nms_cfg=nms_cfg)
    results = merged_instances[keeps]

    return results


# 旋转系列库

def rotate_image(img, angle):
    """ 使用 OpenCV 旋转图像 """
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    return rotated_img


def rotate_bbox(bbox, rotation_degree, img_width, img_height):
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
    rotated_bboxes = torch.stack([rotate_bbox(
        bbox, angle, img_width, img_height) for bbox in results['bboxes']], dim=0)
    rotated_masks = [rotate_mask(mask, angle) for mask in results['masks']]

    results['bboxes'] = rotated_bboxes
    results['masks'] = rotated_masks
    return results


def filter_by_score(pred_instances, score_threshold):
    """Filter instances by score."""
    keep = pred_instances.scores > score_threshold
    pred_instances = pred_instances[keep]
    return pred_instances

def filter_bboxes_within_bounds(bboxes, start_point, end_point):
    """
    过滤出完全在指定起始点和终点定义的矩形区域内的边界框。
    
    Args:
    - bboxes (Tensor): 边界框，形状为 [N, 4]，其中 N 是边界框数量，每个边界框格式为 [x1, y1, x2, y2]
    - start_point (Tuple[int, int]): 矩形区域的起始点 (x_min, y_min)
    - end_point (Tuple[int, int]): 矩形区域的终点 (x_max, y_max)
    
    Returns:
    - Tensor: 过滤后的边界框索引
    """
    # 解包起始点和终点坐标
    x_min, y_min = start_point
    x_max, y_max = end_point

    # 获取边界框的坐标点
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    # 定义条件，检查每个边界框的坐标是否完全位于指定的矩形区域内
    inside_x = (x1 > x_min) & (x2 < x_max)
    inside_y = (y1 > y_min) & (y2 < y_max)
    
    # 找出同时满足x和y条件的边界框索引
    keep_indices = torch.where(inside_x & inside_y)[0]
    
    return keep_indices

def compute_slice_boundaries(starting_pixels, img_height, img_width, slice_height, slice_width, buffer):
    """
    根据给定的切片起始点和切片参数计算每个切片的局部保留区域。

    Args:
        starting_pixels (list): 切片的起始点列表，每个元素是一个元组 (start_x, start_y)。
        img_height (int): 图像的高度。
        img_width (int): 图像的宽度。
        slice_height (int): 每个切片的高度。
        slice_width (int): 每个切片的宽度。
        overlap (float): 切片之间的重叠比例。
        buffer (int): 需要过滤掉的边界内框的缓冲区大小。

    Returns:
        dict: 字典，键为切片的起始点，值为该切片的保留区域局部坐标。
    """
    slice_boundaries_dict = {}

    for (start_x, start_y) in starting_pixels:
        end_x = min(start_x + slice_width, img_width)
        end_y = min(start_y + slice_height, img_height)

        global_filter_top = start_y + buffer if start_y > 0 else 0
        global_filter_bottom = end_y - buffer if end_y < img_height else img_height
        global_filter_left = start_x + buffer if start_x > 0 else 0
        global_filter_right = end_x - buffer if end_x < img_width else img_width

        local_filter_top = global_filter_top - start_y
        local_filter_bottom = global_filter_bottom - start_y
        local_filter_left = global_filter_left - start_x
        local_filter_right = global_filter_right - start_x

        slice_boundaries_dict[(start_x, start_y)] = {
            'start_point': (local_filter_left, local_filter_top),
            'end_point': (local_filter_right, local_filter_bottom)
        }

    return slice_boundaries_dict