import json

import os
import json
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector
from pycocotools.mask import encode

from process_ import (
    infer_one_image,
    infer_one_tta,
    infer_slice,
    infer_slice_tta
)

# 初始化模型
config_file = r'C:\Users\Mikky\Desktop\树冠数据集\USIS10K\work_dirs\USIS10KDataset_2\base\multiclass_usis_train.py'
checkpoint_file = r'C:\Users\Mikky\Desktop\树冠数据集\USIS10K\work_dirs\USIS10KDataset_2\base\epoch_30.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 图片文件夹路径
folder_path = r'C:\Users\Mikky\Desktop\树冠数据集\第二次数据集\Testing_set_phase1\Testing_set\images'

# 读取映射文件
mapping_file_path = r'C:\Users\Mikky\Desktop\树冠数据集\第二次数据集\Testing_set_phase1\Testing_set\image_ids\validation_img_id.json'

results_save_path = r'C:\Users\Mikky\Desktop\树冠数据集\results_save.json'

# 加载图片ID映射
with open(mapping_file_path, 'r') as file:
    data = json.load(file)
file_to_id = {img['file_name']: img['id'] for img in data['images']}

# # 加载预存结果
# with open(results_save_path, 'r') as file:
#     saved_results = json.load(file)
# saved_results_map = {result['image_id']: result for result in saved_results}

# Load saved results and create a map of image_id to list of results
with open(results_save_path, 'r') as file:
    saved_results = json.load(file)
saved_results_map = {}
for result in saved_results:
    image_id = result['image_id']  # Ensure the key is a string if necessary
    if image_id not in saved_results_map:
        saved_results_map[image_id] = []
    saved_results_map[image_id].append(result)

# 定义不同数据集的处理方法
scale_factors = {
    'Dataset_3': (1.5, "slice-tta"),
    'Dataset_4': (3, "slice-tta"),
    'Dataset_5': (1, "one-tta")
}

# 结果列表
results_list = []

# 遍历图片
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
for img_path in tqdm(image_files, desc="Processing Images"):
    img_file_name = os.path.basename(img_path)
    img_id = file_to_id.get(img_file_name, -1)
    
    # 检查是否使用预存结果
    if img_id in saved_results_map and img_file_name.startswith('Dataset_4'):
        results_list.extend(saved_results_map[img_id])
        continue

    # 根据数据集类型选择推理方式
    dataset_label = img_file_name.split('_')[0] + '_' + img_file_name.split('_')[1]
    scale_factor, infer_type = scale_factors.get(dataset_label, (1, "one"))
    
    if infer_type == "one":
        pred_instances = infer_one_image(img_path, model)
    elif infer_type == "one-tta":
        pred_instances = infer_one_tta(img_path, model, iou_threshold=0.4)
    elif infer_type == "slice":
        pred_instances = infer_slice(img_path, model, scale_factor)
    elif infer_type == "slice-tta":
        pred_instances = infer_slice_tta(img_path, model, scale_factor, iou_threshold=0.4)

    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks # .cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()

    for score, mask, bbox in zip(scores, masks, bboxes):
        # 获取RLE编码
        # print(mask.shape)
        rle = encode(np.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')

        # 边界框转换
        bbox_out = [float(bbox[0]), float(bbox[1]),float(bbox[2]), float(bbox[3])] # [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]

        result_dict = {
            "image_id": img_id,
            "bbox": bbox_out,
            "score": float(score),
            "category_id": 1,  # 假设类别ID为1，根据实际情况调整
            "segmentation": {
                "size": [mask.shape[0], mask.shape[1]],
                "counts": rle['counts']
            }
        }
        results_list.append(result_dict)

# 保存所有结果
output_json_path = r'C:\Users\Mikky\Desktop\树冠数据集\results_sample.json'
with open(output_json_path, 'w') as f:
    json.dump(results_list, f, indent=4)

print(f"Completed. Results saved to {output_json_path}")