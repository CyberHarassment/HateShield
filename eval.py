from utils.metrics import cls_metrics, convert_iou
import numpy as np
import pickle
from data.process import downsample_sec_gt
import torch

gt = pickle.load(open('data/hatemm_gt.pkl', 'rb'))

pred = pickle.load(open('data/trainquery.pkl', 'rb'))

keys = list(pred.keys())
print(f"Total number of predictions: {len(keys)}")
all_preds = []
all_labels = []
all_seg_preds = []
all_seg_labels = []
all_iou_preds = []
all_iou_labels = []
for key in keys:
    all_preds.append(pred[key]['class'])
    all_labels.append(gt[key]['class'])


    pred_seg = pred[key]['segment']
    gt_seg = gt[key]['segment']
    # gt_seg = downsample_sec_gt(gt_seg).numpy()
    gt_seg = gt_seg.numpy()
    # upsample pred_seg to 5 seconds by torch.repeat_interleave
    pred_seg = torch.tensor(pred_seg).repeat_interleave(5).numpy()[:len(gt_seg)]
    
    # pred_seg = np.array(pred_seg[:len(gt_seg)])
    # print(gt_seg, pred_seg)
    all_seg_preds.extend(pred_seg)
    all_seg_labels.extend(gt_seg)

    gt_iou, pred_iou = convert_iou(gt_seg, pred_seg, threshold=0.5)
    all_iou_preds.extend(pred_iou)
    all_iou_labels.extend(gt_iou)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
metrics_cls = cls_metrics(all_labels, all_preds)
all_seg_preds = np.array(all_seg_preds)
all_seg_labels = np.array(all_seg_labels)
metrics_seg = cls_metrics(all_seg_labels, all_seg_preds)
all_iou_preds = np.array(all_iou_preds)
all_iou_labels = np.array(all_iou_labels)
metrics_iou = cls_metrics(all_iou_labels, all_iou_preds)

keys = list(metrics_cls.keys())
header = 'metrics:\t' + '\t'.join(keys)
print(header)
cls_out = 'classify:\t' + '\t'.join([f"{metrics_cls[k]:.4f}" for k in keys])
print(cls_out)
seg_out = 'segment:\t' + '\t'.join([f"{metrics_seg[k]:.4f}" for k in keys])
print(seg_out)
iou_out = 'iou:     \t' + '\t'.join([f"{metrics_iou[k]:.4f}" for k in keys])
print(iou_out)
    