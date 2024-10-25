import os
import torch
from torch.utils.data import Dataset

class HateMMClipclapDataset(Dataset):
    def __init__(self, data_dir, gt, split='train', preprocess=None):
        self.data_dir = data_dir
        self.gt = gt[split]
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        sample = self.gt[idx]
        video_name = sample['name']
        cls_label = sample['class']
        seg_label = sample['seg']
        if 'non_hate' in video_name:
            video_dir = os.path.join(self.data_dir, 'non_hate_videos', video_name)
        else:
            video_dir = os.path.join(self.data_dir, 'hate_videos', video_name)
        pt_file = None
        for file in os.listdir(video_dir):
            if file.endswith('.pt'):
                pt_file = os.path.join(video_dir, file)
                break
        if pt_file is None:
            # print(f"Cannot find .pt file in {video_dir}")
            features = {
                'video': torch.zeros(1, 512),
                'audio': torch.zeros(1, 512),
                'transcript': torch.zeros(1, 512),
                'ocr': torch.zeros(1, 512)
            }
        else:
            features = torch.load(pt_file)
        visual_features = features['video']
        audio_features = features['audio']
        transcript_features = features['transcript']
        ocr_features = features['ocr']
        if self.preprocess:
            visual_features = self.preprocess(visual_features)
            audio_features = self.preprocess(audio_features)
            transcript_features = self.preprocess(transcript_features)
            ocr_features = self.preprocess(ocr_features)
        output = {
            'visual': visual_features,
            'audio': audio_features,
            'ocr': ocr_features,
            'transcript': transcript_features,
            'class': cls_label,
            'seg': seg_label,
            'name': video_name
        }
        return output

class HateMMLanguagebindDataset(Dataset):
    def __init__(self, data_dir, gt, split='train', preprocess=None):
        self.data_dir = data_dir
        self.gt = gt[split]
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        sample = self.gt[idx]
        video_name = sample['name']
        cls_label = sample['class']
        seg_label = sample['seg']
        if 'non_hate' in video_name:
            video_dir = os.path.join(self.data_dir, 'non_hate_videos', video_name)
        else:
            video_dir = os.path.join(self.data_dir, 'hate_videos', video_name)
        # video_dir = os.path.join(self.data_dir, video_name)
        pt_file = None
        for file in os.listdir(video_dir):
            if file.endswith('.pt'):
                pt_file = os.path.join(video_dir, file)
                break
        if pt_file is None:
            # print(f"Cannot find .pt file in {video_dir}")
            features = {
                'video': torch.zeros(1, 768),
                'audio': torch.zeros(1, 768),
                'transcript': torch.zeros(1, 768),
                'ocr': torch.zeros(1, 768)
            }
        else:
            features = torch.load(pt_file)
        visual_features = features['video']
        audio_features = features['audio']
        transcript_features = features['transcript']
        ocr_features = features['ocr']
        if visual_features.shape == (1, 512):
            visual_features = torch.zeros(1, 768)
            audio_features = torch.zeros(1, 768)
            transcript_features = torch.zeros(1, 768)
            ocr_features = torch.zeros(1, 768)
        if self.preprocess:
            visual_features = self.preprocess(visual_features)
            audio_features = self.preprocess(audio_features)
            transcript_features = self.preprocess(transcript_features)
            ocr_features = self.preprocess(ocr_features)
        output = {
            'visual': visual_features,
            'audio': audio_features,
            'transcript': transcript_features,
            'ocr': ocr_features,
            'class': cls_label,
            'seg': seg_label,
            'name': video_name
        }
        return output

class GAP:
    def __call__(self, features):
        return torch.mean(features, dim=0)