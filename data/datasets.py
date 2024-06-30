import os
import torch
from torch.utils.data import Dataset

class FinalDataset(Dataset):
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
        video_dir = os.path.join(self.data_dir, video_name)
        visual_features = torch.load(os.path.join(video_dir, 'visual.pt'))
        audio_features = torch.load(os.path.join(video_dir, 'audio.pt'))
        ocr_features = torch.load(os.path.join(video_dir, 'ocr.pt'))
        transcript_features = torch.load(os.path.join(video_dir, 'transcript.pt'))
        if self.preprocess:
            visual_features = self.preprocess(visual_features)
            audio_features = self.preprocess(audio_features)
            ocr_features = self.preprocess(ocr_features)
            transcript_features = self.preprocess(transcript_features)
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

class GAP:
    def __call__(self, features):
        return torch.mean(features, dim=0)