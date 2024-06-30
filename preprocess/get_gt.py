"""
The annotation csv file has this format:

video_file_name,label,hate_snippet,target
hate_video_1.mp4,Hate,"[['00:00:34', '00:01:24']]",Blacks
hate_video_2.mp4,Hate,"[['00:00:06', '00:02:06']]",Blacks

Convert this csv file into a list of dict annotations:
each dict should contain 
    - "name": the video_file_name without extension 
    - "class": Hate is 1, Non hate is 0
    - "seg": a list of 1 or 0 indicator, the length of the list should be (length of video)/5
             each indicator is a signal of whether a 5-second clip is overlapped with hate_snippet annotation
 
there is another folder contains non-hate videos, include them in the annotations:
    - "name": the video_file_name without extension
    - "class": 0
    - "seg": a list of 0.
"""

import os
import cv2
import pandas as pd
import pickle
from moviepy.editor import VideoFileClip

def get_duration(video_path):
    # Open the video file and get its duration
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
    except:
        # if hate_video_147.mp4 duration is 00:02:30
        # elif hate_video_292.mp4 duration is 00:02:14
        if video_path.endswith('hate_video_147.mp4'):
            duration = 150
        elif video_path.endswith('hate_video_292.mp4'):
            duration = 134
        else:
            print(f'Error: {video_path}')
            exit()
    return int(duration)

def get_seg_gt(video_length, ann_list):
    seg = [0] * video_length
    for ann in ann_list:
        start_time = int(ann[0].split(':')[2]) + int(ann[0].split(':')[1]) * 60 + int(ann[0].split(':')[0]) * 3600
        end_time = int(ann[1].split(':')[2]) + int(ann[1].split(':')[1]) * 60 + int(ann[1].split(':')[0]) * 3600
        seg[start_time:end_time] = [1] * (end_time - start_time)
    return seg


def gtcsv_to_ann(csv_file, root_dir):
    hate_dir = os.path.join(root_dir, "hate_videos")
    nonhate_dir = os.path.join(root_dir, "non_hate_videos")
    
    # read and parse csv file
    df = pd.read_csv(csv_file)
    annotations = []
    for index, row in df.iterrows():
        video_file_name = row['video_file_name']
        label = row['label']
        hate_snippet = eval(row['hate_snippet'])
        target = row['target']
        # get the video length in seconds
        video_path = os.path.join(hate_dir, video_file_name)
        video_length = get_duration(video_path)

        annotation = {
            'name': os.path.splitext(video_file_name)[0],
            'class': 1 if label == 'Hate' else 0,
            'seg': get_seg_gt(video_length, hate_snippet)
            }
        
        annotations.append(annotation)
    
    # include non-hate videos in the annotations
    nonhate_files = os.listdir(nonhate_dir)
    for file in nonhate_files:
        video_path = os.path.join(nonhate_dir, file)
        video_length = get_duration(video_path)
        annotation = {
            'name': os.path.splitext(file)[0],
            'class': 0,
            'seg': [0] * video_length
        }
        
        annotations.append(annotation)
    return annotations

def split_annotations(annotations):
    fold1, fold2, fold3, fold4, fold5 = [], [], [], [], []
    hate_id = 0
    nonhate_id = 0
    for i, ann in enumerate(annotations):
        if ann['class'] == 1:
            if hate_id % 5 == 0:
                fold1.append(ann)
            elif hate_id % 5 == 1:
                fold2.append(ann)
            elif hate_id % 5 == 2:
                fold3.append(ann)
            elif hate_id % 5 == 3:
                fold4.append(ann)
            else:
                fold5.append(ann)
            hate_id += 1
        else:
            if nonhate_id % 5 == 0:
                fold1.append(ann)
            elif nonhate_id % 5 == 1:
                fold2.append(ann)
            elif nonhate_id % 5 == 2:
                fold3.append(ann)
            elif nonhate_id % 5 == 3:
                fold4.append(ann)
            else:
                fold5.append(ann)
            nonhate_id += 1
    # print the number of hate and non-hate videos in each fold
    # for fold in [fold1, fold2, fold3, fold4, fold5]:
    #     hate = 0
    #     nonhate = 0
    #     for ann in fold:
    #         if ann['class'] == 1:
    #             hate += 1
    #         else:
    #             nonhate += 1
    #     print(f"{len(fold)}: {hate} hate, {nonhate} non-hate")

    final_dict = {
        'fold1': {'train': fold2 + fold3 + fold4 + fold5, 'test': fold1},
        'fold2': {'train': fold1 + fold3 + fold4 + fold5, 'test': fold2},
        'fold3': {'train': fold1 + fold2 + fold4 + fold5, 'test': fold3},
        'fold4': {'train': fold1 + fold2 + fold3 + fold5, 'test': fold4},
        'fold5': {'train': fold1 + fold2 + fold3 + fold4, 'test': fold5}
    }
    return final_dict

def save_annotations(annotations, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(annotations, f)

if __name__ == '__main__':
    root_dir = "/scratch/jin7/datasets/hatemm"
    csv_file = os.path.join(root_dir, "HateMM-clean_annotation.csv")
    save_path = "../data/final_clean_gt.pkl"
    annotations = gtcsv_to_ann(csv_file, root_dir)
    annotations = split_annotations(annotations)
    # print the number of hate and non-hate videos in each fold
    for fold in annotations:
        print(f"{fold}: {len(annotations[fold]['train'])} train, {len(annotations[fold]['test'])} test")

    save_annotations(annotations, save_path)
    print("Annotations saved at", save_path)
