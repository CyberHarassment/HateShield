import os
import torch
import pickle
from data.datasets import HateMMClipclapDataset, GAP
import numpy as np
from utils.metrics import cls_metrics
from tqdm import tqdm
import random
import csv 
import time
    
def get_query_features(trainset):
    visual_features_c0 = []
    visual_features_c1 = []
    transcript_features_c0 = []
    transcript_features_c1 = []
    audio_features_c0 = []
    audio_features_c1 = []
    ocr_features_c0 = []
    ocr_features_c1 = []
    for i in tqdm(range(len(trainset))):
        data = trainset[i]
        visual = data["visual"]
        transcript = data["transcript"]
        audio = data["audio"]
        ocr = data["ocr"]
        label = data["class"]
        # print(label.item())
        if label == 0:
            visual_features_c0.append(visual.unsqueeze(0))
            transcript_features_c0.append(transcript.unsqueeze(0))
            audio_features_c0.append(audio.unsqueeze(0))
            ocr_features_c0.append(ocr.unsqueeze(0))
        else:
            visual_features_c1.append(visual.unsqueeze(0))
            transcript_features_c1.append(transcript.unsqueeze(0))
            audio_features_c1.append(audio.unsqueeze(0))
            ocr_features_c1.append(ocr.unsqueeze(0))
    
    # random select 50 samples
    num_samples = 50
    visual_features_c0 = random.sample(visual_features_c0, num_samples)
    visual_features_c1 = random.sample(visual_features_c1, num_samples)
    transcript_features_c0 = random.sample(transcript_features_c0, num_samples)
    transcript_features_c1 = random.sample(transcript_features_c1, num_samples)
    audio_features_c0 = random.sample(audio_features_c0, num_samples)
    audio_features_c1 = random.sample(audio_features_c1, num_samples)
    ocr_features_c0 = random.sample(ocr_features_c0, num_samples)
    ocr_features_c1 = random.sample(ocr_features_c1, num_samples)
    
    visual_features_c0 = torch.cat(visual_features_c0, dim=0).mean(dim=0, keepdim=True)
    visual_features_c1 = torch.cat(visual_features_c1, dim=0).mean(dim=0, keepdim=True)
    transcript_features_c0 = torch.cat(transcript_features_c0, dim=0).mean(dim=0, keepdim=True)
    transcript_features_c1 = torch.cat(transcript_features_c1, dim=0).mean(dim=0, keepdim=True)
    audio_features_c0 = torch.cat(audio_features_c0, dim=0).mean(dim=0, keepdim=True)
    audio_features_c1 = torch.cat(audio_features_c1, dim=0).mean(dim=0, keepdim=True)
    ocr_features_c0 = torch.cat(ocr_features_c0, dim=0).mean(dim=0, keepdim=True)
    ocr_features_c1 = torch.cat(ocr_features_c1, dim=0).mean(dim=0, keepdim=True)
    
    # combined_c0 = visual_features_c0 + transcript_features_c0 + ocr_features_c0
    # combined_c1 = visual_features_c1 + transcript_features_c1 + ocr_features_c1
    res = {
        'visual': torch.cat([visual_features_c0, visual_features_c1], dim=0),
        'transcript': torch.cat([transcript_features_c0, transcript_features_c1], dim=0),
        'audio': torch.cat([audio_features_c0, audio_features_c1], dim=0),
        'ocr': torch.cat([ocr_features_c0, ocr_features_c1], dim=0),
        # 'combined': torch.cat([combined_c0, combined_c1], dim=0)
    }
    return res

def compute_similarity(query, features):
    start_time = time.time()
    query /= query.norm(dim=-1, keepdim=True)
    query[torch.isnan(query)] = 0
    features /= features.norm(dim=-1, keepdim=True)
    features[torch.isnan(features)] = 0
    similarity = (100.0 * features.float() @ query.float().T).softmax(dim=-1)
    # print(f"Time for similarity: {time.time() - start_time}")
    return similarity

def pred_per_video(video_data, query, show_info=True, sim_thres=0.65, num_modals_thres=2, num_clips_thres=2):
    if show_info:
        print("current video label: ", video_data["class"].item())
    visual = video_data["visual"]
    transcript = video_data["transcript"]
    audio = video_data["audio"]
    ocr = video_data["ocr"]
    visual_sim = compute_similarity(query["visual"], visual)
    transcript_sim = compute_similarity(query["transcript"], transcript)
    audio_sim = compute_similarity(query["audio"], audio)
    ocr_sim = compute_similarity(query["ocr"], ocr)
    combined_sim = compute_similarity(query["visual"] + query["transcript"] + query["ocr"], visual + transcript + ocr)
    if show_info:
        print("visual similarity: ", visual_sim)
        print("transcript similarity: ", transcript_sim)
        print("audio similarity: ", audio_sim)
        print("ocr similarity: ", ocr_sim)

    all_sim_positives = torch.cat([visual_sim[:,1].unsqueeze(1), 
                                   transcript_sim[:,1].unsqueeze(1), 
                                   audio_sim[:,1].unsqueeze(1), 
                                   ocr_sim[:,1].unsqueeze(1)
                                   ], dim=1)
    # all_sim_positives = combined_sim[:,1]

    all_sim_positives[all_sim_positives < sim_thres] = 0
    all_sim_positives[all_sim_positives >= sim_thres] = 1
    all_sim_positives = all_sim_positives.sum(dim=1)
    all_sim_positives[all_sim_positives < num_modals_thres] = 0
    all_sim_positives[all_sim_positives >= num_modals_thres] = 1
    if all_sim_positives.sum() >= num_clips_thres or all_sim_positives.sum() == len(all_sim_positives):
        pred = 1
    else:
        pred = 0
    return pred, all_sim_positives



def test_fold(data_dir, all_data_dict, fold, sim_thres=0.65, num_modals_thres=2, num_clips_thres=2, save_pred=None):
    data_dict = all_data_dict[fold]
    trainset = HateMMClipclapDataset(data_dir, data_dict, split='train', preprocess=GAP())
    testset = HateMMClipclapDataset(data_dir, data_dict, split='test')

    with torch.no_grad():
        query = get_query_features(trainset)

    all_preds = []
    all_labels = []
    progress_bar = tqdm(range(len(testset)))
    for i in progress_bar:
        data = testset[i]
        name = data["name"]
        save_pred[name] = {}
        # print(testset.video_names[i])
        with torch.no_grad():
            pred, seg = pred_per_video(data, query, show_info=False, sim_thres=sim_thres, num_modals_thres=num_modals_thres, num_clips_thres=num_clips_thres)
        save_pred[name]['class'] = pred
        save_pred[name]['seg'] = seg
        all_preds.append(pred)
        all_labels.append(data["class"])
        progress_bar.set_description(f"{fold}: {i}/{len(testset)}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = cls_metrics(all_labels, all_preds)
    return metrics, save_pred

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    data_dir = 'path to the processed features of CLIPCLAP'
    data_file = './data/crossval_gt.pkl'
    all_data_dict = pickle.load(open(data_file, 'rb'))
    folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
    save_file = './data/HateMM_clipclap.pkl'

    sim_thres = 0.75
    num_modals_thres = 2
    num_clips_thres = 2
    save_pred = {}
    cur_results = {}
    for fold in folds:
        cur_results[fold], save_pred = test_fold(data_dir, all_data_dict, fold, sim_thres=sim_thres, num_modals_thres=num_modals_thres, num_clips_thres=num_clips_thres, save_pred=save_pred)

    
    print(f"Total number of predictions: {len(save_pred.keys())}")
    with open(save_file, 'wb') as f:
        pickle.dump(save_pred, f)