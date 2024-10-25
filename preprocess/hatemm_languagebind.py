import os
import sys
import torch
import whisper
import pickle
from tqdm import tqdm
import laion_clap
import clip
import easyocr
import moviepy.editor as mp
import numpy as np
from PIL import Image
LBDIR = 'path to languagebind directory'
sys.path.append(LBDIR)
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def build_languagebind(device):
    clip_type = {
        'video': 'LanguageBind_Video_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'thermal': 'LanguageBind_Thermal',
        'image': 'LanguageBind_Image',
        'depth': 'LanguageBind_Depth',
    }
    model = LanguageBind(clip_type=clip_type, cache_dir=LBDIR+'cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir=LBDIR+'cache_dir/tokenizer_cache_dir')
    transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}
    return model, tokenizer, transform

MODEL, TOKENIZER, TRANSFORM = build_languagebind(device)
EASYOCR = easyocr.Reader(['en', 'ch_sim'], gpu=True)
WHISPER = whisper.load_model("base").to(device)

def combine_lines(lines, overlap_threshold=0.5):
    def overlap_ratio(s1, s2):
        len_s1, len_s2 = len(s1), len(s2)
        if len_s1 == 0 or len_s2 == 0:
            return 0
        overlap_len = len(set(s1.split()) & set(s2.split()))
        return overlap_len / min(len_s1, len_s2)

    combined_lines = []
    for line in lines:
        if not combined_lines:
            combined_lines.append(line)
        else:
            for i, combined_line in enumerate(combined_lines):
                if overlap_ratio(line, combined_line) > overlap_threshold:
                    break
            else:
                combined_lines.append(line)
    return ' '.join(combined_lines)

def process_video(video_file, save_folder, clip_duration=5):
    # 1. read video file with moviepy, every 5 seconds, save a clip
    # for each clip:
    #   2. Get the audio and save it as a .wav file
    #   3. Get the transcript with whisper and save it as a .txt file
    #   4. Sample 10 frames from each clip and use easyocr to get the text and save it as a .txt file
    #   5. Use CLIP to get the visual features
    #   6. Use CLAP to get the audio features
    #   7. Use CLIP text encoder to get the text features from the transcript and OCR
    # endfor
    # 8. Save all the features in a .pt file

    if video_file.endswith('hate_video_147.mp4') or video_file.endswith('hate_video_292.mp4') \
        or video_file.endswith('hate_video_95.mp4') or video_file.endswith('hate_video_385.mp4'):
        # these videos do not have image frames
        return

    video = mp.VideoFileClip(video_file)
    duration = video.duration
    num_clips = max(int(duration / clip_duration), 1)
    num_clips = min(num_clips, 24)
    v_features = []
    a_features = []
    t_features = []
    o_features = []
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = min((i + 1) * clip_duration, duration)
        clip_segment = video.subclip(start_time, end_time)
        clip_segment_path = os.path.join(save_folder, f'{video_file.split("/")[-1].split(".")[0]}_clip{i+1}.mp4')
        try:
            clip_segment.write_videofile(clip_segment_path)
        except:
            continue
        # 2. Get the audio and save it as a .wav file
        audio_file = clip_segment_path.replace('.mp4', '.wav')
        try:    
            clip_segment.audio.write_audiofile(audio_file)
        except:
            # no audio in the clip, create a silent audio file
            os.system(f'ffmpeg -f lavfi -i anullsrc=r=11025:cl=mono -t 1 -acodec aac {audio_file}')
        # 3. Get the transcript with whisper and save it as a .txt file
        transcript_file = clip_segment_path.replace('.mp4', '.txt')
        transcript = WHISPER.transcribe(audio_file)['text']
        with open(transcript_file, 'w') as f:
            f.write(transcript)
        # 4. Sample 10 frames from each clip and use easyocr to get the text and save it as a .txt file
        try:
            frames = clip_segment.iter_frames(fps=2, dtype='uint8')
            # save the frames
            for i, frame in enumerate(frames):
                frame_path = clip_segment_path.replace('.mp4', f'_frame{i}.jpg')
                Image.fromarray(frame).save(frame_path)
        except:
            # save 10 black frames
            for i in range(10):
                frame_path = clip_segment_path.replace('.mp4', f'_frame{i}.jpg')
                frame = np.zeros((1080, 1920, 3), dtype='uint8')
                Image.fromarray(frame).save(frame_path)
        ocr_text = []
        frame_files = [x for x in os.listdir(save_folder) if x.endswith('.jpg')]
        for frame_file in frame_files:
            frame_path = os.path.join(save_folder, frame_file)
            ocr_result = EASYOCR.readtext(frame_path, detail=0, paragraph=True)
            # the ocr_result is a list of strings, put them together
            ocr_result = ' '.join(ocr_result)
            ocr_result = ocr_result.replace('\n', ' ')
            ocr_text.append(ocr_result)
        # remove redundant text
        ocr_text = combine_lines(ocr_text)
        # print(ocr_text)
        ocr_text_file = clip_segment_path.replace('.mp4', '_ocr.txt')
        with open(ocr_text_file, 'w') as f:
            f.write(ocr_text)
        
        # 5. Use Languagebind to extract features
        frame_paths = [os.path.join(save_folder, x) for x in frame_files]
        inputs = {
            'image': to_device(TRANSFORM['image'](frame_paths), device),
            'audio': to_device(TRANSFORM['audio']([audio_file]), device),
        }
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        with open(ocr_text_file, 'r') as f:
            ocr_text = f.read()
        inputs['language'] = to_device(TOKENIZER([transcript, ocr_text], max_length=77, padding='max_length',
                                                truncation=True, return_tensors='pt'), device)

        start_time = time.time()
        with torch.no_grad():
            embeddings = MODEL(inputs)
        print(f"Time for LanguageBind: {time.time() - start_time}")

        v_features.append(torch.mean(embeddings['image'], 0, True).cpu())
        a_features.append(embeddings['audio'].cpu())
        t_features.append(embeddings['language'][0].unsqueeze(0).cpu())
        o_features.append(embeddings['language'][1].unsqueeze(0).cpu())

        # remove intermediate files
        os.remove(clip_segment_path)
        os.remove(audio_file)
        # remove frames
        for frame_file in frame_files:
            os.remove(os.path.join(save_folder, frame_file))
        os.remove(transcript_file)
        os.remove(ocr_text_file)
    
    if len(v_features) == 0:
        # return zero tensors
        v_features = torch.zeros(1, 768)
        a_features = torch.zeros(1, 768)
        t_features = torch.zeros(1, 768)
        o_features = torch.zeros(1, 768)
    else:
        v_features = torch.cat(v_features, dim=0)
        a_features = torch.cat(a_features, dim=0)
        t_features = torch.cat(t_features, dim=0)
        o_features = torch.cat(o_features, dim=0)

    features = {
        'video': v_features,
        'audio': a_features,
        'transcript': t_features,
        'ocr': o_features,
    }
    save_file = os.path.join(save_folder, f'{video_file.split("/")[-1].split(".")[0]}.pt')
    torch.save(features, save_file)
    return

if __name__ == '__main__':
    dataset_dir = 'path to HateMM dataset'
    save_dir = 'path to save the processed features'
    subfolders = ['hate_videos', 'non_hate_videos']
    for subfolder in subfolders:
        for video in tqdm(os.listdir(os.path.join(dataset_dir, subfolder))):
            # find the .mp4 file in the video folder
            video_name = video.split('.')[0]
            video_file = os.path.join(dataset_dir, subfolder, video)
            save_path = os.path.join(save_dir, subfolder, video_name)
            os.makedirs(save_path, exist_ok=True)
            # if os.path.exists(os.path.join(save_path, f'{video_file.split("/")[-1].split(".")[0]}.pt')):
            #     continue
            process_video(video_file, save_path)
