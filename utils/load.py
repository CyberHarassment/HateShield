from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np

def load_video(vis_path, num_frm=5):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    try:
        vr = VideoReader(vis_path, ctx=cpu(0))
    except:
        # If the video is corrupted, return black images
        return [[Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(num_frm)]]
    total_frame_num = len(vr)

    # n_clips = how many 5-second clips to extract
    fps = vr.get_avg_fps()
    n_clips = min(int(np.floor(total_frame_num / (fps * 5))), 24)
    total_frame_num = n_clips * fps * 5

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm * n_clips)
    if total_num_frm == 0:
        return [[Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(num_frm)]]
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = []
    for i in range(n_clips):
        cur_clip_imgs = [Image.fromarray(img_array[i, j]) for j in range(num_frm)]
        clip_imgs.append(cur_clip_imgs)

    return clip_imgs

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq