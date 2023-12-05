import random
import numpy as np


class BaseClipSampler():
    def __init__(self, num_clips, clip_length):
        self.num_clips = num_clips
        self.clip_length = clip_length


class RandomClipSampler(BaseClipSampler):
    def __init__(self, num_clips, clip_length):
        super().__init__(num_clips, clip_length)
    
    def __call__(self, segment_start_sec, segment_end_sec):
        start = segment_start_sec
        end = segment_end_sec - self.clip_length
        if end <= start:
            return [segment_start_sec, segment_end_sec]
        clip_start = random.uniform(start, end)
        return [[clip_start, clip_start + self.clip_length]]


class LastClipSampler(BaseClipSampler):
    def __init__(self, num_clips, clip_length):
        super().__init__(num_clips, clip_length)
    
    def __call__(self, segment_start_sec, segment_end_sec):
        clip_start = segment_end_sec - self.clip_length
        return [[max(clip_start, segment_start_sec), segment_end_sec]]


class MultiUniformeSampler(BaseClipSampler):
    def __init__(self, num_clips, clip_length):
        super().__init__(num_clips, clip_length)

    def __call__(self, segment_start_sec, segment_end_sec):
        start = segment_start_sec
        end = segment_end_sec - self.clip_length
        clip_starts = np.linspace(start, end, num=self.num_clips)
        clip_ends = clip_starts + self.clip_length
        clip_starts = np.clip(clip_starts, a_min=segment_start_sec)
        clip_ends = np.clip(clip_ends, a_max=segment_end_sec)
        return np.stack([clip_starts, clip_ends], dim=1).tolist()
