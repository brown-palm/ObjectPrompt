import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision
import copy
import pdb
import itertools
import random
from collections import defaultdict
from tqdm import tqdm

from pytorchvideo.transforms import (
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsampleRepeated,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)

from . import clip_samplers
from ..utils import file_util
from src.utils.parser import load_config, parse_args


class BaseVideoDataset(Dataset):
    def __init__(self, cfg, annotation_path, is_train, is_test) -> None:
        super().__init__()
        self.cfg = cfg
        self.annotation_path = annotation_path
        if len(self.cfg.data.examples_to_keep) > 0:
            self.examples_to_keep = set(file_util.load_json(self.cfg.data.examples_to_keep))  # set of last_observed_id 
        self.is_train = is_train
        self.is_test = is_test
        # # use pre_transform to accelerate possibly
        self.pre_transform, self.spatial_transform, self.temporal_transform = self.get_transform()
        self.clip_sampler = self.get_clip_sampler(cfg.data.num_clips_per_segment, cfg.data.clip_length)
        if self.cfg.data.use_pred_text:
            self.predictions = file_util.load_json(self.cfg.data.prediction_path)  
        self.annotations = self.convert(file_util.load_json(self.annotation_path))

    def get_transform(self):
        cfg = self.cfg

        # pre  (c, t, h, w)
        pre_temporal_subsample = [UniformTemporalSubsample(cfg.data.num_frames)]

        # spatial
        spatial_norm = [
            Lambda(lambda x: x / 255.0),
            Normalize(cfg.data.mean, cfg.data.std)
        ]
        if self.is_train:
            spatial_crop = [
                RandomShortSideScale(
                    min_size=cfg.data.jitter_scales[0],
                    max_size=cfg.data.jitter_scales[1],
                ),
                RandomCrop(cfg.data.crop_size),
                RandomHorizontalFlip(p=cfg.data.random_flip)
            ]
        else:
            spatial_crop = [
                ShortSideScale(cfg.data.jitter_scales[0]),
                CenterCrop(cfg.data.crop_size),
            ]
        
        # temporal
        if len(self.cfg.DATA.INPUT_CHANNEL_NUM) == 2:
            temporal_subsample = [UniformTemporalSubsampleRepeated((cfg.SLOWFAST.ALPHA, 1))]
        else:
            temporal_subsample = [UniformTemporalSubsampleRepeated((1,))]
        
        return Compose(pre_temporal_subsample), Compose(spatial_norm + spatial_crop), Compose(temporal_subsample)

    def get_clip_sampler(self, num_clips, clip_length):
        if self.cfg.data.clip_sampler == 'random':
            return clip_samplers.RandomClipSampler(num_clips, clip_length)
        elif self.cfg.data.clip_sampler == 'last':
            return clip_samplers.LastClipSampler(num_clips, clip_length)
        elif self.cfg.data.clip_sampler == 'multi_uniform':
            return clip_samplers.MultiUniformeSampler(num_clips, clip_length)
        else:
            raise NotImplementedError(f'Clip sampler {self.cfg.data.clip_sampler} not implemented')
    
    def __len__(self):
        return len(self.annotations)

    def get_future_anno(self, segments, idx):
        [start, end] = self.cfg.data.output_segments
        if idx + start >= len(segments) or idx + end <= 0:
            # no label at all
            return None
        if idx + start < 0 or idx + end > len(segments):
            # need to be partially masked
            if not self.cfg.data.output_mask:
                return None
        output = {
            "verb": [-1] * (end - start),
            "noun": [-1] * (end - start),
            "mask": [True] * (end - start),
            "last_observed_id": "{}_{}".format(segments[idx-1]['clip_uid'], segments[idx-1]['action_idx']),
        }
        for i in range(idx+start, idx+end):
            if i < 0 or i >= len(segments):
                continue
            if not self.is_test:
                output['verb'][i-start-idx] = segments[i]['verb_label']
                output['noun'][i-start-idx] = segments[i]['noun_label']
            output['mask'][i-start-idx] = False
        return output
    
    def get_video_anno(self, segments, idx):
        [start, end] = self.cfg.data.input_segments
        num_segments = end - start
        anno = {
            "path": '{}/{}.{}'.format(self.cfg.data.base_path, segments[idx]['clip_uid'], self.cfg.data.suffix),  # path to this video
            "verb": [-1] * num_segments,  # verb idx
            "noun": [-1] * num_segments,  # noun idx
            "meta_data": [None] * num_segments,  # each element: None or a list of [clip_start_sec, clip_end_sec]
            "mask": [True] * num_segments,  # True: masked
        }
        if self.cfg.data.input_from_annotated_segments:
            if idx + start >= len(segments) or idx + end <= 0:
                # no label at all
                return None
            if idx + start < 0 or idx + end > len(segments):
                # need to be partially masked
                if not self.cfg.data.input_mask:
                    return None
            for i in range(idx + start, idx + end):
                if i < 0 or i >= len(segments):
                    continue
                
                if not self.is_test:
                    anno["verb"][i-idx-start] = segments[i]['verb_label']
                    anno["noun"][i-idx-start] = segments[i]['noun_label']
                anno["mask"][i-idx-start] = False

                segment_start_sec = segments[i]['action_clip_start_sec']
                segment_end_sec = segments[i]['action_clip_end_sec']
                # clips: list of [clip_start_sec, clip_end_sec]
                clips = self.clip_sampler(segment_start_sec, segment_end_sec)
                anno["meta_data"][i-idx-start] = clips
        else:
            # construct segments
            segment_length, segment_interval = self.cfg.data.segment_length, self.cfg.data.segment_interval
            end_sec = segments[idx]['action_clip_start_sec'] - self.cfg.data.tau_a
            start_sec = end_sec - segment_length * num_segments - segment_interval * (num_segments - 1)
            if end_sec - segment_length < 0:
                # no input segments at all
                return None
            if start_sec < 0:
                # need to be partially masked
                if not self.cfg.data.input_mask:
                    return None
            t = start_sec
            for i in range(num_segments):
                if t >= 0:
                    # clips: list of [clip_start_sec, clip_end_sec]
                    clips = self.clip_sampler(t, t + segment_length,
                        self.cfg.data.num_clips_per_segments, self.cfg.data.clip_length)
                    anno['meta_data'][i] = clips
                    anno['mask'][i] = False
                t = t + segment_length + segment_interval
        return anno
    
    def get_text_anno(self, segments, idx):
        [start, end] = self.cfg.data.input_segments
        num_segments = end - start
        anno = {
            "verb": [-1] * num_segments,  # verb idx
            "noun": [-1] * num_segments,  # noun idx
            "mask": [True] * num_segments,  # True: masked
        }
        if self.cfg.data.input_from_annotated_segments:
            if idx + start >= len(segments) or idx + end <= 0:
                # no label at all
                return None
            if idx + start < 0 or idx + end > len(segments):
                # need to be partially masked
                if not self.cfg.data.input_mask:
                    return None
            for i in range(idx + start, idx + end):
                if i < 0 or i >= len(segments):
                    continue
                
                if not self.is_test:
                    anno["verb"][i-idx-start] = segments[i]['verb_label']
                    anno["noun"][i-idx-start] = segments[i]['noun_label']
                else:
                    raise NotImplementedError("Test set has no ground truth inputs.")
                anno["mask"][i-idx-start] = False
        else:
            raise NotImplementedError("Text modality only supports labelled segments as inputs.")
        return anno
    
    def get_prediction_anno(self, segments, idx):
        [start, end] = self.cfg.data.output_segments
        if idx + start >= len(segments) or idx + end <= 0:
            # no label at all
            return None
        if idx + start < 0 or idx + end > len(segments):
            # need to be partially masked
            if not self.cfg.data.output_mask:
                return None
        anno = {
            "verb": [[0] * self.cfg.data.num_pred_seqs for _ in range(end - start)],
            "noun": [[0] * self.cfg.data.num_pred_seqs for _ in range(end - start)],
            "mask": [True] * (end - start),
        }
        if idx + start - 1 < 0:
            # not in prediction file, mask all
            return anno 
        last_observed_id = "{}_{}".format(segments[idx+start-1]['clip_uid'], segments[idx+start-1]['action_idx'])
        if last_observed_id not in self.predictions:
            # not in prediction file, mask all
            return anno
        verb_mat = self.predictions[last_observed_id]['verb']
        verb_list = [el[:self.cfg.data.num_pred_seqs] for el in verb_mat]
        noun_mat = self.predictions[last_observed_id]['noun']
        noun_list = [el[:self.cfg.data.num_pred_seqs] for el in noun_mat]
        for i in range(end-start):
            if idx + start + i >= len(segments):
                continue
            anno['verb'][i] = verb_list[i]
            anno['noun'][i] = noun_list[i]
            anno['mask'][i] = False
        return anno
    
    def get_image_anno(self, segments, idx):
        [start, end] = self.cfg.data.image.input_segments
        image_fps = self.cfg.data.image.fps
        strict = self.cfg.data.strict
        anno = {
            "path": '{}/{}.pt'.format(self.cfg.data.image.base_path, segments[idx]['clip_uid']),
            "verb": [],  # verb idx
            "noun": [],  # noun idx
            "meta_data": [],  # each element: frame index when extracting image features
            "mask": [],  # True: masked
        }
        if self.cfg.data.image.input_from_annotated_segments:
            if idx + start >= len(segments) or idx + end <= 0:
                # no label at all
                return None
            if idx + start < 0 or idx + end > len(segments):
                # need to be partially masked
                if not self.cfg.data.image.input_mask:
                    return None
            num_images_per_segment = self.cfg.data.image.num_images_per_segment
            for i in range(idx + start, idx + end):
                if i < 0 or i >= len(segments):
                    anno["verb"].extend([-1] * num_images_per_segment)
                    anno["noun"].extend([-1] * num_images_per_segment)
                    anno["meta_data"].extend([0] * num_images_per_segment)
                    anno["mask"].extend([True] * num_images_per_segment)
                    continue

                if self.is_test:
                    anno["verb"].extend([-1] * num_images_per_segment)
                    anno["noun"].extend([-1] * num_images_per_segment)
                else:
                    anno["verb"].extend([segments[i]['verb_label']] * num_images_per_segment)
                    anno["noun"].extend([segments[i]['noun_label']] * num_images_per_segment)

                segment_start_sec = segments[i]['action_clip_start_sec']
                segment_end_sec = segments[i]['action_clip_end_sec']
                if i == idx + end - 1 and strict:
                    segment_end_sec -= 1 / (image_fps * 2)
                    assert segment_end_sec > segment_start_sec, 'segment too short, consider turning off strict'

                '''
                Divide (segment_start_sec, segment_end_sec) to 'num_images_per_segment' intervals, then sample one frame from each interval.
                During training: sample a random frame from each interval.
                TODO: During evaluation: sample the center frame from each interval.
                '''
                intervals = []  # list of [interval_start, interval_end]
                interval_duration = (segment_end_sec - segment_start_sec) / num_images_per_segment
                interval_start, interval_end = segment_start_sec, segment_start_sec
                for _ in range(num_images_per_segment):
                    interval_start = interval_end
                    interval_end = min(segment_end_sec, interval_end + interval_duration)
                    intervals.append([interval_start * image_fps, interval_end * image_fps])

                # from each interval, we randomly sample 1 image
                frame_indices = []
                for interval in intervals:
                    frame_idx = int(interval[0] + random.random() * (interval[1] - interval[0]))
                    frame_indices.append(frame_idx)

                anno["meta_data"].extend(frame_indices)
                anno['mask'].extend([False] * len(frame_indices))

        else:
            num_images = self.cfg.data.image.num_images_per_segment
            interval = self.cfg.data.image.image_interval
            if self.cfg.data.image.from_end:
                end_sec = segments[idx]['action_clip_end_sec'] - self.cfg.data.tau_a
            else:
                end_sec = segments[idx]['action_clip_start_sec'] - self.cfg.data.tau_a
            if strict:
                end_sec -= 1 / (image_fps * 2)
            start_sec = end_sec - (num_images - 1) * interval
            if end_sec < 0:
                # no inputs at all
                return None
            if start_sec < 0:
                # need to be partially masked
                if not self.cfg.data.image.input_mask:
                    return None
            anno = {
                "path": '{}/{}.pt'.format(self.cfg.data.image.base_path, segments[idx]['clip_uid']),
                "verb": [-1] * num_images,  # verb idx
                "noun": [-1] * num_images,  # noun idx
                "meta_data": [0] * num_images,  # each element: frame index when extracting image features
                "mask": [True] * num_images,  # True: masked
            }
            t = start_sec
            timesteps = []
            for i in range(num_images):
                if t < 0:
                    timesteps.append(0)
                else:
                    timesteps.append(t)
                    anno['mask'][i] = False
                t += interval
            anno['meta_data'] = [int(t * image_fps) for t in timesteps]
        return anno

    def get_object_anno(self, segments, idx):
        [start, end] = self.cfg.data.object.input_segments
        object_fps = self.cfg.data.object.fps
        strict = self.cfg.data.strict
        anno = {
            "path": '{}/{}.pt'.format(self.cfg.data.object.base_path, segments[idx]['clip_uid']),
            "verb": [],  # verb idx
            "noun": [],  # noun idx
            "meta_data": [],  # each element: frame index when extracting object features
            "mask": [],  # True: masked
        }
        if self.cfg.data.object.input_from_annotated_segments:
            if idx + start >= len(segments) or idx + end <= 0:
                # no label at all
                return None
            if idx + start < 0 or idx + end > len(segments):
                # need to be partially masked
                if not self.cfg.data.object.input_mask:
                    return None
            num_images_per_segment = self.cfg.data.object.num_images_per_segment
            for i in range(idx + start, idx + end):
                if i < 0 or i >= len(segments):
                    anno["verb"].extend([-1] * num_images_per_segment)
                    anno["noun"].extend([-1] * num_images_per_segment)
                    anno["meta_data"].extend([0] * num_images_per_segment)
                    anno["mask"].extend([True] * num_images_per_segment)
                    continue

                if self.is_test:
                    anno["verb"].extend([-1] * num_images_per_segment)
                    anno["noun"].extend([-1] * num_images_per_segment)
                else:
                    anno["verb"].extend([segments[i]['verb_label']] * num_images_per_segment)
                    anno["noun"].extend([segments[i]['noun_label']] * num_images_per_segment)

                segment_start_sec = segments[i]['action_clip_start_sec']
                segment_end_sec = segments[i]['action_clip_end_sec']
                if i == idx + end - 1 and strict:
                    segment_end_sec -= 1 / (object_fps * 2)
                    assert segment_end_sec > segment_start_sec

                '''
                Divide (segment_start_sec, segment_end_sec) to 'num_images_per_segment' intervals, then sample one frame from each interval.
                During training: sample a random frame from each interval.
                TODO: During evaluation: sample the center frame from each interval.
                '''
                intervals = []  # list of [interval_start, interval_end]
                interval_duration = (segment_end_sec - segment_start_sec) / num_images_per_segment
                interval_start, interval_end = segment_start_sec, segment_start_sec
                for _ in range(num_images_per_segment):
                    interval_start = interval_end
                    interval_end = min(segment_end_sec, interval_end + interval_duration)
                    intervals.append([interval_start * object_fps, interval_end * object_fps])

                # from each interval, we randomly sample 1 image
                frame_indices = []
                for interval in intervals:
                    frame_idx = int(interval[0] + random.random() * (interval[1] - interval[0]))
                    frame_indices.append(frame_idx)

                anno["meta_data"].extend(frame_indices)
                anno['mask'].extend([False] * len(frame_indices))
        else:
            num_images = self.cfg.data.object.num_images_per_segment
            interval = self.cfg.data.object.image_interval
            if self.cfg.data.object.from_end:
                end_sec = segments[idx]['action_clip_end_sec'] - self.cfg.data.tau_a
            else:
                end_sec = segments[idx]['action_clip_start_sec'] - self.cfg.data.tau_a
            if strict:
                end_sec -= 1 / (object_fps * 2)
            start_sec = end_sec - (num_images - 1) * interval
            if end_sec < 0:
                # no inputs at all
                return None
            if start_sec < 0:
                # need to be partially masked
                if not self.cfg.data.object.input_mask:
                    return None
            anno = {
                "path": '{}/{}.pt'.format(self.cfg.data.object.base_path, segments[idx]['clip_uid']),
                "verb": [-1] * num_images,  # verb idx
                "noun": [-1] * num_images,  # noun idx
                "meta_data": [0] * num_images,  # each element: frame index when extracting object features
                "mask": [True] * num_images,  # True: masked
            }
            t = start_sec
            timesteps = []
            for i in range(num_images):
                if t < 0:
                    timesteps.append(0)
                else:
                    timesteps.append(t)
                    anno['mask'][i] = False
                t += interval
            anno['meta_data'] = [int(t * object_fps) for t in timesteps]
        return anno
    
    def convert(self, row_annotations):
        # get modalities
        modalities = ['future']
        if self.cfg.model.use_vid > 0:
            modalities.append('video')
        if self.cfg.data.use_gt_text:
            modalities.append('text')
        if self.cfg.model.img_feat_size > 0:
            modalities.append('image')
        if self.cfg.model.obj_feat_size > 0:
            modalities.append('object')
        if self.cfg.data.use_pred_text:
            modalities.append('prediction')

        segments_all = row_annotations['clips']
        annotations = []
        for _, group in itertools.groupby(segments_all, key=lambda x: x['clip_uid']):
            segment_info = sorted(list(group), key=lambda x: x['action_idx'])
            for i in range(len(segment_info)):
                anno = {}   # {modality_name: anno}
                for modality in modalities:
                    get_anno_func = getattr(self, f'get_{modality}_anno')
                    anno_single = get_anno_func(segment_info, i)
                    if anno_single is not None:
                        anno[modality] = anno_single
                if 'future' in anno and len(anno) > 1:
                    # have labels and at least one modality as inputs
                    annotations.append(anno)
                    
        # filter 
        if len(self.cfg.data.examples_to_keep) == 0:
            return annotations
        filtered_annotations = []
        for annotation in annotations:
            if annotation['future']['last_observed_id'] in self.examples_to_keep:
                filtered_annotations.append(annotation)
        return filtered_annotations
        
    def fill_future(self, anno):
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_video(self, anno):
        video_path = anno['path']
        num_input_segments = self.cfg.data.input_segments[1] - self.cfg.data.input_segments[0]
        clips = [None] * num_input_segments

        # anno['meta_data']: each element is None or a list of [clip_start_sec, clip_end_sec]
        for i, clips_meta_data in enumerate(anno['meta_data']):
            if clips_meta_data is None:
                continue
            clips_in_segment = []
            for clip_start_end in clips_meta_data:
                [clip_start, clip_end] = clip_start_end
                clip, _, _ = torchvision.io.read_video(
                    video_path, clip_start, clip_end, pts_unit='sec')  # THWC
                clip = torch.permute(clip, (3, 0, 1, 2))  # CTHW
                clip = self.pre_transform(clip)  # CTHW
                clips_in_segment.append(clip)
            clips[i] = torch.concat(clips_in_segment, dim=0)  # (num_clips_per_segment * c, t, h, w)
            clips[i] = self.spatial_transform(clips[i])  # (num_clips_per_segment * c, t, h, w)

        # fill None in clips with dummy values
        mask = anno['mask']
        first_non_mask_idx = -1
        last_non_mask_idx = len(mask)
        for i in range(len(mask)):
            if not mask[i]:
                first_non_mask_idx = i
                break
        for i in range(len(mask)-1, -1, -1):
            if not mask[i]:
                last_non_mask_idx = i
                break
        assert first_non_mask_idx >= 0 and last_non_mask_idx < len(mask)
        for i in range(first_non_mask_idx):
            clips[i] = clips[first_non_mask_idx]
        for i in range(last_non_mask_idx+1, first_non_mask_idx):
            clips[i] = clips[last_non_mask_idx]
        
        clips = torch.concat(clips, dim=0)  # (num_segments * num_clips_per_segment * c, t, h, w)
        clips = self.temporal_transform(clips)  
        # a list of (num_segments * num_clips_per_segment * c, t_i, h, w)
        for i in range(len(clips)):
            clips[i] = clips[i].reshape((num_input_segments * self.cfg.data.num_clips_per_segment, -1, \
                clips[i].shape[-3], clips[i].shape[-2], clips[i].shape[-1]))
        anno['inputs'] = clips
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_text(self, anno):
        anno['inputs'] = torch.tensor([anno['verb'], anno['noun']]).T
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_image(self, anno):
        # TODO: handle EK style image features
        indices = anno['meta_data']
        num_frames_per_file = self.cfg.data.image.split 
        if num_frames_per_file > 0:
            file_id_and_offsets = defaultdict(list)  # {file_id: [offsets]}
            for frame_index in indices:
                file_id_and_offsets[frame_index // num_frames_per_file].append(frame_index % num_frames_per_file)
            file_id_and_offsets = sorted(list(file_id_and_offsets.items()), key=lambda x: x[0])  # [(file_id, [offsets])]
            image_features = []
            for file_id, offset_list in file_id_and_offsets:
                emb_fp = '{}-{}.pt'.format(anno['path'][:-3], file_id)
                embs = torch.load(emb_fp, map_location='cpu')  # (N, D)
                image_features.append(embs[offset_list])
            image_features = torch.cat(image_features, dim=0)
        else:
            image_features = torch.load(anno['path'], map_location='cpu')  # (N, D)
            image_features = image_features[indices]
        anno['inputs'] = image_features
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_object(self, anno):
        # TODO: handle EK style object features
        indices = anno['meta_data']
        num_frames_per_file = self.cfg.data.image.split 
        if num_frames_per_file > 0:
            file_id_and_offsets = defaultdict(list)  # {file_id: [offsets]}
            for frame_index in indices:
                file_id_and_offsets[frame_index // num_frames_per_file].append(frame_index % num_frames_per_file)
            file_id_and_offsets = sorted(list(file_id_and_offsets.items()), key=lambda x: x[0])  # [(file_id, [offsets])]
            object_features = []
            for file_id, offset_list in file_id_and_offsets:
                emb_fp = '{}-{}.pt'.format(anno['path'][:-3], file_id)
                embs = torch.load(emb_fp, map_location='cpu')  # (N, D)
                object_features.append(embs[offset_list])
            object_features = torch.cat(object_features, dim=0)
        else:
            object_features = torch.load(anno['path'], map_location='cpu')  # (N, D)
            object_features = object_features[indices]

        object_features = object_features.reshape((object_features.shape[0] * object_features.shape[1], -1))
        anno['inputs'] = object_features
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def fill_prediction(self, anno):
        anno['inputs'] = torch.tensor([anno['verb'], anno['noun']])  # 2, L, K
        anno['inputs'] = torch.permute(anno['inputs'], (1, 2, 0))  # L, K, 2
        anno['mask'] = torch.tensor(anno['mask'], dtype=torch.bool)

    def __getitem__(self, index):
        annotation = copy.deepcopy(self.annotations[index])
        for modality in annotation:
            fill_func = getattr(self, f'fill_{modality}')
            fill_func(annotation[modality])

        # TODO: optimize observed_labels_idx for multiple modalities
        observed_labels_idx = None
        for modality in annotation:
            if modality != 'future' and modality != 'prediction':
                observed_labels_idx = torch.tensor([annotation[modality]['verb'], annotation[modality]['noun']]).T
        item = {
            'forecast_labels_idx': torch.tensor([annotation['future']['verb'], annotation['future']['noun']]).T,
            'observed_labels_idx': observed_labels_idx,
            'last_observed_id': annotation['future']['last_observed_id'],
        }
        if self.cfg.model.use_vid:
            item['video'] = annotation['video']['inputs']
            item['mask_video'] = annotation['video']['mask']
        if self.cfg.data.use_gt_text:
            item['text'] = annotation['text']['inputs']
            item['mask_text'] = annotation['text']['mask']
        if self.cfg.model.img_feat_size > 0:
            item['image_features'] = annotation['image']['inputs']
            item['mask_image'] = annotation['image']['mask']
        if self.cfg.model.obj_feat_size > 0:
            item['object_features'] = annotation['object']['inputs']
            item['mask_object'] = annotation['object']['mask']
        if self.cfg.data.use_pred_text:
            item['pred_text'] = annotation['prediction']['inputs']
            item['mask_pred_text'] = annotation['prediction']['mask']
        return item
    

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if not hasattr(self, 'train_set'):
                self.train_set = BaseVideoDataset(self.cfg, self.cfg.data.train_anno_path, True, False)
            if not hasattr(self, 'val_set'):
                self.val_set = BaseVideoDataset(self.cfg, self.cfg.data.val_anno_path, False, False)

        if stage == "test" or stage is None:
            self.test_set = BaseVideoDataset(self.cfg,self.cfg.data.test_anno_path, False, True)

    def train_dataloader(self):
        if not hasattr(self, 'train_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.train.batch_size % num_gpus == 0
            batch_size = self.cfg.train.batch_size // num_gpus
            self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=batch_size, num_workers=self.cfg.train.num_workers)
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'val_loader'):
            num_gpus = self.cfg.num_gpus
            assert self.cfg.val.batch_size % num_gpus == 0
            batch_size = self.cfg.val.batch_size // num_gpus
            self.val_loader = DataLoader(self.val_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.val.num_workers)
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, 'test_loader'):
            # num_gpus = self.cfg.num_gpus
            num_gpus = 1
            assert self.cfg.test.batch_size % num_gpus == 0
            batch_size = self.cfg.test.batch_size // num_gpus
            self.test_loader = DataLoader(self.test_set, shuffle=False, batch_size=batch_size, num_workers=self.cfg.test.num_workers, drop_last=False)
        return self.test_loader

    
    
def sanity_check():
    args = parse_args()
    cfg = load_config(args)
    dm = BaseDataModule(cfg)
    dm.setup(stage="fit")

    # for batch in tqdm(dm.test_set):
    #     pass

    i = 0
    for annotation in dm.val_set.annotations:
        i += 1
        print(i)
        print(annotation)
        pdb.set_trace()

if __name__ == '__main__':
    sanity_check()
    # main()


'''
python -m src.datasets.base_dataset --cfg configs/ego4d/recognition_sf_video.yaml --exp_name ego4d/null \
    val.batch_size 1

python -m src.datasets.base_dataset --cfg configs/ego4d/text.yaml --exp_name ego4d/null \
    val.batch_size 1 train.batch_size 1

python -m src.datasets.base_dataset --cfg configs/ego4d/image_reproduce.yaml --exp_name ego4d/null \
    val.batch_size 1 train.batch_size 1

python -m src.datasets.base_dataset --cfg configs/gaze/recognition_image_clip.yaml --exp_name gaze/null val.batch_size 1 train.batch_size 1


python -m src.datasets.base_dataset --cfg configs/ego4d/image_pred_in8.yaml --exp_name gaze/null val.batch_size 1 train.batch_size 1 data.prediction_path data/ego4d/fake_all.json
'''
    