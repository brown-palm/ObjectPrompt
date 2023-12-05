import os
import torch
import numpy as np
from .video_task import VideoTask
from ..utils import eval_util
from ..utils import file_util
from ..models.losses import LTAWeightedLoss


class VideoClassificationTask(VideoTask):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__(cfg, steps_in_epoch)
        # TODO: check if cfg.train.checkpoint_metric is indeed in the logs
        self.loss_fn = LTAWeightedLoss(cfg.model.loss_wts_heads, cfg.model.loss_wts_temporal)

    def forward(self, batch):
        '''
        return:
            logits: [(B, Z, #verbs), (B, Z, #nouns)]
        '''
        forecast_labels_idx = batch['forecast_labels_idx']   # (B, num_actions_to_predict, 2)

        input_clips = batch['video'] if self.cfg.model.use_vid else None 
        # input_clips: [(B, num_segments, C, T_fast, H, W), ((B, num_segments, C, T_slow, H, W))] for Slowfast, or
        # [(B, num_segments, C, T, H, W)] for MViT
        mask_video = batch['mask_video'] if self.cfg.model.use_vid else None  # (B, num_segments)

        input_texts = batch['text'] if self.cfg.data.use_gt_text else None   # (B, num_segments, 2)
        mask_text = batch['mask_text'] if self.cfg.data.use_gt_text else None

        image_features = batch['image_features'] if self.cfg.model.img_feat_size > 0 else None 
        mask_image = batch['mask_image'] if self.cfg.model.img_feat_size > 0 else None  # (B, num_images)
        object_features = batch['object_features'] if self.cfg.model.obj_feat_size > 0 else None
        mask_object = batch['mask_object'] if self.cfg.model.obj_feat_size > 0 else None # (B, num_images) not (B, num_objects)
        if mask_object is not None:
            mask_object = mask_object[..., None].repeat(self.cfg.data.object.num_objects_per_image).reshape((len(forecast_labels_idx), -1))
            # (B, num_images * num_objects_per_image)
        input_pred_text = batch['pred_text'] if self.cfg.data.use_pred_text else None   # (B, Z, 2)
        mask_pred_text = batch['mask_pred_text'] if self.cfg.data.use_pred_text else None

        logits = self.model.forward(input_clips, input_texts, image_features, object_features, input_pred_text, mask_video, mask_text, mask_image, mask_object, mask_pred_text) 
        return logits

    def generate(self, batch):
        '''
        return:
            sequences_all: {sampling_method: [(B, K, Z), (B, K, Z)]}
            logits: [(B, Z, #verbs), (B, Z, #nouns)]
        '''
        logits = self.forward(batch)
        sequences_all, logits = self.model.generate(logits, k=self.cfg.model.num_sequences_to_predict)
        return sequences_all, logits

    def training_step(self, batch, batch_idx):
        forecast_labels_idx = batch['forecast_labels_idx']   # (B, num_actions_to_predict, 2)
        # sequences_all: {sampling_method: [(B, K, Z), (B, K, Z)]}
        # logits: [(B, Z, #verbs), (B, Z, #nouns)]
        sequences_all, logits = self.generate(batch)

        step_results = {}
        loss, per_step_losses = self.loss_fn(logits, forecast_labels_idx)
        step_results['loss'] = loss  # used to do backprop
        step_results['train/loss'] = loss
        self.log('train/loss_step', loss.item(), rank_zero_only=True)

        # log ED
        last_action_id = self.cfg.model.num_actions_to_predict - 1
        for sampling_method, sequences in sequences_all.items():
            aueds, eds_last = [], []
            for head_idx in range(len(logits)):
                sequences_permuted = torch.permute(sequences[head_idx], (0, 2, 1))  # (B, Z, K)
                eds = eval_util.distributed_AUED(sequences_permuted, forecast_labels_idx[..., head_idx])
                for i in range(self.cfg.model.num_actions_to_predict):
                    step_results[f'train_ED/head{head_idx}_step{i}_{sampling_method}'] = eds[f'ED_{i}']
                step_results[f'train/AUED_head{head_idx}_{sampling_method}'] = eds['AUED']
                step_results[f'train/ED_head{head_idx}_step{last_action_id}_{sampling_method}'] = eds[f'ED_{last_action_id}']
                aueds.append(eds['AUED'])
                eds_last.append(eds[f'ED_{last_action_id}'])
            step_results[f'train/AUED_mean_{sampling_method}'] = sum(aueds) / len(aueds)
            step_results[f'train/ED_step{last_action_id}_mean_{sampling_method}'] = sum(eds_last) / len(eds_last)

        # log Acc
        for head_idx in range(len(logits)):
            logits_head = logits[head_idx]  # (B, Z, C)
            for seq_idx in range(logits_head.shape[1]):
                top1_acc, top5_acc = eval_util.distributed_topk_accs(
                    logits_head[:, seq_idx], forecast_labels_idx[:, seq_idx, head_idx], (1, 5))
                step_results[f"train_acc/step{seq_idx}_head{head_idx}_top1_acc"] = top1_acc
                step_results[f"train_acc/step{seq_idx}_head{head_idx}_top5_acc"] = top5_acc
            step_results[f"train/top1_acc_head{head_idx}"] = np.mean(
                [v for k, v in step_results.items() if f"{head_idx}_top1" in k])
            step_results[f"train/top5_acc_head{head_idx}"] = np.mean(
                [v for k, v in step_results.items() if f"{head_idx}_top5" in k])
        step_results[f"train/top1_acc_mean"] = (step_results[f"train/top1_acc_head0"] + \
                                                step_results[f"train/top1_acc_head1"]) / 2
        step_results[f"train/top5_acc_mean"] = (step_results[f"train/top5_acc_head0"] + \
                                                step_results[f"train/top5_acc_head1"]) / 2

        return step_results

    def training_epoch_end(self, outputs):
        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = sum([x[key] for x in outputs]) / len(outputs)
            self.log(key, metric, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        forecast_labels_idx = batch['forecast_labels_idx']   # (B, num_actions_to_predict, 2)
        # sequences_all: {sampling_method: [(B, K, Z), (B, K, Z)]}
        # logits: [(B, Z, #verbs), (B, Z, #nouns)]
        sequences_all, logits = self.generate(batch)

        step_results = {}
        loss, per_step_losses = self.loss_fn(logits, forecast_labels_idx)
        step_results['val/loss'] = loss

        # log ED
        last_action_id = self.cfg.model.num_actions_to_predict - 1
        for sampling_method, sequences in sequences_all.items():
            aueds, eds_last = [], []
            for head_idx in range(len(logits)):
                sequences_permuted = torch.permute(sequences[head_idx], (0, 2, 1))  # (B, Z, K)
                eds = eval_util.distributed_AUED(sequences_permuted, forecast_labels_idx[..., head_idx])
                for i in range(self.cfg.model.num_actions_to_predict):
                    step_results[f'val_ED/head{head_idx}_step{i}_{sampling_method}'] = eds[f'ED_{i}']
                step_results[f'val/AUED_head{head_idx}_{sampling_method}'] = eds['AUED']
                step_results[f'val/ED_head{head_idx}_step{last_action_id}_{sampling_method}'] = eds[f'ED_{last_action_id}']
                aueds.append(eds['AUED'])
                eds_last.append(eds[f'ED_{last_action_id}'])
            step_results[f'val/AUED_mean_{sampling_method}'] = sum(aueds) / len(aueds)
            step_results[f'val/ED_step{last_action_id}_mean_{sampling_method}'] = sum(eds_last) / len(eds_last)

        # log Acc
        for head_idx in range(len(logits)):
            logits_head = logits[head_idx]  # (B, Z, C)
            for seq_idx in range(logits_head.shape[1]):
                top1_acc, top5_acc = eval_util.distributed_topk_accs(
                    logits_head[:, seq_idx], forecast_labels_idx[:, seq_idx, head_idx], (1, 5))
                step_results[f"val_acc/step{seq_idx}_head{head_idx}_top1_acc"] = top1_acc
                step_results[f"val_acc/step{seq_idx}_head{head_idx}_top5_acc"] = top5_acc
            step_results[f"val/top1_acc_head{head_idx}"] = np.mean(
                [v for k, v in step_results.items() if f"{head_idx}_top1" in k])
            step_results[f"val/top5_acc_head{head_idx}"] = np.mean(
                [v for k, v in step_results.items() if f"{head_idx}_top5" in k])
        step_results[f"val/top1_acc_mean"] = (step_results[f"val/top1_acc_head0"] + \
                                                step_results[f"val/top1_acc_head1"]) / 2
        step_results[f"val/top5_acc_mean"] = (step_results[f"val/top5_acc_head0"] + \
                                                step_results[f"val/top5_acc_head1"]) / 2

        return step_results

    def validation_epoch_end(self, outputs):
        keys = [x for x in outputs[0].keys()]
        for key in keys:
            metric = sum([x[key] for x in outputs]) / len(outputs)
            self.log(key, metric, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # sequences_all: {sampling_method: [(B, K, Z), (B, K, Z)]}
        # logits: [(B, Z, #verbs), (B, Z, #nouns)]
        sequences_all, logits = self.generate(batch)

        last_observed_ids = batch['last_observed_id']  # list of "<clip_uid>_<last_visible_action_idx>"

        # TODO: optimize sequences
        sequences = sequences_all['naive']

        step_result = {
            'last_observed_ids': last_observed_ids,
            'verb_preds': sequences[0],
            'noun_preds': sequences[1],
        }

        if self.cfg.test.gen_logits:
            step_result['verb_preds_logits'] = logits[0]  # (B, Z, #)
            step_result['noun_preds_logits'] = logits[1]
        
        return step_result

    def test_epoch_end(self, outputs):
        test_outputs = {}

        last_observed_ids = []
        for step_result in outputs:
            for last_observed_id in step_result['last_observed_ids']:
                last_observed_ids.append(last_observed_id)
        test_outputs['last_observed_ids'] = last_observed_ids

        for key in outputs[0].keys():
            if key == 'last_observed_ids':
                continue
            test_outputs[key] = torch.cat([step_result[key] for step_result in outputs], 0)

        # save submit file  {last_observed_id: {'verb': (K, Z), 'noun': (K, Z)}}
        pred_dict = {}
        for idx, last_observed_id in enumerate(test_outputs['last_observed_ids']):
            pred_dict[last_observed_id] = {
                'verb': test_outputs['verb_preds'][idx].cpu().tolist(),  # (K, Z)
                'noun': test_outputs['noun_preds'][idx].cpu().tolist(),  # (K, Z)
            }
        base_dir = self.trainer.log_dir
        save_dir = os.path.join(base_dir, 'submit.json')
        file_util.save_json(pred_dict, save_dir)

        # gen logits file  {last_observed_id: {'verb': (Z, #verbs), 'noun': (Z, #nouns)}}
        if self.cfg.test.gen_logits:
            pred_logits_dict = {}
            for idx, last_observed_id in enumerate(test_outputs['last_observed_ids']):
                pred_logits_dict[last_observed_id] = {
                    'verb': test_outputs['verb_preds_logits'][idx].cpu(),  # (Z, #)
                    'noun': test_outputs['noun_preds_logits'][idx].cpu(),  # (Z, #)
                }
            save_dir = os.path.join(base_dir, 'logits.pkl')
            file_util.save_pickle(pred_logits_dict, save_dir)
