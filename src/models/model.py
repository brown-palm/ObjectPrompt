import torch
from torch import nn
from torch.distributions import Categorical
import math
from .backbones import MViT, SlowFast


class DropToken(nn.Module):
    def __init__(self, dim, drop_prob) -> None:
        super().__init__()
        self.dim = dim  # (D, )
        self.pad = torch.nn.parameter.Parameter(torch.randn(dim))  # (D)
        # TODO: figure out initialization
        # self.pad = torch.nn.parameter.Parameter(torch.zeros(dim))  # (D)
        self.drop_prob = drop_prob

    def forward(self, x):
        # x: (..., D)
        if self.training:
            input_shape = x.shape
            x = torch.reshape(x, (-1, input_shape[-1]))
            rand_tensor = torch.rand(x.shape[0], device=x.device)
            x[rand_tensor < self.drop_prob, :] = self.pad
            x = torch.reshape(x, input_shape)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)


class MMPositionalEncoding(nn.Module):
    def __init__(self, cfg, d_model, num_input_clips=1, dropout=0.1, max_len=100,
                 use_vid=True, use_text=False, use_img=True, use_obj=True):
        super(MMPositionalEncoding, self).__init__()
        self.cfg = cfg
        self.num_input_clips = num_input_clips
        self.d_model = d_model
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

        modality_tokens_init = torch.randn((6, d_model))  # (6, D)
        self.modality_tokens = torch.nn.parameter.Parameter(modality_tokens_init)
        rel_pe_init = torch.zeros((cfg.data.image.num_images_per_segment, 1, d_model))  # (L, 1, D)
        if use_img or use_obj:
            self.rel_pe = torch.nn.parameter.Parameter(rel_pe_init)
        else:
            self.rel_pe = torch.nn.parameter.Parameter(rel_pe_init, requires_grad=False)
        

    def forward(self, vid_feat=None, text_feat=None, img_feat=None, obj_feat=None, pred_text_feat=None, query_feat=None):
        '''
        vid_feat: None or (num_input_clips, B, D)
        img_feat: None or (num_input_clips * num_images_per_clip, B, D)
        obj_feat: None or (num_input_clips * num_images_per_clip * num_objects_per_clip, B, D)
        pred_text_feat: None or (Z, B, D)
        query_feat: None or (Z, B, D)
        '''
        # TODO: reverse PE for inputs
        num_input_clips = self.num_input_clips
        num_img_per_clip = self.cfg.data.image.num_images_per_segment
        num_obj_per_image = self.cfg.data.object.num_objects_per_image

        abs_pe = self.pe[:num_input_clips, :, :]

        if vid_feat is not None:
            # TODO: check norm. vid_feat is very different from abs_pe and modality_tokens
            vid_feat = vid_feat + abs_pe + self.modality_tokens[0]
            vid_feat = self.layernorm(vid_feat)
            vid_feat = self.dropout(vid_feat)
        if text_feat is not None:
            # TODO: check norm. vid_feat is very different from abs_pe and modality_tokens
            text_feat = text_feat + abs_pe + self.modality_tokens[1]
            text_feat = self.layernorm(text_feat)
            text_feat = self.dropout(text_feat)
        if img_feat is not None:
            # img_feat: (num_input_clips * num_img_per_clip, B, D)
            # abs_pe: (num_input_clips, 1, D)
            # rel_pe: (num_img_per_clip, 1, D)
            batch_size = img_feat.shape[1]
            # (num_input_clips, num_img_per_clip, B, D)
            img_feat = torch.reshape(img_feat, (num_input_clips, num_img_per_clip, batch_size, -1))
            # (num_input_clips, num_img_per_clip, B, D)
            img_feat = img_feat + torch.unsqueeze(abs_pe, 1) + self.rel_pe + self.modality_tokens[2]
            img_feat = torch.reshape(img_feat, (num_input_clips * num_img_per_clip, batch_size, -1))
            img_feat = self.layernorm(img_feat)
            img_feat = self.dropout(img_feat)
        if obj_feat is not None:
            # obj_feat: (num_input_clips * num_img_per_clip * num_obj_per_image, B, D)
            # abs_pe: (num_input_clips, 1, D)
            # rel_pe: (num_img_per_clip, 1, D)
            # ensure that objects in the same timestamp have the same pos encoding.
            batch_size = obj_feat.shape[1]
            # (num_input_clips, num_img_per_clip, num_obj_per_image, B, D)
            obj_feat = torch.reshape(obj_feat, (num_input_clips, num_img_per_clip, num_obj_per_image, batch_size, -1))
            # (num_input_clips, num_img_per_clip, num_obj_per_image, B, D)
            obj_feat = obj_feat + abs_pe.unsqueeze(1).unsqueeze(1) + torch.unsqueeze(self.rel_pe, 1) + self.modality_tokens[3]
            obj_feat = torch.reshape(obj_feat, (num_input_clips * num_img_per_clip * num_obj_per_image, batch_size, -1))
            obj_feat = self.layernorm(obj_feat)
            obj_feat = self.dropout(obj_feat)
        if pred_text_feat is not None:
            len_pred_text_feat = pred_text_feat.shape[0]
            abs_pe_pos = self.pe[num_input_clips:num_input_clips + len_pred_text_feat, :, :]
            pred_text_feat = pred_text_feat + abs_pe_pos + self.modality_tokens[4]
            # pred_text_feat = pred_text_feat + self.modality_tokens[3]
            pred_text_feat = self.layernorm(pred_text_feat)
            pred_text_feat = self.dropout(pred_text_feat)
        if query_feat is not None:
            len_query_feat = query_feat.shape[0]
            abs_pe_pos = self.pe[num_input_clips:num_input_clips + len_query_feat, :, :]
            query_feat = query_feat + abs_pe_pos + self.modality_tokens[5]
            # query_feat = query_feat + self.modality_tokens[3]
            query_feat = self.layernorm(query_feat)
            query_feat = self.dropout(query_feat)

        return vid_feat, text_feat, img_feat, obj_feat, pred_text_feat, query_feat

# --------------------------------------------------------------------#

class PredictiveTransformerEncoder(nn.Module):
    def __init__(self, cfg, num_queries):
        super().__init__()
        self.cfg = cfg
        dim_in = cfg.model.base_feat_size
        num_heads = cfg.model.pte.num_heads
        num_layers = cfg.model.pte.num_layers
        self.num_queries = num_queries

        # TODO: check initialization
        self.queries = torch.nn.parameter.Parameter(torch.randn((cfg.model.num_actions_to_predict, dim_in)))  # (Z, D)
        # self.queries = torch.nn.parameter.Parameter(torch.zeros((num_queries, dim_in)))  # (Z, D)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads, dropout=cfg.model.pte.enc_dropout),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = MMPositionalEncoding(
            cfg, dim_in, num_input_clips=self.cfg.data.input_segments[1]-self.cfg.data.input_segments[0], 
            dropout=cfg.model.pte.pos_dropout, use_vid=cfg.model.use_vid, use_text=self.cfg.data.use_gt_text, use_img=cfg.model.img_feat_size > 0, use_obj=cfg.model.obj_feat_size > 0)

    def forward(self, clip_features, text_features, image_features, object_features, pred_text_features, mask_video, mask_text, mask_image, mask_object, mask_pred_text):
        # clip_features, image_features, object_features: (B, L, D)  L could be different
        if clip_features is not None:
            # print(clip_features.shape)  # (12, 2048)
            batch_size, num_inputs, _ = clip_features.shape
            clip_features = torch.transpose(clip_features, 0, 1)   # (num_inputs, B, D)
        if text_features is not None:
            batch_size, num_inputs, _ = text_features.shape
            text_features = torch.transpose(text_features, 0, 1)   # (num_inputs, B, D)
        if image_features is not None:
            batch_size, num_inputs, _ = image_features.shape
            image_features = torch.transpose(image_features, 0, 1)   # (num_inputs, B, D)
        if object_features is not None:
            batch_size, num_inputs, _ = object_features.shape
            object_features = torch.transpose(object_features, 0, 1)   # (num_inputs, B, D)
        if pred_text_features is not None:
            batch_size, num_inputs, _ = pred_text_features.shape
            pred_text_features = torch.transpose(pred_text_features, 0, 1)   # (Z, B, D)

        queries = self.queries.expand(batch_size, -1, -1).permute(1, 0, 2)  # (Z, B, D)

        clip_features, text_features, image_features, object_features, pred_text_features, queries = self.pos_encoder(
            clip_features, text_features, image_features, object_features, pred_text_features, queries)
        x = torch.cat([feat for feat in [clip_features, text_features, image_features, 
                        object_features, pred_text_features, queries] if feat is not None], dim=0)

        mask_query = torch.zeros((batch_size, self.num_queries), dtype=torch.bool, device=queries.device)  # (B, Z)
        mask = torch.cat([feat for feat in [mask_video, mask_text, mask_image,
                    mask_object, mask_query, mask_pred_text] if feat is not None], dim=-1)
        x = self.encoder(x, src_key_padding_mask=mask)  # (num_inputs + Z, B, D)
        x = x[-self.cfg.model.num_actions_to_predict:, ...]  # (Z, B, D)
        return torch.transpose(x, 0, 1)  # (B, Z, D)

    def get_attn_rollout(self, clip_features, image_features, object_features):
        # clip_features, image_features, object_features: (B, L, D)  L could be different
        if clip_features is not None:
            batch_size, num_inputs, _ = clip_features.shape
            clip_features = torch.transpose(clip_features, 0, 1)   # (num_inputs, B, D)
        if image_features is not None:
            batch_size, num_inputs, _ = image_features.shape
            image_features = torch.transpose(image_features, 0, 1)   # (num_inputs, B, D)
        if object_features is not None:
            batch_size, num_inputs, _ = object_features.shape
            object_features = torch.transpose(object_features, 0, 1)   # (num_inputs, B, D)

        queries = self.queries.expand(batch_size, -1, -1).permute(1, 0, 2)  # (Z, B, D)

        clip_features, image_features, object_features, queries = self.pos_encoder(
            clip_features, image_features, object_features, queries)
        x = torch.cat([feat for feat in [clip_features, image_features,
                      object_features, queries] if feat is not None], dim=0)

        # x = self.encoder(x) # (num_inputs + Z, B, D)
        src = x   # [L, B, E]
        batch_size, seq_len = src.shape[1], src.shape[0]
        batch_eye = torch.eye(seq_len, device=src.device).expand(
            batch_size, -1, -1)
        batch_zeros = torch.zeros(
            seq_len, seq_len, device=src.device).expand(batch_size, -1, -1)
        attn_maps = []
        for encoder_layer in self.encoder.layers:
            src2, attn_map = encoder_layer.self_attn(src, src, src)
            # attn_map: [B, L, L]. Each row is a query's attn to all keys.
            batch_size, seq_len = attn_map.shape[0], attn_map.shape[1]
            # consider residual connection
            attn_map = (batch_eye + attn_map) / 2
            attn_maps.append(attn_map)
            src = src + encoder_layer.dropout1(src2)
            src = encoder_layer.norm1(src)
            src2 = encoder_layer.linear2(encoder_layer.dropout(
                encoder_layer.activation(encoder_layer.linear1(src))))
            src = src + encoder_layer.dropout2(src2)
            src = encoder_layer.norm2(src)

        # compute rollout
        rollout_attn = attn_maps[0]  # [B, L, L]
        # helper_util.save_json(rollout_attn.tolist(), 'debug/attention_0.json')
        for i, attn_map in enumerate(attn_maps[1:]):
            # helper_util.save_json(rollout_attn.tolist(), 'debug/attention_{}.json'.format(i + 1))
            rollout_attn = torch.bmm(attn_map, rollout_attn)

        # # eliminate diagonal
        # rollout_attn = torch.where(batch_eye==1, batch_zeros, rollout_attn)  # (B, L, L)
        for i in range(rollout_attn.shape[0]):
            for j in range(rollout_attn.shape[1]):
                rollout_attn[i, j, j] = 0
        # (B, Z, L)
        rollout_attn = rollout_attn[:, -self.cfg.model.num_actions_to_predict:, :]
        return rollout_attn


# --------------------------------------------------------------------#

class MLPDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = nn.Linear(cfg.model.base_feat_size, sum(cfg.model.num_classes))
    
    def forward(self, x):
        # x: (B, Z, D)
        logits = self.head(x)  # (B, Z, #verbs + #nouns)
        logits = torch.split(logits, self.cfg.model.num_classes, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return logits
    

class MultiHeadDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.head = nn.Linear(cfg.model.base_feat_size, sum(cfg.model.num_classes) * cfg.model.num_actions_to_predict)
    
    def forward(self, x):
        # x: (B, 1, D)
        logits = self.head(x)  # (B, 1, Z*(#verbs + #nouns))
        logits = logits.reshape((logits.shape[0], self.cfg.model.num_actions_to_predict, -1))  # (B, Z, (#verbs + #nouns))
        logits = torch.split(logits, self.cfg.model.num_classes, dim=-1)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return logits
    
# --------------------------------------------------------------------#


class ClassificationModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.model.base_feat_size > 0:
            self.build_clip_backbone()
        if cfg.data.use_gt_text or cfg.data.use_pred_text:
            self.build_text_encoder()
        if cfg.model.img_feat_size > 0:
            self.build_img_feat_proj()
        if cfg.model.obj_feat_size > 0:
            self.build_obj_feat_proj()
        self.build_aggregator()
        self.build_decoder()

    def build_clip_backbone(self):
        cfg = self.cfg
        # Use head as a projection.
        # Slowfast needs a head to convert feature maps to a representation.
        if cfg.model.backbone == "mvit":
            self.backbone = MViT(cfg, with_head=True, num_classes=cfg.model.base_feat_size, head_dropout_rate=0)
        elif cfg.model.backbone == "slowfast":
            self.backbone = SlowFast(cfg, with_head=True, num_classes=cfg.model.base_feat_size, head_dropout_rate=0)
        else:
            raise NotImplementedError(f"backbone {cfg.model.backbone} not supported")

        if cfg.model.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def build_text_encoder(self):
        emb_size = self.cfg.model.text_feat_size
        self.embedding_layer_verb = nn.Embedding(self.cfg.model.num_classes[0] + 1, emb_size)
        self.embedding_layer_noun = nn.Embedding(self.cfg.model.num_classes[1] + 1, emb_size)
        self.text_proj = nn.Linear(emb_size+emb_size, self.cfg.model.base_feat_size)
        if self.cfg.model.drop_text > 0:
            self.drop_text = DropToken(self.cfg.model.text_feat_size * len(self.cfg.model.num_classes), self.cfg.model.drop_text)

    def build_img_feat_proj(self):
        proj_size = self.cfg.model.base_feat_size
        self.img_feat_proj = nn.Linear(self.cfg.model.img_feat_size, proj_size)
        if self.cfg.model.drop_img > 0:
            self.drop_img = DropToken(self.cfg.model.img_feat_size, self.cfg.model.drop_img)

    def build_obj_feat_proj(self):
        if self.cfg.model.drop_obj > 0:
            self.drop_obj = DropToken(self.cfg.model.obj_feat_size, self.cfg.model.drop_obj)
        if self.cfg.model.use_obj_loc:
            self.obj_loc_proj = nn.Linear(4, self.cfg.model.obj_feat_size)
        if self.cfg.model.use_obj_cat:
            self.obj_cat_proj = nn.Embedding(self.cfg.data.object.num_object_classes, self.cfg.model.obj_feat_size)
        if self.cfg.model.obj_thre > 0:
            # self.obj_pad = torch.nn.parameter.Parameter(torch.randn(self.cfg.model.obj_feat_size))  # (obj_feat_size)
            # TODO: figure out initialization
            self.obj_pad = torch.nn.parameter.Parameter(torch.zeros(self.cfg.model.obj_feat_size))  # (obj_feat_size)
        proj_size = self.cfg.model.base_feat_size
        self.obj_feat_proj = nn.Linear(self.cfg.model.obj_feat_size, proj_size)

    def build_aggregator(self):
        cfg = self.cfg
        aggregator = None
        if cfg.model.aggregator == 'pte':
            aggregator = PredictiveTransformerEncoder(cfg, num_queries=cfg.model.num_actions_to_predict)
        elif cfg.model.aggregator == 'trf':
            aggregator = PredictiveTransformerEncoder(cfg, num_queries=1)
        else:
            raise NotImplementedError(f"aggregator {cfg.model.aggregator} not supported")
        self.aggregator = aggregator
    
    def build_decoder(self):
        cfg = self.cfg
        decoder = None
        if cfg.model.decoder == 'mlp':
            decoder = MLPDecoder(cfg)
        elif cfg.model.aggregator == 'mlp_multihead':
            decoder = MultiHeadDecoder(cfg)
        else:
            raise NotImplementedError(f"decoder {cfg.model.decoder} not supported")
        self.decoder = decoder

    def encode_clips(self, clips):
        # clips: [(B, num_segments, C, T_fast, H, W), ((B, num_segments, C, T_slow, H, W))] for Slowfast, or
        # [(B, num_segments, C, T, H, W)] for MViT
        # outputs should be (B, num_segments, D_clip)
        assert isinstance(clips, list) and len(clips) >= 1

        num_segments = clips[0].shape[1]
        batch_size = clips[0].shape[0]

        for i in range(len(clips)):
            clips[i] = clips[i].reshape((-1,) + tuple(clips[i].shape[-4:]))  # (B * num_segments, C, T_fast, H, W)
        features = self.backbone(clips)  # (B * num_segments, D_clip)
        # TODO: check memory
        features = features.reshape((batch_size, num_segments, -1))  # (B, num_segments, D_clip)

        return features

    def encode_text(self, texts):
        verb_features = self.embedding_layer_verb(texts[..., 0])  # (B, num_segments, text_feat_size) or (B, num_segments, num_seqs, text_feat_size)
        if len(verb_features.shape) == 4:
            verb_features = verb_features.sum(dim=-2)  # (B, num_segments, text_feat_size)
        noun_features = self.embedding_layer_noun(texts[..., 1])  # (B, num_segments, text_feat_size) or (B, num_segments, num_seqs, text_feat_size)
        if len(noun_features.shape) == 4:
            noun_features = noun_features.sum(dim=-2)  # (B, num_segments, text_feat_size)
        text_features = torch.cat([verb_features, noun_features], dim=-1)   # (B, num_segments, text_feat_size * 2)
        if self.cfg.model.drop_text > 0:
            text_features = self.drop_text(text_features)
        text_features = self.text_proj(text_features)  # (B, num_segments, base_feat_size)
        return text_features
    
    def encode_image_features(self, x):
        if self.cfg.model.drop_img > 0:
            x = self.drop_img(x)
        out = self.img_feat_proj(x)
        return out

    def encode_object_features(self, x):
        # x: (..., D)
        out = x[..., :self.cfg.model.obj_feat_size]
        if self.cfg.model.use_obj_loc:
            loc_feat = self.obj_loc_proj(x[..., -6:-2])  # (..., obj_feat_size)
            out = out + loc_feat   # (..., proj_size)
        if self.cfg.model.use_obj_cat:
            indices = x[..., -2].to(torch.long)  # (...)
            cat_feat = self.obj_cat_proj(indices)  # (..., proj_size)
            out = out + cat_feat  # (..., proj_size)
        if self.cfg.model.obj_thre > 0:
            thres = x[..., -1]  # (...)
            out[thres <= self.cfg.model.obj_thre, :] = self.obj_pad
        if self.cfg.model.drop_obj > 0:
            out = self.drop_obj(out)
        out = self.obj_feat_proj(out)
        return out

    def aggregate(self, clip_features, text_features, image_features, object_features, pred_text_features, mask_video, mask_text, mask_image, mask_object, mask_pred_text):
        return self.aggregator(clip_features, text_features, image_features, object_features, pred_text_features, mask_video, mask_text, mask_image, mask_object, mask_pred_text)

    def decode(self, features):
        return self.decoder(features)

    def forward(self, clips, texts, image_features, object_features, pred_text, mask_video, mask_text, mask_image, mask_object, mask_pred_text):
        # clips: list[2]. For slowfast: clips[0]: (B, num_input_clips, C, T_slow, H, W), clips[1]: (B, num_input_clips, C, T_fast, H, W). For MViT, x[0]: (B, num_input_clips, C, T, H, W)
        # texts: (B, num_input_clips, 2)
        # input_pred_text: (B, Z, 2)
        # image_features: (B, num_input_clips * num_images_per_clip, D_img)
        # object_features: (B, num_input_clips * num_images_per_clip * num_objects_per_image, D_obj)

        # encode
        clip_features = self.encode_clips(clips) if clips is not None else None
        text_features = self.encode_text(texts) if texts is not None else None
        if image_features is not None:
            image_features = self.encode_image_features(image_features)
        if object_features is not None:
            object_features = self.encode_object_features(object_features)
        pred_text_features = self.encode_text(pred_text) if pred_text is not None else None

        # aggregate
        features = self.aggregate(clip_features, text_features, image_features, object_features, pred_text_features, mask_video, mask_text, mask_image, mask_object, mask_pred_text)  # (B, ?, D)

        # decode
        x = self.decode(features)  # [(B, Z, #verbs), (B, Z, #nouns)]
        return x
    

    def generate(self, logits, k=1):
        x = logits # [(B, Z, C), (B, Z, C)]

        def match(v, n):
            if f'{v}_{n}' in self.vocab:
                return True
            return False

        results_all = {}  # sampling_method --> results

        for sampling_method in self.cfg.model.sampleing_method:
            results = []
            if sampling_method == 'naive':
                for head_x in x:
                    if k > 1:
                        preds_dist = Categorical(
                            logits=torch.clamp(head_x, min=0))
                        preds = [preds_dist.sample()
                                 for _ in range(k)]  # [(B, Z)] * K
                    elif k == 1:
                        preds = [head_x.argmax(2)]
                    head_x = torch.stack(preds, dim=1)  # (B, K, Z)
                    results.append(head_x)
            elif sampling_method == 'action_sample':
                num_tries = 20
                [head_verb, head_noun] = x
                batch_size, Z = head_verb.shape[0], head_verb.shape[1]
                preds_verb_dist = Categorical(
                    logits=torch.clamp(head_verb, min=0))
                preds_noun_dist = Categorical(
                    logits=torch.clamp(head_noun, min=0))
                verb_sampled = [preds_verb_dist.sample()
                                for _ in range(num_tries)]  # (B, Z) * num_tries
                noun_sampled = [preds_noun_dist.sample()
                                for _ in range(num_tries)]  # (B, Z) * num_tries
                verb_sampled = torch.stack(
                    verb_sampled, dim=0)  # (num_tries, B, Z)
                noun_sampled = torch.stack(
                    noun_sampled, dim=0)  # (num_tries, B, Z)
                verb_sampled_k = verb_sampled[:k, ...]  # (K, B, Z)
                noun_sampled_k = noun_sampled[:k, ...]  # (K, B, Z)
                matched_num = torch.zeros_like(
                    verb_sampled_k[0, ...])  # (B, Z)
                for i in range(num_tries):
                    for b in range(batch_size):
                        for z in range(Z):
                            # Do not use k after this loop
                            verb_idx, noun_idx = verb_sampled[i,
                                                              b, z], noun_sampled[i, b, z]
                            if not match(verb_idx, noun_idx):
                                continue
                            if matched_num[b, z] >= k:
                                continue
                            verb_sampled_k[matched_num[b, z], b, z] = verb_idx
                            noun_sampled_k[matched_num[b, z], b, z] = noun_idx
                            matched_num[b, z] += 1
                results = [verb_sampled_k.permute(1, 0, 2), noun_sampled_k.permute(
                    1, 0, 2)]  # (B, K, Z), (B, K, Z)
                # print(torch.mean(matched_num.to(torch.float)))
            elif sampling_method == 'action_max':
                num_tries = 5
                [head_verb, head_noun] = x  # (B, Z, C), (B, Z, C)
                batch_size, Z, _ = head_verb.shape
                pred_verb_score, pred_verb_idx = torch.sort(
                    head_verb, descending=True)  # (B, Z, num_verbs)
                pred_noun_score, pred_noun_idx = torch.sort(
                    head_noun, descending=True)  # (B, Z, num_nouns)
                # (B, Z, num_tries)
                pred_verb_score, pred_verb_idx = pred_verb_score[...,
                                                                 :num_tries], pred_verb_idx[..., :num_tries]
                # (B, Z, num_tries)
                pred_noun_score, pred_noun_idx = pred_noun_score[...,
                                                                 :num_tries], pred_noun_idx[..., :num_tries]
                verb_sampled, noun_sampled = torch.zeros((batch_size, Z, k), device=head_verb.device), torch.zeros(
                    (batch_size, Z, k), device=head_verb.device)  # (B, Z, K)
                for b in range(batch_size):
                    for z in range(Z):
                        info = []
                        for t1 in range(num_tries):
                            verb_idx = pred_verb_idx[b, z, t1]
                            verb_score = pred_verb_score[b, z, t1]
                            for t2 in range(num_tries):
                                noun_idx = pred_noun_idx[b, z, t2]
                                noun_score = pred_noun_score[b, z, t2]
                                if match(verb_idx, noun_idx):
                                    info.append(
                                        [verb_idx, noun_idx, 1e4 + verb_score*noun_score])
                                else:
                                    info.append(
                                        [verb_idx, noun_idx, verb_score*noun_score])
                        info.sort(key=lambda x: x[-1], reverse=True)
                        for i in range(k):
                            verb_sampled[b, z, i] = info[i][0]
                            noun_sampled[b, z, i] = info[i][1]
                results = [verb_sampled.permute(0, 2, 1), noun_sampled.permute(
                    0, 2, 1)]  # (B, K, Z), (B, K, Z)
            else:
                raise NotImplementedError(
                    f'sampling method {self.cfg.model.sampling_method} not implemented')
            results_all[sampling_method] = results
        return results_all, x