import torch
from ..optimizers import lr_scheduler
from ..models.model import ClassificationModule
from pytorch_lightning.core import LightningModule


class VideoTask(LightningModule):
    def __init__(self, cfg, steps_in_epoch):
        super().__init__()
        self.cfg = cfg
        self.steps_in_epoch = steps_in_epoch
        self.save_hyperparameters()
        self.model = self.build_model()

    def build_model(self):
        if self.cfg.model.model == 'classification':
            model = ClassificationModule(self.cfg)
            if self.cfg.model.use_vid and len(self.cfg.pretrained_backbone_path) > 0:
                load_pretrained_backbone(model, self.cfg)
            return model
        else:
            raise NotImplementedError(f'model {self.cfg.model.model} not implemmented')

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        # steps_in_epoch = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        return lr_scheduler.lr_factory(self.model, self.cfg, self.steps_in_epoch)


def load_pretrained_backbone(model, cfg):
    backbone = model.backbone
    # Load slowfast weights into backbone submodule
    ckpt = torch.load(
        cfg.pretrained_backbone_path,
        map_location=lambda storage, loc: storage,
    )

    def remove_first_module(key):
        return ".".join(key.split(".")[1:])

    key = "state_dict" if "state_dict" in ckpt.keys() else "model_state"

    state_dict = {
        remove_first_module(k): v
        for k, v in ckpt[key].items()
        if "head" not in k
    }

    missing_keys, unexpected_keys = backbone.load_state_dict(
        state_dict, strict=False
    )

    print('missing', missing_keys)
    print('unexpected', unexpected_keys)

    # Ensure only head key is missing.w
    assert len(unexpected_keys) == 0
    assert all(["head" in x for x in missing_keys])

    for key in missing_keys:
        print(f"Could not load {key} weights")
