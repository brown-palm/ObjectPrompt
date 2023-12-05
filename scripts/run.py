import pprint
import tensorboard
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed as dist

from src.utils.parser import load_config, parse_args
from src.utils import logging
from src.datasets.base_dataset import BaseDataModule
from src.tasks import load_task


logger = logging.get_logger(__name__)


def cleanup():
    dist.destroy_process_group()


def train(args, cfg):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}'.format(args.exp_name)

    # dataset module
    dm = BaseDataModule(cfg)
    dm.setup(stage='fit')
    dm.train_dataloader()  # initialize dm.train_loader
    steps_in_epoch = len(dm.train_loader) // cfg.num_gpus
    print('steps_in_epoch: ', steps_in_epoch)

    # task module
    task = load_task(cfg, steps_in_epoch)

    # trainer setting
    tb_logger = TensorBoardLogger(save_dir='.', name='lightning_logs', version=log_path)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.checkpoint_metric,
        mode=cfg.train.checkpoint_mode,
        save_last=True,
        save_top_k=1
    )
    learning_rate_callback = LearningRateMonitor()
    trainer_args = {}
    if cfg.train.strategy == 'ddp':
        trainer_args = {
            'accelerator': 'gpu',
            'devices': cfg.num_gpus,   # devices=[1, 3] to specify GPUs
            'strategy': 'ddp',
            # 'strategy': DDPStrategy(find_unused_parameters=False, static_graph=True, gradient_as_bucket_view=True),
        }
    elif cfg.train.strategy == 'cpu':
        trainer_args = {}
    trainer = Trainer(
        max_epochs=cfg.solver.num_epochs,

        benchmark=True,

        fast_dev_run=args.fast_dev_run,
        limit_train_batches=cfg.train.limit_train_batches,  # to avoid tensorboard issue
        limit_val_batches=cfg.train.limit_val_batches,  # to avoid tensorboard issue

        logger=tb_logger,
        callbacks=[learning_rate_callback, checkpoint_callback],

        **trainer_args,
    )
    trainer.fit(model=task, datamodule=dm, ckpt_path=cfg.ckpt_path)
    cleanup()
    return checkpoint_callback.best_model_path


def test(args, cfg, ckpt_path):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}/test'.format(args.exp_name)

    # dataset module
    dm = BaseDataModule(cfg)

    # task module
    task = load_task(cfg)

    tb_logger = TensorBoardLogger(save_dir='.', name='lightning_logs', version=log_path)

    # devices=1 to avoid distributed sampler.
    trainer = Trainer(
        accelerator = 'gpu',
        logger = tb_logger,
        devices = 1,
        limit_test_batches = cfg.test.limit_test_batches,
    )
    trainer.test(model=task, datamodule=dm, ckpt_path=ckpt_path)


def val(args, cfg, ckpt_path):
    seed_everything(cfg.seed)

    # setup logger
    log_path = '{}/val'.format(args.exp_name)

    # dataset module
    dm = BaseDataModule(cfg)
    dm.setup('fit')

    # task module
    task = load_task(cfg)

    tb_logger = TensorBoardLogger(save_dir='.', name='lightning_logs', version=log_path)

    trainer_args = {}
    if cfg.train.strategy == 'ddp':
        trainer_args = {
            'accelerator': 'gpu',
            'devices': cfg.num_gpus,   # devices=[1, 3] to specify GPUs
            'strategy': 'ddp',
            # 'strategy': DDPStrategy(find_unused_parameters=False, static_graph=True, gradient_as_bucket_view=True),
        }
    elif cfg.train.strategy == 'cpu':
        trainer_args = {}

    trainer = Trainer(
        limit_val_batches=cfg.train.limit_val_batches,  # to avoid tensorboard issue
        logger=tb_logger,
        **trainer_args,
    )

    trainer.validate(model=task, dataloaders=dm.val_dataloader(), ckpt_path=ckpt_path)


def main():
    # parse arg and cfg
    args = parse_args()
    cfg = load_config(args)

    # set seed
    seed_everything(cfg.seed)

    ckpt_path = cfg.ckpt_path
    if cfg.val.val_only:
        val(args, cfg, ckpt_path)
    else:
        if cfg.train.enable:
            ckpt_path = train(args, cfg)
        if cfg.test.enable:
            test(args, cfg, ckpt_path)


if __name__ == "__main__":
    main()


'''
lta_v_i_o CLIP:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4dv2/sf_video_image_object_clip.yaml \
--exp_name ego4dv2/sf_video_image_object_clip_lr1e-3_epoch30 \
data.image.base_path IMG_EMB_FILE_PATH \
data.object.base_path OBJ_EMB_FILE_PATH

lta_v_i_o GLIP:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4dv2/sf_video_image_object_glip.yaml \
--exp_name ego4dv2/sf_video_image_object_clip_lr1e-3_epoch30 \
data.image.base_path IMG_EMB_FILE_PATH \
data.object.base_path OBJ_EMB_FILE_PATH

lta_v:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video.yaml \
--exp_name ego4d/sf_video


test:
NCCL_P2P_DISABLE="1" python -m scripts.run \
--cfg configs/ego4d/sf_video.yaml \
--exp_name ego4d/sf_video \
train.enable False \
test.enable True \
ckpt_path CKPT_PATH 
'''