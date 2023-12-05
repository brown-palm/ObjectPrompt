from .video_classification_task import VideoClassificationTask


def load_task(cfg, steps_in_epoch=1):
    return VideoClassificationTask(cfg, steps_in_epoch)