import logging
import random

from torch.utils.data import DataLoader

from ..hparams import HParams
from ..utils import DistributedEvalSampler
from .dataset import Dataset
from .dataset import InferenceDataset
from .utils import mix_fg_bg, rglob_audio_files

logger = logging.getLogger(__name__)


def _create_datasets(hp: HParams, mode, val_size=10, seed=123):
    paths = rglob_audio_files(hp.fg_dir)
    logger.info(f"Found {len(paths)} audio files in {hp.fg_dir}")

    random.Random(seed).shuffle(paths)
    train_paths = paths[:-val_size]
    val_paths = paths[-val_size:]

    train_ds = Dataset(train_paths, hp, training=True, mode=mode)
    val_ds = Dataset(val_paths, hp, training=False, mode=mode)

    logger.info(f"Train set: {len(train_ds)} samples - Val set: {len(val_ds)} samples")

    return train_ds, val_ds


def _get_dataset(path, sr, seed=123):
    paths = rglob_audio_files(path)

    random.Random(seed).shuffle(paths)
    
    ds = InferenceDataset(paths, sr)
    
    return ds


def create_dataloaders(hp: HParams, mode):
    train_ds, val_ds = _create_datasets(hp=hp, mode=mode)

    train_dl = DataLoader(
        train_ds,
        batch_size=hp.batch_size_per_gpu,
        shuffle=True,
        num_workers=hp.nj,
        drop_last=True,
        collate_fn=train_ds.collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=hp.nj,
        drop_last=False,
        collate_fn=val_ds.collate_fn,
    )
    return train_dl, val_dl


def create_dataloader(in_dir, batch_size, sr, device, world_size):
    ds = _get_dataset(path=in_dir, sr=sr)
    distributed_sampler = DistributedEvalSampler(ds, num_replicas=world_size, rank=device)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=distributed_sampler,
        collate_fn=ds.collate_fn
    )
    return dl