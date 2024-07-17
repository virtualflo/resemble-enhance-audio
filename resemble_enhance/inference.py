import logging
import time

import torch
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram
from tqdm import trange, tqdm

from resemble_enhance.data import create_dataloader

from .hparams import HParams

logger = logging.getLogger(__name__)


@torch.inference_mode()
def inference_chunk(model, dwav, device, npad=441):
    length = dwav.shape[-1]
    abs_max = dwav.abs().max().clamp(min=1e-7)

    assert dwav.dim() == 1, f"Expected 1D waveform, got {dwav.dim()}D"
    dwav = dwav.to(device)
    dwav = dwav / abs_max  # Normalize
    dwav = F.pad(dwav, (0, npad))
    with torch.no_grad():
        hwav = model(dwav[None])[0].cpu()  # (T,)
    hwav = hwav[:length]  # Trim padding
    hwav = hwav * abs_max  # Unnormalize

    return hwav


def compute_corr(x, y):
    return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()


def compute_offset(chunk1, chunk2, sr=44100):
    """
    Args:
        chunk1: (T,)
        chunk2: (T,)
    Returns:
        offset: int, offset in samples such that chunk1 ~= chunk2.roll(-offset)
    """
    hop_length = sr // 200  # 5 ms resolution
    win_length = hop_length * 4
    n_fft = 2 ** (win_length - 1).bit_length()

    mel_fn = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=80,
        f_min=0.0,
        f_max=sr // 2,
    )

    spec1 = mel_fn(chunk1).log1p()
    spec2 = mel_fn(chunk2).log1p()

    corr = compute_corr(spec1, spec2)  # (F, T)
    corr = corr.mean(dim=0)  # (T,)

    argmax = corr.argmax().item()

    if argmax > len(corr) // 2:
        argmax -= len(corr)

    offset = -argmax * hop_length

    return offset


def merge_chunks(chunks, chunk_length, hop_length, sr=44100, length=None):
    signal_length = (len(chunks) - 1) * hop_length + chunk_length
    overlap_length = chunk_length - hop_length
    signal = torch.zeros(signal_length, device=chunks[0].device)

    fadein = torch.linspace(0, 1, overlap_length, device=chunks[0].device)
    fadein = torch.cat([fadein, torch.ones(hop_length, device=chunks[0].device)])
    fadeout = torch.linspace(1, 0, overlap_length, device=chunks[0].device)
    fadeout = torch.cat([torch.ones(hop_length, device=chunks[0].device), fadeout])

    for i, chunk in enumerate(chunks):
        start = i * hop_length
        end = start + chunk_length

        if len(chunk) < chunk_length:
            chunk = F.pad(chunk, (0, chunk_length - len(chunk)))

        if i > 0:
            pre_region = chunks[i - 1][-overlap_length:]
            cur_region = chunk[:overlap_length]
            offset = compute_offset(pre_region, cur_region, sr=sr)
            start -= offset
            end -= offset

        if i == 0:
            chunk = chunk * fadeout
        elif i == len(chunks) - 1:
            chunk = chunk * fadein
        else:
            chunk = chunk * fadein * fadeout

        signal[start:end] += chunk[: len(signal[start:end])]

    signal = signal[:length]

    return signal


def remove_weight_norm_recursively(module):
    for _, module in module.named_modules():
        try:
            remove_parametrizations(module, "weight")
        except Exception:
            pass


def inference(model, dwav, sr, device, chunk_seconds: float = 30.0, overlap_seconds: float = 1.0):
    remove_weight_norm_recursively(model)

    hp: HParams = model.hp

    dwav = resample(
        dwav,
        orig_freq=sr,
        new_freq=hp.wav_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    del sr  # Everything is in hp.wav_rate now

    sr = hp.wav_rate

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length

    chunks = []
    for start in trange(0, dwav.shape[-1], hop_length):
        chunks.append(inference_chunk(model, dwav[start : start + chunk_length], device))

    hwav = merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=dwav.shape[-1])

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.3f} s, {hwav.shape[-1] / elapsed_time / 1000:.3f} kHz")

    return hwav, sr


def parallel_inference(model, in_dir, out_path, batch_size, device, world_size):
    dist.init_process_group("nccl", rank=device, world_size=world_size)
    remove_weight_norm_recursively(model)
    hp: HParams = model.hp
    ddp_model = DDP(model, device_ids=[device])

    loader = create_dataloader(in_dir,batch_size,hp.wav_rate,device,world_size)
    
    for batch in loader:
        pbar = tqdm(zip(batch["audios"],batch["paths"]))
        for wav,in_path in pbar:
            out_dir = out_path / in_path.relative_to(in_dir)
            if out_dir.exists():
                pbar.set_description(f"{out_dir} Already processed!!!")
                continue
            start_time = time.perf_counter()
            hwav = inference_chunk(ddp_model,wav,device)
            pbar.set_description(f"Processing {out_dir}")
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_dir, hwav[None], hp.wav_rate)
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Elapsed time: {elapsed_time:.3f} s, {hwav.shape[-1] / elapsed_time / 1000:.3f} kHz")
        torch.cuda.empty_cache()