import argparse
import random
import time
from pathlib import Path

import torch
import torchaudio
import torch.multiprocessing as mp
from tqdm import tqdm

from .inference import denoise, enhance, parallel_denoise, parallel_enhance


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", type=Path, help="Path to input audio folder")
    parser.add_argument("out_dir", type=Path, help="Output folder")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Path to the enhancer run folder, if None, use the default model",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".wav",
        help="Audio file suffix",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation, recommended to use CUDA",
    )
    parser.add_argument(
        "--denoise_only",
        action="store_true",
        help="Only apply denoising without enhancement",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1.0,
        help="Denoise strength for enhancement (0.0 to 1.0)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="CFM prior temperature (0.0 to 1.0)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="midpoint",
        choices=["midpoint", "rk4", "euler"],
        help="Numerical solver to use",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=64,
        help="Number of function evaluations",
    )
    parser.add_argument(
        "--parallel_mode",
        action="store_true",
        help="Shuffle the audio paths and skip the existing ones, enabling multiple jobs to run in parallel",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size per GPU for inference (only in parallel_mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling audio paths",
    )

    args, _ = parser.parse_known_args()

    if args.parallel_mode and (args.batch_size is None):
        parser.error('--parallel_mode requires --batch_size')

    args = parser.parse_args()

    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available but --device is set to cuda, using CPU instead")
        device = "cpu"

    world_size = torch.cuda.device_count()

    start_time = time.perf_counter()
    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))

    if args.parallel_mode:
        if args.denoise_only:
            mp.spawn(parallel_denoise,
                     args=(args.in_dir,
                           args.out_dir,
                           args.batch_size,
                           world_size,
                           args.seed,   ),
                     nprocs = world_size)
        else:
            mp.spawn(parallel_enhance,
                     args=(args.in_dir,
                           args.out_dir,
                           args.batch_size,
                           world_size,
                           args.nfe,
                           args.solver,
                           args.lambd,
                           args.tau,
                           args.run_dir,
                           args.seed,   ),
                     nprocs = world_size)
    else:
        if len(paths) == 0:
            print(f"No {args.suffix} files found in the following path: {args.in_dir}")
            return

        pbar = tqdm(paths)

        for path in pbar:
            out_path = args.out_dir / path.relative_to(args.in_dir)
            if out_path.exists():
                pbar.set_description(f"{out_path} Already processed!!!")
                continue
            pbar.set_description(f"Processing {out_path}")
            dwav, sr = torchaudio.load(path)
            dwav = dwav.mean(0)
            if args.denoise_only:
                hwav, sr = denoise(
                    dwav=dwav,
                    sr=sr,
                    device=device,
                    run_dir=args.run_dir,
                )
            else:
                hwav, sr = enhance(
                    dwav=dwav,
                    sr=sr,
                    device=device,
                    nfe=args.nfe,
                    solver=args.solver,
                    lambd=args.lambd,
                    tau=args.tau,
                    run_dir=args.run_dir,
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_path, hwav[None], sr)

    # Cool emoji effect saying the job is done
    elapsed_time = time.perf_counter() - start_time
    print(f"🌟 Enhancement done! {len(paths)} files processed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
