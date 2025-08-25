import argparse
import os
import sys
import glob
import platform

import torch
import pytorch_lightning as pl
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

import model as M
import features
import nnue_dataset


def make_data_loaders(train_filename, val_filename, feature_set, num_workers, batch_size,
                      filtered, random_fen_skipping, main_device):
    # Epoch and validation sizes are arbitrary
    epoch_size = 20_000_000
    val_size = 1_000_000
    features_name = feature_set.name

    train_infinite = nnue_dataset.SparseBatchDataset(
        features_name, train_filename, batch_size,
        num_workers=num_workers, filtered=filtered,
        random_fen_skipping=random_fen_skipping, device=main_device)

    val_infinite = nnue_dataset.SparseBatchDataset(
        features_name, val_filename, batch_size,
        filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)

    # num_workers has to be 0 for sparse, and 1 for dense
    # it currently cannot work in parallel mode but it shouldn't need to
    train = DataLoader(
        nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size),
        batch_size=None, batch_sampler=None)

    val = DataLoader(
        nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size),
        batch_size=None, batch_sampler=None)

    return train, val


def _mk_checkpoint_callback():
    """
    Make a ModelCheckpoint that works across PL versions.
    PL >= 1.0 uses 'every_n_epochs', older used 'period'.
    """
    try:
        # Preferred on newer PL
        return pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=-1, every_n_epochs=1)
    except TypeError:
        # Fallback for older PL that still understands 'period'
        try:
            return pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=-1, period=1)
        except TypeError:
            # Last resort: no interval arg
            return pl.callbacks.ModelCheckpoint(save_last=True, save_top_k=-1)


def _debug_env():
    if not os.environ.get("NNUE_DEBUG"):
        return

    # DLL discovery (mirrors nnue_dataset logic but only prints here)
    try:
        matches = [n for n in glob.glob('./*training_data_loader.*')
                   if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
        if matches:
            print(f"[dbg] Loading training_data_loader from: {os.path.abspath(matches[0])}", flush=True)
        else:
            print("[dbg] training_data_loader.* not found next to train.py", flush=True)
    except Exception as e:
        print(f"[dbg] Error while globbing training_data_loader.*: {e}", flush=True)

    # Platform / versions
    try:
        os_name = platform.system()
        os_rel = platform.release()
        py_ver = platform.python_version()
        print(f"Platform: {os_name} {os_rel} | Python {py_ver}", flush=True)
    except Exception:
        print(f"Platform: {platform.platform()}", flush=True)

    try:
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}",
              flush=True)
    except Exception as e:
        print(f"PyTorch version/CUDA info error: {e}", flush=True)

    try:
        print(f"PyTorch Lightning: {pl.__version__}", flush=True)
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            idx = 0
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            print(f"GPU[{idx}]: {name} (capability {cap})", flush=True)
        except Exception as e:
            print(f"GPU info error: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument("train", help="Training data (.bin or .binpack)")
    parser.add_argument("val", help="Validation data (.bin or .binpack)")

    # Inherit Trainer CLI flags (gpus, max_epochs, etc.)
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_',
                        help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, "
                             "interpolates between (default=1.0).")
    parser.add_argument("--num-workers", default=1, type=int, dest='num_workers',
                        help="Number of worker threads to use for data loading. Currently only works well for binpack.")
    parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size',
                        help="Number of positions per batch / per iteration. Default on GPU = 16384.")
    parser.add_argument("--threads", default=-1, type=int, dest='threads',
                        help="Number of torch threads to use. Default automatic (cores).")
    parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
    parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping_deprecated',
                        help="Deprecated flag; smart skipping is controlled by --no-smart-fen-skipping.")
    parser.add_argument("--no-smart-fen-skipping", action='store_true', dest='no_smart_fen_skipping',
                        help="Disable smart FEN skipping.")
    parser.add_argument("--random-fen-skipping", default=3, type=int, dest='random_fen_skipping',
                        help="Skip fens randomly on average N before using one.")
    parser.add_argument("--resume-from-model", dest='resume_from_model',
                        help="Initialize training using the weights from the given .pt model")

    # Add feature-set CLI
    features.add_argparse_args(parser)
    args = parser.parse_args()

    # Extra environment + device diagnostics
    _debug_env()

    # Input existence checks (Windows-friendly)
    if not os.path.exists(args.train):
        raise Exception(f'{args.train} does not exist')
    if not os.path.exists(args.val):
        raise Exception(f'{args.val} does not exist')

    feature_set = features.get_feature_set_from_name(args.features)

    if args.resume_from_model is None:
        nnue = M.NNUE(feature_set=feature_set, lambda_=args.lambda_)
        nnue.cuda()
    else:
        nnue = torch.load(args.resume_from_model)
        nnue.set_feature_set(feature_set)
        nnue.lambda_ = args.lambda_
        nnue.cuda()

    print(f"Feature set: {feature_set.name}")
    print(f"Num real features: {feature_set.num_real_features}")
    print(f"Num virtual features: {feature_set.num_virtual_features}")
    print(f"Num features: {feature_set.num_features}")

    print(f"Training with {args.train} validating with {args.val}")

    pl.seed_everything(args.seed)
    print(f"Seed {args.seed}")

    batch_size = args.batch_size if args.batch_size and args.batch_size > 0 else 16384
    print(f'Using batch size {batch_size}')

    print(f'Smart fen skipping: {not args.no_smart_fen_skipping}')
    print(f'Random fen skipping: {args.random_fen_skipping}')

    if args.threads > 0:
        print(f'limiting torch to {args.threads} threads.')
        t_set_num_threads(args.threads)

    logdir = args.default_root_dir if args.default_root_dir else 'logs/'
    print(f'Using log dir {logdir}', flush=True)
    try:
        os.makedirs(logdir, exist_ok=True)
    except Exception as e:
        print(f"[warn] Could not create log dir '{logdir}': {e}", flush=True)

    tb_logger = pl_loggers.TensorBoardLogger(logdir)

    # FIX: use every_n_epochs (or fallback) instead of 'period'
    checkpoint_callback = _mk_checkpoint_callback()

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=tb_logger)

    # Determine main device robustly across PL versions
    if hasattr(trainer, "root_device") and trainer.root_device is not None:
        main_device = trainer.root_device
    elif getattr(trainer, "root_gpu", None) is not None:
        main_device = torch.device(f"cuda:{trainer.root_gpu}")
    else:
        main_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Using c++ data loader')
    train, val = make_data_loaders(
        args.train, args.val, feature_set, args.num_workers, batch_size,
        not args.no_smart_fen_skipping, args.random_fen_skipping, main_device)

    print("Starting trainer.fit ...", flush=True)
    trainer.fit(nnue, train, val)
    print("trainer.fit finished.", flush=True)


if __name__ == '__main__':
    main()
