# train.py  — with extra debug prints for Windows and faulthandler
import argparse
import os
import sys
import platform
import time
import traceback

# Enable faulthandler very early (prints Python tracebacks on segfaults/aborts)
try:
    import faulthandler
    faulthandler.enable(all_threads=True)
except Exception as _e:
    print(f"[dbg] faulthandler.enable failed: {_e}", flush=True)

import torch
import pytorch_lightning as pl
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

import model as M
import nnue_dataset
import features

# ---------- small debug helpers ----------
def _is_true_env(name: str) -> bool:
    v = os.environ.get(name, "")
    return v not in ("", "0", "false", "False", "no", "No")

DEBUG = _is_true_env("NNUE_DEBUG")

def dprint(msg: str):
    if DEBUG:
        print(f"[dbg] {msg}", flush=True)

# Make CUDA launches synchronous if asked (helps surface device-side errors)
if _is_true_env("NNUE_DEBUG"):
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

# ---------- DataLoaders ----------
def make_data_loaders(train_filename, val_filename, feature_set, num_workers, batch_size,
                      filtered, random_fen_skipping, main_device):
    # Epoch and validation sizes are arbitrary
    epoch_size = 20_000_000
    val_size = 1_000_000

    features_name = feature_set.name
    dprint(f"make_data_loaders(): features={features_name}, "
           f"num_workers={num_workers}, batch_size={batch_size}, "
           f"filtered={filtered}, random_fen_skipping={random_fen_skipping}, device={main_device}")

    # The C++ loader is iterable; we wrap it with a fixed-size dataset for Lightning
    train_infinite = nnue_dataset.SparseBatchDataset(
        features_name, train_filename, batch_size,
        num_workers=num_workers, filtered=filtered,
        random_fen_skipping=random_fen_skipping, device=main_device
    )
    val_infinite = nnue_dataset.SparseBatchDataset(
        features_name, val_filename, batch_size,
        filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device
    )

    # num_workers has to be 0 for sparse, and 1 for dense; cannot parallelize further here
    train = DataLoader(
        nnue_dataset.FixedNumBatchesDataset(
            train_infinite, (epoch_size + batch_size - 1) // batch_size
        ),
        batch_size=None, batch_sampler=None
    )
    val = DataLoader(
        nnue_dataset.FixedNumBatchesDataset(
            val_infinite, (val_size + batch_size - 1) // batch_size
        ),
        batch_size=None, batch_sampler=None
    )
    dprint("make_data_loaders(): DataLoaders constructed")
    return train, val

# ---------- Lightning callback to log lifecycle & first steps ----------
class DebugPrintCallback(pl.Callback):
    def __init__(self, log_first_n_batches=2):
        super().__init__()
        self.log_first_n_batches = log_first_n_batches

    # Fit lifecycle
    def on_fit_start(self, trainer, pl_module):
        dprint("callback: on_fit_start")

    def on_sanity_check_start(self, trainer, pl_module):
        dprint("callback: on_sanity_check_start")

    def on_sanity_check_end(self, trainer, pl_module):
        dprint("callback: on_sanity_check_end")

    def on_train_start(self, trainer, pl_module):
        dprint("callback: on_train_start")

    def on_validation_start(self, trainer, pl_module):
        dprint("callback: on_validation_start")

    def on_train_end(self, trainer, pl_module):
        dprint("callback: on_train_end")

    def on_validation_end(self, trainer, pl_module):
        dprint("callback: on_validation_end")

    def on_exception(self, trainer, pl_module, exception):
        print(f"[dbg] callback: on_exception: {repr(exception)}", flush=True)
        traceback.print_exc()

    # First few batches (train/val)
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx < self.log_first_n_batches and DEBUG:
            try:
                us, them, w_idx, w_val, b_idx, b_val, outcome, score, psqt_idx, ls_idx = batch
                dprint(f"train_batch_start[{batch_idx}]: sizes: "
                       f"us={tuple(us.shape)}, w_idx={tuple(w_idx.shape)}, b_idx={tuple(b_idx.shape)}, "
                       f"score={tuple(score.shape)}, outcome={tuple(outcome.shape)}, "
                       f"psqt={tuple(psqt_idx.shape)}, ls={tuple(ls_idx.shape)}")
            except Exception as e:
                dprint(f"train_batch_start[{batch_idx}]: shape inspect failed: {e}")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx < self.log_first_n_batches and DEBUG:
            try:
                us, them, w_idx, w_val, b_idx, b_val, outcome, score, psqt_idx, ls_idx = batch
                dprint(f"val_batch_start[{batch_idx}]: sizes: "
                       f"us={tuple(us.shape)}, w_idx={tuple(w_idx.shape)}, b_idx={tuple(b_idx.shape)}, "
                       f"score={tuple(score.shape)}, outcome={tuple(outcome.shape)}, "
                       f"psqt={tuple(psqt_idx.shape)}, ls={tuple(ls_idx.shape)}")
            except Exception as e:
                dprint(f"val_batch_start[{batch_idx}]: shape inspect failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument("train", help="Training data (.bin or .binpack)")
    parser.add_argument("val", help="Validation data (.bin or .binpack)")
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
    parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for binpack.")
    parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
    parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
    parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
    parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping_deprecated', help="(deprecated flag) left for backward compatibility.")
    parser.add_argument("--no-smart-fen-skipping", action='store_true', dest='no_smart_fen_skipping', help="Disable smart fen skipping.")
    parser.add_argument("--random-fen-skipping", default=3, type=int, dest='random_fen_skipping', help="Skip fens randomly on average N before using one.")
    parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initialize training from a .pt model")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    # Early environment + device info (very useful on Windows)
    print(f"Platform: {platform.system()} {platform.release()} | Python {platform.python_version()}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU[0]: {torch.cuda.get_device_name(0)} (capability {torch.cuda.get_device_capability(0)})")
        except Exception as _e:
            dprint(f"get_device_name failed: {_e}")

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

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 16384
    print(f'Using batch size {batch_size}')

    print(f'Smart fen skipping: {not args.no_smart_fen_skipping}')
    print(f'Random fen skipping: {args.random_fen_skipping}')

    if args.threads > 0:
        print(f'limiting torch to {args.threads} threads.')
        t_set_num_threads(args.threads)

    logdir = args.default_root_dir if args.default_root_dir else 'logs/'
    print(f'Using log dir {logdir}', flush=True)

    tb_logger = pl_loggers.TensorBoardLogger(logdir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, period=1, save_top_k=-1)

    # NOTE: We also pass num_sanity_val_steps explicitly so it's visible in logs.
    # (the CLI can still override it)
    debug_cb = DebugPrintCallback()
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, debug_cb], logger=tb_logger, num_sanity_val_steps=2
    )

    main_device = trainer.root_device if trainer.root_gpu is None else 'cuda:' + str(trainer.root_gpu)

    print('Using c++ data loader')
    dprint("about to build DataLoaders...")
    train, val = make_data_loaders(
        args.train, args.val, feature_set, args.num_workers, batch_size,
        not args.no_smart_fen_skipping, args.random_fen_skipping, main_device
    )
    dprint("DataLoaders ready, calling trainer.fit(...)")

    try:
        trainer.fit(nnue, train, val)
        dprint("trainer.fit(...) returned normally")
    except SystemExit as e:
        print(f"[dbg] trainer.fit raised SystemExit: {e}", flush=True)
        raise
    except Exception as e:
        print(f"[dbg] trainer.fit raised Exception: {e}", flush=True)
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
