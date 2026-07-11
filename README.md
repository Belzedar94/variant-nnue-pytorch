# Variant NNUE trainer

This is the chess variant NNUE training code for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish). See the documentation in the [wiki](https://github.com/fairy-stockfish/variant-nnue-pytorch/wiki) for more details on the training process and [join our discord](https://discord.gg/FYUGgmCFB4) to ask questions. This project is derived from the [trainer for standard chess](https://github.com/glinscott/nnue-pytorch) used for official Stockfish.

# Setup

CUDA is recommended for full training speed, but the trainer and DataLoader
also support CPU-only forward/backward runs. You do not need to download the
CUDA Toolkit when using the packaged CUDA dependencies.

#### Package dependencies

At least Python 3.9 needed.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA 11.8 will be installed along with the matching CuPy version.
If your GPU is CUDA 12.8 capable, use `requirements-CUDA128.txt` instead.
Both files pin the NumPy and TorchMetrics compatibility bounds required by
PyTorch Lightning 1.9.5.

For CPU-only use, install a CPU PyTorch wheel and omit CuPy. The portable
PyTorch feature-transformer path is selected automatically.

#### Build the fast DataLoader
This requires a C++17 compiler and cmake.

Windows:
```
compile_data_loader.bat
```

Linux/Mac:
```
sh compile_data_loader.bat
```

# Tests

Build the DataLoader first, then run the focused Python regression suite:

```
ctest --test-dir build -C Release --output-on-failure
python -m pytest
```

CI runs the native and CPU-only Python suites on both Linux and Windows. When
the suite is run on a CUDA machine with matching CuPy support, it additionally
executes the optimized-kernel forward/backward test.

# Train a network

```
source env/bin/activate
python train.py train_data.bin val_data.bin
```

Training and validation inputs must be different files. The escape hatch
`--allow-train-as-validation` exists only for explicit smoke/debug runs and
must not be used for real training or model comparison.

The same seed controls model initialization and native random skipping. Native
batch order is stable across worker counts; validation never applies random
skipping, even when training does, and restarts from the beginning of its
dedicated file for every validation pass.

Smart FEN skipping retains the trainer's historical heuristic: positions with
a teacher capture (including en passant) or a geometric orthodox check are
skipped. It is a target-quality filter, not an Atomic legal-position oracle.

## Resuming training

Resume all Lightning state from a trusted checkpoint:

```
python train.py train_data.bin val_data.bin --resume_from_checkpoint <path> ...
```

Initialize a new training run from model weights or directly from a legacy
Atomic V1 network:

```
python train.py train_data.bin val_data.bin \
  --resume-from-model atomic_run3b_e202_l05.nnue \
  --features="HalfKAv2^"
```

The `.nnue` header identifies the serialized real HalfKAv2 layout. The loader
then adds the 768 zeroed factorizer rows required by `HalfKAv2^`; serializing
immediately back to `.nnue` is byte-exact. See
[`docs/legacy-atomic-v1.md`](docs/legacy-atomic-v1.md) for the frozen contract.

## Training on GPU
```
python train.py train_data.bin val_data.bin --accelerator gpu --devices 1 ...
```
## Feature set selection
By default the trainer uses a factorized HalfKAv2 feature set (named "HalfKAv2^")
If you wish to change the feature set used then you can use the `--features=NAME` option. For the list of available features see `--help`
The default is:
```
python train.py ... --features="HalfKAv2^"
```

# Logging

```
pip install tensorboard
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

