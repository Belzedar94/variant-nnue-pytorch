# nnue_dataset.py — with extra debug prints and Windows-safe DLL loading
import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

# ---------- small debug helpers ----------
def _is_true_env(name: str) -> bool:
    v = os.environ.get(name, "")
    return v not in ("", "0", "false", "False", "no", "No")

DEBUG = _is_true_env("NNUE_DEBUG")

def dprint(msg: str):
    if DEBUG:
        print(f"[dbg] {msg}", flush=True)

# ---------- locate & load the C++ loader DLL ----------
local_dllpath = [n for n in glob.glob('./*training_data_loader.*')
                 if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)

dllpath = os.path.abspath(local_dllpath[0])

# Windows: make sure the DLL directory is on the search path for Python 3.8+
if os.name == "nt":
    try:
        os.add_dll_directory(os.path.dirname(dllpath))
    except Exception as _e:
        dprint(f"os.add_dll_directory failed: {_e}")

dprint(f"Loading training_data_loader from: {dllpath}")
dll = ctypes.cdll.LoadLibrary(dllpath)

# ---------- ctypes structures ----------
class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('psqt_indices', ctypes.POINTER(ctypes.c_int)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        # Shapes come directly from the batch metadata
        try:
            if DEBUG:
                dprint(f"SparseBatch.get_tensors(size={self.size}, max_active={self.max_active_features}, num_inputs={self.num_inputs})")
            white_values = torch.from_numpy(
                np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))
            ).pin_memory().to(device=device, non_blocking=True)
            black_values = torch.from_numpy(
                np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))
            ).pin_memory().to(device=device, non_blocking=True)
            white_indices = torch.from_numpy(
                np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))
            ).pin_memory().to(device=device, non_blocking=True)
            black_indices = torch.from_numpy(
                np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))
            ).pin_memory().to(device=device, non_blocking=True)
            us = torch.from_numpy(
                np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))
            ).pin_memory().to(device=device, non_blocking=True)
            them = 1.0 - us
            outcome = torch.from_numpy(
                np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))
            ).pin_memory().to(device=device, non_blocking=True)
            score = torch.from_numpy(
                np.ctypeslib.as_array(self.score, shape=(self.size, 1))
            ).pin_memory().to(device=device, non_blocking=True)
            psqt_indices = torch.from_numpy(
                np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))
            ).long().pin_memory().to(device=device, non_blocking=True)
            layer_stack_indices = torch.from_numpy(
                np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))
            ).long().pin_memory().to(device=device, non_blocking=True)

            if DEBUG:
                dprint(f"tensors: us={tuple(us.shape)}, w_idx={tuple(white_indices.shape)}, "
                       f"b_idx={tuple(black_indices.shape)}, score={tuple(score.shape)}, "
                       f"psqt={tuple(psqt_indices.shape)}, ls={tuple(layer_stack_indices.shape)}")

            return (us, them, white_indices, white_values, black_indices, black_values,
                    outcome, score, psqt_indices, layer_stack_indices)
        except Exception as e:
            print(f"[dbg] get_tensors failed: {e}", flush=True)
            raise

SparseBatchPtr = ctypes.POINTER(SparseBatch)

# ---------- provider ----------
class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu'):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device

        if DEBUG:
            dprint(f"TrainingDataProvider.create_stream("
                   f"feature_set={feature_set}, workers={num_workers}, "
                   f"file={filename}, batch={batch_size}, cyclic={cyclic}, "
                   f"filtered={filtered}, rand_skip={random_fen_skipping})")

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename,
                                             batch_size, cyclic, filtered, random_fen_skipping)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename,
                                             cyclic, filtered, random_fen_skipping)

        dprint(f"TrainingDataProvider: stream created @ {int(ctypes.cast(self.stream, ctypes.c_void_p).value or 0)}")

    def __iter__(self):
        return self

    def __next__(self):
        dprint("TrainingDataProvider.fetch_next() ...")
        v = self.fetch_next(self.stream)
        if v:
            tensors = v.contents.get_tensors(self.device)
            self.destroy_part(v)
            dprint("TrainingDataProvider.fetch_next() -> batch OK")
            return tensors
        else:
            dprint("TrainingDataProvider.fetch_next() -> StopIteration")
            raise StopIteration

    def __del__(self):
        try:
            dprint("TrainingDataProvider.__del__(): destroying stream")
            self.destroy_stream(self.stream)
        except Exception as _e:
            dprint(f"destroy_stream failed: {_e}")

# ---------- ctypes function prototypes ----------
create_sparse_batch_stream = dll.create_sparse_batch_stream
# IMPORTANT: signature includes random_fen_skipping (int) as the 7th argument (Windows ABI)!
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ctypes.c_bool, ctypes.c_bool, ctypes.c_int
]

destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]

destroy_sparse_batch = dll.destroy_sparse_batch
destroy_sparse_batch.argtypes = [SparseBatchPtr]

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu'):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device)

class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu'):
        super(SparseBatchDataset, self).__init__()
        self.feature_set = feature_set
        self.filename = filename
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device

    def __iter__(self):
        return SparseBatchProvider(
            self.feature_set, self.filename, self.batch_size,
            cyclic=self.cyclic, num_workers=self.num_workers,
            filtered=self.filtered, random_fen_skipping=self.random_fen_skipping,
            device=self.device
        )

class FixedNumBatchesDataset(Dataset):
    def __init__(self, dataset, num_batches):
        super(FixedNumBatchesDataset, self).__init__()
        self.dataset = dataset
        self.iter = iter(self.dataset)
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return next(self.iter)
