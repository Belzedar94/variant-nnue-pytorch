#!/bin/sh
set -eu

git submodule update --init --recursive
git -C external/Atomic-Stockfish fetch --no-tags origin main:refs/remotes/origin/main
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DCMAKE_INSTALL_PREFIX=$(pwd)"
cmake --build build --config RelWithDebInfo --target install
