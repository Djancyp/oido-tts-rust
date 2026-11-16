#!/usr/bin/env bash
# Builds the vendored llama.cpp shared libraries that oido links against.
#
#   scripts/build-llama.sh          Vulkan + CPU (default)
#   scripts/build-llama.sh --cpu    CPU only
#
# Fetches the pinned submodule if needed and applies patches/llama.cpp/*.patch once.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/llama.cpp
vulkan=ON
[[ ${1:-} == --cpu ]] && vulkan=OFF

if [[ ! -e $src/CMakeLists.txt ]]; then
    git -C "$root" submodule update --init --depth 1 third_party/llama.cpp
fi

for patch in "$root"/patches/llama.cpp/*.patch; do
    if git -C "$src" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch")"
    else
        git -C "$src" apply "$patch"
        echo "applied: $(basename "$patch")"
    fi
done

cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_VULKAN=$vulkan -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build "$src/build" -j "$(nproc)" --target llama-tts
