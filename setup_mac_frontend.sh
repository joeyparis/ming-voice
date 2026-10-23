#!/usr/bin/env bash
# Build the upstream speech frontend without Homebrew or system Python changes.
set -euo pipefail
cd "$(dirname "$0")"
python="${1:-.venv/bin/python}"
prefix="$PWD/.native/openfst-1.8.4"
if [ ! -f "$prefix/.built" ]; then
  if ! xcrun --find clang++ >/dev/null 2>&1; then
    printf '%s\n' 'Xcode Command Line Tools are required. Run xcode-select --install, then rerun setup.' >&2
    exit 1
  fi
  mkdir -p .native/build
  archive="$PWD/.native/build/openfst-1.8.4.tar.gz"
  curl -fL https://openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.4.tar.gz -o "$archive"
  printf '%s  %s\n' a8ebbb6f3d92d07e671500587472518cfc87cb79b9a654a5a8abb2d0eb298016 "$archive" | shasum -a 256 -c -
  tar -xzf "$archive" -C .native/build
  (
    cd .native/build/openfst-1.8.4
    CXXFLAGS="-std=c++17 -O2 -g0" ./configure --prefix="$prefix" --enable-grm --disable-static
    make -j "${MING_BUILD_JOBS:-4}"
    make install
  )
  touch "$prefix/.built"
fi
export CPLUS_INCLUDE_PATH="$prefix/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$prefix/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib${LDFLAGS:+ $LDFLAGS}"
# A cached Pynini wheel can retain another checkout's native library paths.
uv pip install --no-cache --python "$python" inflect==7.5.0 pynini==2.1.7 fasttext-wheel==0.9.2
