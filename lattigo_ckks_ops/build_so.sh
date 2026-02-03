#!/usr/bin/env bash
# build_so.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

SRC="./ckks_ops.go"
OUT="./libLATTIGO_HEVM.so"

echo "[build] script_dir=$script_dir"
echo "[build] src=$script_dir/$(basename "$SRC")"
echo "[build] out=$script_dir/$(basename "$OUT")"


# Build .so in current dir with required name
go build -v -buildmode=c-shared -o "$OUT" "$SRC"

echo "Built: $OUT"
echo "Header: ./libLATTIGO_HEVM.h"
ldd "$OUT" || true
