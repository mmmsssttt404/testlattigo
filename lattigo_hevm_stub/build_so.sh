#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

SRC="./hevm_stub.go"
OUT="./libLATTIGO_HEVM.so"

echo "[build] script_dir=$script_dir"
echo "[build] src=$script_dir/$(basename "$SRC")"
echo "[build] out=$script_dir/$(basename "$OUT")"

# 不要清代理；只设置 Go 的模块代理
export GOPROXY="https://goproxy.cn,direct"
export GOSUMDB=off
export CGO_ENABLED=1

go version

# 固定 lattigo 版本
go get github.com/tuneinsight/lattigo/v6@v6.1.1
go mod tidy

go build -v -buildmode=c-shared -o "$OUT" "$SRC"

echo "Built: $OUT"
echo "Header: ./libLATTIGO_HEVM.h"
ldd "$OUT" || true
