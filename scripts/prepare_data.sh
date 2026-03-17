#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

DEST="$ROOT/data/benchmark/bucc_style_data"
mkdir -p "$DEST"

# From PaSeMiLL: rename bucc_style_dsb-de -> dsb-de, bucc_style_hsb-de -> hsb-de
cp -r "$ROOT/third_party/PaSeMiLL/data/bucc_style_data/bucc_style_dsb-de" "$DEST/dsb-de"
cp -r "$ROOT/third_party/PaSeMiLL/data/bucc_style_data/bucc_style_hsb-de" "$DEST/hsb-de"

# From Belopsem: chv-ru, oci-es (skip hsb-de)
cp -r "$ROOT/third_party/Belopsem/bucc_style_data/chv-ru" "$DEST/chv-ru"
cp -r "$ROOT/third_party/Belopsem/bucc_style_data/oci-es" "$DEST/oci-es"

echo "Done. Files copied to $DEST"
