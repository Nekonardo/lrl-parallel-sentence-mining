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

# Distillation data from OPUS-Europarl v8
DIST="$ROOT/data/distillation"
mkdir -p "$DIST"

# Sorbian MT data for distillation
mkdir -p "$DIST/MT"
cp -r "$ROOT/third_party/llms-limited-resources2025/Sorbian/hsb/MT/." "$DIST/MT/"
cp -r "$ROOT/third_party/llms-limited-resources2025/Sorbian/dsb/MT/." "$DIST/MT/"

wget -q --show-progress -P "$DIST" https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/cs-de.txt.zip
wget -q --show-progress -P "$DIST" https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/de-pl.txt.zip

unzip -o "$DIST/cs-de.txt.zip" -d "$DIST"
unzip -o "$DIST/de-pl.txt.zip" -d "$DIST"

echo "Done. Files copied to $DEST, distillation data in $DIST"
