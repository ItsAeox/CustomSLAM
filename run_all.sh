#!/bin/bash
set -e  # stop immediately if a command fails

# Change to project directory
cd ~/Documents/FinalYearResearch/CustomSLAM/webar-vio

# Rebuild WASM target
echo "Rebuilding WebAssembly build..."
rm -rf build_wasm
mkdir build_wasm
cd build_wasm
emcmake cmake .. -DBUILD_WASM=ON -DCMAKE_BUILD_TYPE=Release
emmake make -j

# Go back to project root
cd ~/Documents/FinalYearResearch/CustomSLAM

# Start local HTTPS server
echo "Starting HTTPS server on port 8000..."
npx http-server -c-1 . -p 8000 -S \
  -C ./192.168.1.14+2.pem \
  -K ./192.168.1.14+2-key.pem \
  -a 0.0.0.0
