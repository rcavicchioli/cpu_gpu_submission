#!/bin/bash

echo "Installing vulkan header"
git clone https://github.com/KhronosGroup/Vulkan-LoaderAndValidationLayers.git

sudo cp -r Vulkan-LoaderAndValidationLayers/include/vulkan /usr/include/
sudo apt-get install -y git cmake build-essential libx11-xcb-dev libxkbcommon-dev libmirclient-dev libwayland-dev libxrandr-dev
cd Vulkan-LoaderAndValidationLayers
./update_external_sources.sh
mkdir build
cd build
cmake ..
make
sudo make install
cd ../..
