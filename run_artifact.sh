#!/bin/bash

echo "Configure platform with Vulkan SDK"
./install_vulkan.sh
echo "Run tests"
./tests.sh
echo "Aggregate results"
./aggregate.sh
echo "Create charts"
pdflatex results.tex
