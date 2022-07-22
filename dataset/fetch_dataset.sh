#!/usr/bin/env bash

echo "Downloading the latest version of Visual-Genome ..."
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -c https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
mkdir -p vg_images

echo "Extracting the images ..."
unzip -q -d vg_images images.zip
unzip -q -d vg_images images2.zip
rm images.zip
rm images2.zip

echo "Downloading image_data.json.zip ..."
wget -c http://visualgenome.org/static/data/dataset/image_data.json.zip

echo "Extracting the file ..."
unzip -q image_data.json.zip
rm image_data.json.zip

echo "Done."
