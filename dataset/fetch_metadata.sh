#!/usr/bin/env bash

echo "Downloading image_data.json.zip ..."
wget -c http://visualgenome.org/static/data/dataset/image_data.json.zip

echo "Extracting the file ..."
unzip -q image_data.json.zip
rm image_data.json.zip

echo "Done."
