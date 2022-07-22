#!/usr/bin/env bash

echo "Downloading NYU_FCRN checkpoint..."
# The following checkpoint is not available anymore
#wget -c "http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip"

#echo "Extracting the checkpoint ..."
#unzip -q NYU_FCRN-checkpoint.zip
#rm NYU_FCRN-checkpoint.zip

wget -c http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy
echo "Done."
