#!/usr/bin/env bash
GREEN='\033[1;32m'
NC='\033[0m'
log() { echo -e "${GREEN}$1${NC}"; }

# Install the requirements
log "(1/4) -- Installing the requirements..."
pip install -r requirements.txt

# Downloading the latest FCRN checkpoint and extract it
log "(2/4) -- Downloading the latest FCRN checkpoint..."
(cd checkpoint && sh fetch_checkpoint.sh)

# Downloading the latest version of Visual-Genome
log "(3/4) -- Downloading the dataset..."
(cd dataset && sh fetch_dataset.sh)

# Downloading the Visual-Genome meta data
log "(4/4) -- Downloading Visual-Genome meta data..."
(cd dataset && sh fetch_metadata.sh)

log "Done."
