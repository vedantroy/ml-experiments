#!/bin/bash -i
set -euxo pipefail

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda_installer.sh
bash miniconda_installer.sh -b -f -p /home/user/miniconda
echo 'export PATH="/home/user/miniconda/bin:$PATH"' >> ~/.bashrc
conda init bash
source ~/.bashrc
conda create -n imagen python=3.9
conda activate imagen 
pip install -r ./requirements.txt

