conda create -n deepsdf python=3.8
conda activate deepsdf

pip install plyfile scikit-image trimesh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

