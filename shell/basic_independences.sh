sudo apt-get install build-essential cmake libgtest-dev libeigen3-dev

# Pangolin
# include/pangolin/gl/colour.h 中添加 #include <limits>
git clone https://github.com/stevenlovegrove/Pangolin.git -b v0.7


# CLI11
mkdir -p ~/Downloads
cd ~/Downloads
git clone https://github.com/CLIUtils/CLI11.git
cd CLI11/
mkdir build
cd build/
cmake ..
make -j4
sudo make install
cd ../..

# nanoflann
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
mkdir build
cd build/
cmake ..
make -j4
sudo make install

# third-party: cnpy
cd ~/Documents/DeepSDF_lj
git submodule update --init
cd third-party/cnpy
mkdir build && cd build
cmake ..
make -j4
cd ../../..

mkdir build && cd build
cmake ..
make -j4