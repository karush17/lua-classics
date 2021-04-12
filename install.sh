#!/bin/bash
# install lua & torch
sudo apt-get install lua
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
bash install-deps
./install.sh
install dependencies
cd ..
for pack in 'image' 'gnuplot' 'paths'
do
    luarocks install $pack
done