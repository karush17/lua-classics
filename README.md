# Lua Classics

This repository is a collection of classical Deep Learning models implemented in Lua using Torch. Implementations of algorithms utilize contemporary gradient-based methods as proposed by seminal works.

# Installation

Install Lua, Torch & Dependencies
```
./install.sh
```

# Algorithms

Following are the two classic algorithms implemented-
|Alogrithm|Learning Rule|Location|
|:-------:|:-----------:|:------:|
|Hopfield Network|Hebb Rule|[`hopfield_network.lua`](hopfield_network.lua)|
|Boltzmann Machine|Contrastive Divergence|[`boltzmann_machine.lua`](boltzmann_machine.lua)|

# Usage
To run a Hopfield Network with its default parameters on the `mnist` dataset, use the following-
```
lua main.lua -algo HopfieldNetwork -data mnist
```

# Cite
If you find this repository helpful for your project then please cite the following-
```
@misc{karush17luaclassics,
  author = {Karush Suri},
  title = {Lua Classics},
  year = {2021},
  howpublished = {\url{https://github.com/karush17/lua-classics}}
}
```
