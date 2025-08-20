<p align="center">
  <img src="./quiverzlive/icons/logo.png" alt="logo" width="100"/>
</p>

**QuiverZLive** is an interactive tool for working with quivers (in $3d\ \ N=4$).


Heavily inspired by [`zxlive`](https://github.com/zxcalc/zxlive).

## Features

#### Calculations:

- Adjacency Matrix
	+ Unframed unitary quivers
- Global Symmetry Subgroup
	+ Coulomb branch:\
	  Mixed unitary gauge groups
	+ Higgs branch:\
	  Unitary and orthosymplectic quivers
- Hilbert Series
	+ Coulomb branch:\
	  Mixed unitary quivers
	+ Higgs branch:\
	  (not implemented)

#### Quiver Manipulations:

- Gauge/Ungauge Quiver
- Find 3d Mirror
	+ Linear mixed unitary quivers
- Hasse Diagram
	+ Unframed unitary quivers

---
## Installation instructions

#### Download `.zip` folder from `github`
Download the `.zip` and unpack directory. 

If dependencies are installed (`PySide6`, `numpy`, `sympy`, `networkx` -- see `requirements.txt`), the application can be run with `run QuiverZLive.bat`.

If desired, a shortcut (desktop or otherwise) to the `.bat` file can be created for easier access.

#### Installing from source
To install from source, you need `Python (>= 3.9)` and `pip`. If you have those, just run:

    git clone https://github.com/jazzooi21/QuiverZLive.git
    cd QuiverZLive
    pip install .

Then, you can run **QuiverZLive** by typing `qvzl`, `quiverzlive` or `QuiverZLive`.







