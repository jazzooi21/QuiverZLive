<p align="center">
  <img src="./quiverzlive/icons/logo.png" alt="logo" width="100"/>
</p>

**`QuiverZLive`** is an interactive tool to assist with calculation/manipulation of quivers in $3d\ \ N=4$ supersymmetric quantum field theories, heavily inspired by [`ZXLive`](https://github.com/zxcalc/zxlive).

Quivers are a central tool in modern mathematical physics, encoding gauge theories as graphs, where nodes represent gauge/flavour groups (circle/square) and edges represent matter fields. They allow the study of rich algebraic and physical phenomena.


## Installation instructions

#### Download from `github`
Download and the `.zip` and unpack directory. 

If dependencies are installed (	`PySide6`, `numpy`, `sympy` and `networkx` --- see `requirements.txt`), the application can be run with `run QuiverZLive.bat`.

If desired, a shortcut (desktop or otherwise) to the `.bat` file can be created for easier access.

#### Installing from source
To install from source, you need `Python (>= 3.9)` and `pip`. If you have those, just run:

    git clone https://github.com/jazzooi21/QuiverZLive.git
    cd QuiverZLive
    pip install .

Then, you can run **QuiverZLive** by typing `qvzl`, `quiverzlive` or `QuiverZLive`.




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

- Gauge/Ungauge Quivers
- Find 3d Mirror
	+ Linear mixed unitary quivers
- Hasse Diagram
	+ Unframed unitary quivers


## File Overview
- `app.py`: Starts main event loop	
- `constants.py`: utility constants

#### Panels (GUI)
- `window_main.py`: Main panel
- `window_calculations.py`: Quiver calculation panel
- `window_hasse.py`: Hasse diagram panel

#### UI
- `Qt_custom_boxes.py`: custom message and warning boxes
- `Qt_zoomable_view.py`: zoomable quiver view (quiver calculation window)

#### Graph Logic
- `graph_model.py`: wrapper for `nx.MultiGraph` with quiver diagram logic
- `quiver_scene.py`: manages quiver editing display (main window)
- `static_scene.py`: manages static quiver display (quiver calculation window)
- `nx_dynkin.py`: Dynkin diagrams encoded in `NetworkX`
- `nx_layouts.py`: Quiver layout logic if automatic layout selected (quiver calculation window)


#### Calculation Logic
- `calc_hasse.py`: Hasse diagram calculation logic
- `calc_HS_C.py`: Hilbert series calculation logic for mixed unitary quivers
- `calc_linearmirror.py`: 3d mirror calculation logic for linear mixed unitary quivers

