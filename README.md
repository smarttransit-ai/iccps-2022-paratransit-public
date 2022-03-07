# iccps-2022-paratransit-public

Please follow the instructions below.

Setup data:
* data link: https://drive.google.com/file/d/1VgtalQ5nongWwxrrfeoC2_NMLyCvyglq/view?usp=sharing
* Unzip data directory and place at top level: paratransit-mdp/data.

Main simulation executable is bin/sim_proc.jl.

Data preparation is int dataprep/ directory. Summary of data preparation steps:
* dataprep/prepare_trips.ipynb: prepares trips for the simulations in data/CARTA/processed
* dataprep/chains.ipynb: prepares test set and generative demand model data in data/CARTA/processed
* dataprep/travel_time_matrix.ipynb: generates the travel time matrix in data/travel_time_matrix
* dataprep/travel_time_matrix_congestion.ipynb: creates the congested travel time matrix.
* dataprep/paper_figs.ipynb: formats results for latex to be presented in the paper.

paratransit-mdp/ICAPS directory contains the source code for the ICAPS baseline (DRLSA).