# README for bin/

bin/ includes executables for running simulations. The main simulation file is
sim_proc.jl, the parameters can be adjusted in lines 987-995 in sim_proc.jl. 
julia_packages.jl will setup the julia environment with all the julia packages needed
for sim_proc.jl.


Execution commands:
```bash
cd <paratransit-mdp>/bin

### Setup Julia packages ###
julia julia_packages.jl

### Run a simulation ###

# cc-iccps-1, pid: 
nohup julia sim_proc.jl > logs/simulation1.log & 

```