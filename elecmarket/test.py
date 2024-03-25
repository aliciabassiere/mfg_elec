# The code for running the simulation
# Parameters are described in appropriate python files
# look into the files for parameter descriptions

from elecmarket import *
import os as os

scenario_name = "test"
cp = getpar("common_params.py")
cagents = [ConventionalExit('Coal exit',cp,getpar('coal_exit.py')),
          ConventionalExit('Gas exit',cp,getpar('gas_exit.py')),
          ConventionalEntry('Gas entry',cp,getpar('gas_entry.py'))]

ragents = [Renewable('Renewable',cp,getpar('renewable.py'))]
Niter = cp['iterations']
tol = cp['tolerance']
sim = Simulation(cagents,ragents,cp)

sim.plannerProblem()