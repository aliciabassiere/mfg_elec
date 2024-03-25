
from elecmarket import *
import os as os

scenario_name = "test"
cp = getpar("common_params.py")
cagents = [Conventional('Coal',cp,getpar('coal.py')),
          Conventional('Gas',cp,getpar('gas.py'))]

ragents = [Renewable('Renewable',cp,getpar('renewable.py'))]
Niter = cp['iterations']
tol = cp['tolerance']
sim = Simulation(cagents,ragents,cp)

sim.plannerProblem()
