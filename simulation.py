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
conv, elapsed, Nit = sim.run(Niter, tol, cp['power'], cp['offset'])
print('Elapsed time: ', elapsed)
out = sim.write(scenario_name)
try:
    os.mkdir(scenario_name)
except FileExistsError:
    print('Directory already exists')
os.system("cp common_params.py "+scenario_name+"/common_params.py")
os.system("cp coal_exit.py "+scenario_name+"/coal_exit.py")
os.system("cp gas_exit.py "+scenario_name+"/gas_exit.py")
os.system("cp gas_entry.py "+scenario_name+"/gas_entry.py")
os.system("cp renewable.py "+scenario_name+"/renewable.py")
os.system("cp "+scenario_name+'.csv '+scenario_name+"/"+scenario_name+".csv")

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.plot(2018+out['time'], out['peak price'], label='peak price')
plt.plot(2018+out['time'], out['offpeak price'], label='offpeak price')
plt.legend()
plt.title('Electricity price')
plt.subplot(122)
plt.plot(2018+out['time'], out['Coal exit capacity'], label='Coal capacity')
plt.plot(2018+out['time'], out['Gas exit capacity'] + out['Gas entry capacity'],label='Gas capacity')
plt.plot(2018+out['time'], out['Renewable capacity'], label='Renewable capacity')
plt.legend()
plt.title('Installed capacity')
plt.savefig(scenario_name+"/"+'price_capacity.pdf', format='pdf')

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(2018+out['time'], sim.pdemand, label='peak demand')
plt.plot(2018+out['time'], sim.opdemand, label='offpeak demand')
plt.legend()
plt.title('Electricity demand')
plt.subplot(122)
plt.plot(2018+out['time'], out['Fuel 0'], label='Coal price')
plt.plot(2018+out['time'], out['Fuel 1'], label='Gas price')
plt.legend()
plt.title('Fuel price')
#plt.plot(2018+out['time'],np.interp(out['time'],cp["carbon tax"][0],cp["carbon tax"][1]))
plt.savefig(scenario_name+"/"+'demand_fuelprice.pdf',format='pdf')


plt.figure(figsize=(14,5))
plt.subplot(121)
plt.bar(2018+out['time'],out['Coal exit peak supply'],width=0.25,label='Coal supply')
plt.bar(2018+out['time'],out['Gas entry peak supply']+out['Gas exit peak supply'],width=0.25,
        bottom=out['Coal exit peak supply'],label='Gas supply')
plt.bar(2018+out['time'],out['Renewable peak supply'],width=0.25,
        bottom=out['Gas entry peak supply']+out['Gas exit peak supply']+out['Coal exit peak supply'],label='Renewable supply')
#plt.bar(2018+out['time'],F0(out['peak price']),width=0.25,
#        bottom=out['Gas entry peak supply']+out['Gas exit peak supply']+out['Coal exit peak supply']+out['Renewable peak supply'],label='Baseline supply')

#plt.xticks([2018,2021,2024,2027,2030,2033,2036],[2018,2021,2024,2027,2030,2033,2036])#plt.plot(2018+T[:60],RRop[:60,Niter-1],label='Residual off-peak demand')
#plt.plot(2018+T[:60],opdemand[:60],label='Total off-peak demand')
plt.ylim([0,80])
plt.title('Conventional/ renewable peak supply, GW')
plt.legend()
plt.subplot(122)
#plt.plot(2018+T[:60],RRp[:60,Niter-1],label='Conventional supply')
#plt.plot(2018+T[:60],pdemand[:60],label='Total demand')
plt.bar(2018+out['time'],out['Coal exit offpeak supply'],width=0.5,label='Coal supply')
plt.bar(2018+out['time'],out['Gas entry offpeak supply']+out['Gas exit offpeak supply'],width=0.25,
        bottom=out['Coal exit offpeak supply'],label='Gas supply')
plt.bar(2018+out['time'],out['Renewable offpeak supply'],width=0.25,
        bottom=out['Gas entry offpeak supply']+out['Gas exit offpeak supply']+out['Coal exit offpeak supply'],label='Renewable supply')
#plt.bar(2018+out['time'],F0(out['offpeak price']),width=0.25,
#        bottom=out['Gas entry offpeak supply']+out['Gas exit offpeak supply']+out['Coal exit offpeak supply']+out['Renewable offpeak supply'],label='Baseline supply')

plt.title('Conventional/ renewable off-peak supply, GW')

#plt.xticks([2018,2021,2024,2027,2030,2033,2036],[2018,2021,2024,2027,2030,2033,2036])
plt.ylim([0,80])


plt.legend()
plt.show()
plt.savefig(scenario_name+"/"+'supply.pdf',format='pdf')



print('Elapsed time: ', elapsed)
