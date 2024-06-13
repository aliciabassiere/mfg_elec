from elecmarket import *
import os as os

np.random.seed(42)

scenario_name = r"C:\Users\Alicia BASSIERE\OneDrive - GENES\Documents\Paper 03 - MFG planner\Results\test"
cp = getpar("common_params.py")
cagents = [Conventional('Coal', cp, getpar('coal.py')),
          Conventional('Gas', cp, getpar('gas.py'))]

ragents = [Renewable('Renewable', cp, getpar('renewable.py'))]
Niter = cp['iterations']
tol = cp['tolerance']

sim = Simulation(cagents,ragents,cp)

start_simu = time.time()

###################### Version MFG ######################

conv, elapsed, Nit, mfg_prices, mfg_revenues = sim.run(Niter, tol, cp['power'], cp['offset'])

print("MFG Revenues: ", mfg_revenues)
print('Elapsed time: ', elapsed/60)

out = sim.write(scenario_name)

try:
    os.mkdir(scenario_name)
except FileExistsError:
    print('Directory already exists')
os.system("cp common_params.py "+scenario_name+"/common_params.py")

# parameter files are copied to output directory; change this if you change agent types
os.system("cp coal.py "+scenario_name+"/coal.py")
os.system("cp gas.py "+scenario_name+"/gas.py")
os.system("cp renewable.py "+scenario_name+"/renewable.py")

os.system("cp "+scenario_name+'.csv '+scenario_name+"/"+scenario_name+".csv")


plt.figure(figsize=(14,5))
plt.subplot(121)
plt.plot(2018+out['time'], out['peak price'], label='peak price')
plt.plot(2018+out['time'], out['offpeak price'], label='offpeak price')
plt.legend()
plt.title('Electricity price')

plt.subplot(122)
# Plotting the capacity for each agent type; modify this if you change agent types
plt.plot(2018+out['time'], out['Coal capacity'], label='Coal capacity')
plt.plot(2018+out['time'], out['Gas capacity'],label='Gas capacity')
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

# Plotting the fuel prices; modify this if you change fuel types
plt.plot(2018+out['time'], out['Fuel 0'], label='Coal price')
plt.plot(2018+out['time'], out['Fuel 1'], label='Gas price')
plt.legend()
plt.title('Fuel price')
#plt.plot(2018+out['time'],np.interp(out['time'],cp["carbon tax"][0],cp["carbon tax"][1]))
plt.savefig(scenario_name+"/"+'demand_fuelprice.pdf',format='pdf')


plt.figure(figsize=(14,5))
plt.subplot(121)

# Plotting the supply for each agent; modify this if you change agent types
plt.bar(2018+out['time'],out['Coal peak supply'],width=0.25,label='Coal supply')
plt.bar(2018+out['time'],out['Gas peak supply'],width=0.25,
        bottom=out['Coal peak supply'],label='Gas supply')
plt.bar(2018+out['time'],out['Renewable peak supply'],width=0.25,
        bottom=out['Gas peak supply']+out['Coal peak supply'],label='Renewable supply')
#plt.ylim([0,80])
plt.title('Conventional/ renewable peak supply, GW')
plt.legend()
plt.subplot(122)
plt.bar(2018+out['time'],out['Coal offpeak supply'],width=0.5,label='Coal supply')
plt.bar(2018+out['time'],out['Gas offpeak supply'],width=0.25,
        bottom=out['Coal offpeak supply'],label='Gas supply')
plt.bar(2018+out['time'],out['Renewable offpeak supply'],width=0.25,
        bottom=out['Gas offpeak supply']+out['Coal offpeak supply'],label='Renewable supply')

plt.title('Conventional/ renewable off-peak supply, GW')

#plt.ylim([0,80])


plt.legend()
plt.savefig(scenario_name+"/"+'supply.pdf',format='pdf')
plt.show()

print('Elapsed time: ', elapsed)
#
# ###################### Version Planner ######################
#
# print("############### Planner Version ###############")
#
# initial_prices = np.random.rand(2 * sim.Nt + sim.Nfuels * sim.Nt)*10
#
# planner_objective, sim.Prp, sim.Prop, sim.fPrice = sim.optimizePrices(mfg_prices)
#
# end_simu = time.time()
#
# print("Time: ", end_simu-start_simu, "s")
#
# out = sim.write(scenario_name)
#
# try:
#     os.mkdir(scenario_name)
# except FileExistsError:
#     print('Directory already exists')
# os.system("cp common_params.py "+scenario_name+"/common_params.py")
#
# # parameter files are copied to output directory; change this if you change agent types
# os.system("cp coal.py "+scenario_name+"/coal.py")
# os.system("cp gas.py "+scenario_name+"/gas.py")
# os.system("cp renewable.py "+scenario_name+"/renewable.py")
#
# os.system("cp "+scenario_name+'.csv '+scenario_name+"/"+scenario_name+".csv")
#
#
# plt.figure(figsize=(14,5))
# plt.subplot(121)
# plt.plot(2018+out['time'], out['peak price'], label='peak price')
# plt.plot(2018+out['time'], out['offpeak price'], label='offpeak price')
# plt.legend()
# plt.title('Electricity price (Planner)')
# plt.subplot(122)
#
# # Plotting the capacity for each agent type; modify this if you change agent types
# plt.plot(2018+out['time'], out['Coal capacity'], label='Coal capacity')
# plt.plot(2018+out['time'], out['Gas capacity'],label='Gas capacity')
# plt.plot(2018+out['time'], out['Renewable capacity'], label='Renewable capacity')
# plt.legend()
# plt.title('Installed capacity (Planner)')
# plt.savefig(scenario_name+"/"+'price_capacity.pdf', format='pdf')
#
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# plt.plot(2018+out['time'], sim.pdemand, label='peak demand')
# plt.plot(2018+out['time'], sim.opdemand, label='offpeak demand')
# plt.legend()
# plt.title('Electricity demand (Planner)')
# plt.subplot(122)
#
# # Plotting the fuel prices; modify this if you change fuel types
# plt.plot(2018+out['time'], out['Fuel 0'], label='Coal price')
# plt.plot(2018+out['time'], out['Fuel 1'], label='Gas price')
# plt.legend()
# plt.title('Fuel price (Planner)')
# #plt.plot(2018+out['time'],np.interp(out['time'],cp["carbon tax"][0],cp["carbon tax"][1]))
# plt.savefig(scenario_name+"/"+'demand_fuelprice.pdf',format='pdf')
#
#
# plt.figure(figsize=(14,5))
# plt.subplot(121)
#
# # Plotting the supply for each agent; modify this if you change agent types
# plt.bar(2018+out['time'],out['Coal peak supply'],width=0.25,label='Coal supply')
# plt.bar(2018+out['time'],out['Gas peak supply'],width=0.25,
#         bottom=out['Coal peak supply'],label='Gas supply')
# plt.bar(2018+out['time'],out['Renewable peak supply'],width=0.25,
#         bottom=out['Gas peak supply']+out['Coal peak supply'],label='Renewable supply')
# #plt.ylim([0,80])
# plt.title('Conventional/ renewable peak supply, GW (Planner)')
# plt.legend()
# plt.subplot(122)
# plt.bar(2018+out['time'],out['Coal offpeak supply'],width=0.5,label='Coal supply')
# plt.bar(2018+out['time'],out['Gas offpeak supply'],width=0.25,
#         bottom=out['Coal offpeak supply'],label='Gas supply')
# plt.bar(2018+out['time'],out['Renewable offpeak supply'],width=0.25,
#         bottom=out['Gas offpeak supply']+out['Coal offpeak supply'],label='Renewable supply')
#
# plt.title('Conventional/ renewable off-peak supply, GW (Planner)')
#
# #plt.ylim([0,80])
#
# plt.legend()
# plt.savefig(scenario_name+"/"+'supply.pdf',format='pdf')
# plt.show()
#
#
