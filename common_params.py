{
    "Nt" : 100, # Time grid size
    "tmin" : 0,
    "tmax" : 25, # simulation length in years
    "power" : 0.8, # parameters of the weight function in fictitious play
    "offset" : 5, # parameters of the weight function in fictitious play
    "iterations" : 200, # Number of iterations in fictitious play
    "tolerance" : 100, # The algorithm stops if objective improvement is less than tolerance or after the number of iterations is reached
    "carbon tax" : ([0,10,25],[30,60,200]), # Carbon tax is computed by linear interpolation: the first list contains dates and the second one values
    "demand ratio" : 1.257580634655325, # Ratio peak to offpeak demand
    "Nfuels" : 2, # Number of different fuels
    "Fsupply" : ([10, 20], [1.0, 0.5]), # Fuel supply functions; they are linear in this version of the model;
    # the first array contains the intercepts and the second one the coefficients
    "demand" : np.array(pd.read_csv(r"C:\Users\Alicia BASSIERE\OneDrive - GENES\Documents\Paper 02 - Mean Field\Data\Load\forecast_load.csv", index_col=0).T)[1].tolist()
}
