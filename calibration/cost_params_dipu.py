import pandas as pd
import numpy as np
from Simulation_Parameters import SimulationParameters

simu_parameters = SimulationParameters()

class CostParameters:
    def __init__(self):

        self.aPV = 15000 # REF2020 EU
        self.bPV = 0 # REF2020 EU
        #self.aPV = 0
        #self.bPV = 0

        self.aR = 21000 # REF2020 EU
        self.bR = 0.25 # REF2020 EU
        #self.aR = 0
        #self.bR = 0

        self.aG = 20000 # REF2020 EU
        self.bG = 2.31 # REF2020 EU
        #self.aG = 0
        #self.bG = 0

        self.aC = 32500 # REF2020 EU
        self.bC = 3 # REF2020 EU

        self.aPeak = 6960
        self.bPeak = 18   # Fraunhoffer

        # Coal price: IEA July 2022 price for German Lignite + IEA NZE 2030 Target for UE

        #self.pc = np.concatenate([np.linspace(56, 52, 2030-2022), np.linspace(52, 42, 2050-2030)])
        self.pc = np.array(pd.read_csv(simu_parameters.path_inputs + r"\coal_price_inputs.csv", index_col=0)).mean(axis=1).T
        self.conv_factor_c = 0.456 # Necessary ton of coal for 1MWh of electricity in US from https://www.eia.gov/tools/faqs/faq.php?id=667&t=6: 1.14 pounds/kWh, 1,120 pounds/MWh, 0.456 tons/MWh

        # Gas price: DUTCH TTF

        self.pG = np.array(pd.read_csv(simu_parameters.path_inputs + r"\gas_price_inputs.csv", index_col=0).T)
        #self.pG = np.repeat(0.000000000000000000000000000001, simu_parameters.H*(simu_parameters.N_D+1)).reshape(simu_parameters.H, simu_parameters.N_D+1).T
        self.pG_mean = self.pG.mean(axis=0)
        self.pG_low = np.quantile(self.pG, 0.25, axis=0)
        self.pG_high = np.quantile(self.pG, 0.75, axis=0)
        self.pG_min = self.pG.min(axis=0)
        self.pG_max = self.pG.max(axis=0)
        self.pG_multiplier = np.array([self.pG_low/self.pG_mean, self.pG_high/self.pG_mean]).mean(axis=1)
        self.pG_low_shock = self.pG[5] # 2020
        self.pG_high_shock = self.pG[7] # 2022
        self.pG = np.vstack([self.pG[:simu_parameters.N_D-2], self.pG_low_shock, self.pG_high_shock])
        self.pG_mean = self.pG.mean(axis=0)
        self.pG_naive = self.pG[:simu_parameters.N_D-2]
        self.pG_mean_naive = self.pG_naive.mean(axis=0)
        self.pG_mimic = np.vstack([self.pG, self.pG_low_shock, self.pG_low_shock, self.pG_low_shock, self.pG_high_shock, self.pG_high_shock, self.pG_high_shock])
        self.pG_mean_mimic = self.pG_mimic.mean(axis=0)

        self.conv_factor_g = 0.44

        # Oil price: conv factor from EIA and conversion

        self.p_peak = 94*0.0019047619*1000

        # Fossil price evolution factor

        self.fossil_evol = np.repeat(np.linspace(1.00, 1 + ((0*simu_parameters.T)/100), simu_parameters.T+1), simu_parameters.H).reshape(simu_parameters.T+1, simu_parameters.H)

        # Carbon intensities
        self.cintensity_g = 0.33  # Strauss 2009 for Combined Cycle New
        self.cintensity_c = 0.96  # Carbon intensity for Super-critical EPA 2010
        self.cintensity_peak = 0.55  # Strauss 2009 for Combustion Turbine

        #self.voll = 7621.06  # Euros/MWh (Calcul excel)
        self.voll = 12240 # REF ACER
        self.penalty_default = 2490 # Borne basse VOLL sectorielle
        #self.penalty_default = 1000


