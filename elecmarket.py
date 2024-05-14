import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize, LinearConstraint, approx_fprime, BFGS
from scipy.stats import gamma, beta
import pandas as pd
from functools import reduce
import logging
from joblib import Parallel, delayed
from line_profiler import LineProfiler

import gurobipy as gp
from gurobipy import GRB

logging.basicConfig(filename='optimization.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants
# peak hours
pcoef = 65./168
# off-peak hours
opcoef = 103./168
# conversion of hourly revenue per MW into annual revenue per kW
convcoef = 24.*365.25/1000.
# baseline supply
def F0(X):
    return 17.5*X/150.
# Integrated baseline supply function
def G0(X):
    return 17.5*X*X/300.

# price cap
Pmax = 200


class Agent: # Base class for any producer

    def __init__(self,name,cp,ap):
        # name : name of agent
        # cp : common parameters
        # ap : agent parameters
        # described in derived classes
        self.name = name
        self.Nt = cp['Nt']
        self.NX = ap['NX']
        self.dX = 1.*(ap['Xmax']-ap['Xmin'])/(self.NX-1)
        self.dt = 1.*(cp['tmax']-cp['tmin'])/(self.Nt-1)
        self.X = np.linspace(ap['Xmin'],ap['Xmax'],self.NX)
        self.T = np.linspace(cp['tmin'],cp['tmax'],self.Nt)
        self.rho = ap['discount rate']
        self.gamma = ap['depreciation rate']
        self.rCost = ap['running cost']
        self.fCost = ap['fixed cost']
        self.sCost = ap['scrap value']
        self.tmax = cp['tmax']
        self.m_ = np.zeros((self.Nt, self.NX))
        self.mhat_ = np.zeros((self.Nt, self.NX))
        self.mu_ = np.zeros((self.Nt, self.NX))
        self.muhat_ = np.zeros((self.Nt, self.NX))

    def preCalc(self,indens,indenshat,V,V1,V2):
        # Some technical preliminary computations
        self.m_[0,:] = indens[:]                              # Measure flow
        self.mhat_[0,:] = indenshat[:]
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Minimum value
            env.start()
            self.model = gp.Model(env=env)

        self.mhat = [[self.model.addVar() for _ in range(self.NX)] for _ in range(self.Nt-1)]
        self.m = [[self.model.addVar() for _ in range(self.NX)] for _ in range(self.Nt-1)]
        self.muhat = [[self.model.addVar() for _ in range(self.NX)] for _ in range(self.Nt-1)]
        self.mu = [[self.model.addVar() for _ in range(self.NX)] for _ in range(self.Nt-1)]        # Exit measure

        for t in range(0,self.Nt-2):

            # Group 1
            for j in range(1,self.NX-1):
                # Group 1
                expr = gp.LinExpr([V[j],V1[j+1],V2[j-1],self.dt,-self.dt,-1.],[self.m[t+1][j],self.m[t+1][j+1],self.m[t+1][j-1],self.mu[t+1][j],self.muhat[t+1][j],self.m[t][j]])
                self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
                # Group 1hat
                expr = gp.LinExpr([V[j],V1[j+1],V2[j-1],self.dt,-1.],[self.mhat[t+1][j],self.mhat[t+1][j+1],self.mhat[t+1][j-1],self.muhat[t+1][j],self.mhat[t][j]]) # mu_hat n°2 = 0
                self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)

            # Group 2
                # 2A
            expr = gp.LinExpr([V[0],V1[1],self.dt,-self.dt,-1.],[self.m[t+1][0],self.m[t+1][1],self.mu[t+1][0],self.muhat[t+1][0],self.m[t][0]])
            self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
                # 2B
            expr = gp.LinExpr([V[self.NX-1],V2[self.NX-1],self.dt,-self.dt,-1.],[self.m[t+1][self.NX-1],self.m[t+1][self.NX-2],self.mu[t+1][self.NX-1],self.muhat[t+1][self.NX-1],self.m[t][self.NX-1]])
            self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)

            # Group 2 hat
            # 2a hat
            expr = gp.LinExpr([V[0],V1[1],self.dt,-1.],[self.mhat[t+1][0],self.mhat[t+1][1],self.muhat[t+1][0],self.mhat[t][0]])
            self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
            # 2b hat
            expr = gp.LinExpr([V[self.NX-1],V2[self.NX-2],self.dt,-1.],[self.mhat[t+1][self.NX-1],self.mhat[t+1][self.NX-2],self.muhat[t+1][self.NX-1],self.mhat[t][self.NX-1]])
            self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)

        # Group 3  # t = 0, Initialisation sur t pour j différent de 0
        for j in range(1,self.NX-1):
            # 3
            expr = gp.LinExpr([V[j],V1[j+1],V2[j-1],self.dt,-self.dt],[self.m[0][j],self.m[0][j+1],self.m[0][j-1],self.mu[0][j],self.muhat[0][j]])
            self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indens[j])
            # 3 hat
            expr = gp.LinExpr([V[j],V1[j+1],V2[j-1],self.dt],[self.mhat[0][j],self.mhat[0][j+1],self.mhat[0][j-1],self.muhat[0][j]])
            self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indenshat[j])

    # Groupe 4 # t = 0, j=0, Initialisation générale + Terminale sur j
        # 4a
        expr = gp.LinExpr([V[0],V1[1],self.dt,-self.dt],[self.m[0][0],self.m[0][1],self.mu[0][0],self.muhat[0][0]])
        self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indens[0])
        # 4b
        expr = gp.LinExpr([V[self.NX-1],V2[self.NX-2],self.dt,-self.dt],[self.m[0][self.NX-1],self.m[0][self.NX-2],self.mu[0][self.NX-1],self.muhat[0][self.NX-1]])
        self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indens[self.NX-1])

        # 4a hat
        expr = gp.LinExpr([V[0],V1[1],self.dt],[self.mhat[0][0],self.mhat[0][1],self.muhat[0][0]])
        self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indenshat[0])
        # 4b hat
        expr = gp.LinExpr([V[self.NX-1],V2[self.NX-2],self.dt],[self.mhat[0][self.NX-1],self.mhat[0][self.NX-2],self.muhat[0][self.NX-1]])
        self.model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indenshat[self.NX-1])

    def preCalc2(self, indens, indenshat, V, V1, V2, model):
        # Some technical preliminary computations
        self.m_[0, :] = indens[:]  # Measure flow
        self.mhat_[0, :] = indenshat[:]

        self.mhat = [[model.addVar() for _ in range(self.NX)] for _ in range(self.Nt - 1)]
        self.m = [[model.addVar() for _ in range(self.NX)] for _ in range(self.Nt - 1)]
        self.muhat = [[model.addVar() for _ in range(self.NX)] for _ in range(self.Nt - 1)]
        self.mu = [[model.addVar() for _ in range(self.NX)] for _ in range(self.Nt - 1)]  # Exit measure

        for t in range(0, self.Nt - 2):
            for j in range(1, self.NX - 1):
                expr = gp.LinExpr([V[j], V1[j + 1], V2[j - 1], self.dt, -self.dt, -1.],
                                  [self.m[t + 1][j], self.m[t + 1][j + 1], self.m[t + 1][j - 1], self.mu[t + 1][j],
                                   self.muhat[t + 1][j], self.m[t][j]])

                model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
                expr = gp.LinExpr([V[j], V1[j + 1], V2[j - 1], self.dt, -1.],
                                  [self.mhat[t + 1][j], self.mhat[t + 1][j + 1], self.mhat[t + 1][j - 1],
                                   self.muhat[t + 1][j], self.mhat[t][j]])
                model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)

            expr = gp.LinExpr([V[0], V1[1], self.dt, -self.dt, -1.],
                              [self.m[t + 1][0], self.m[t + 1][1], self.mu[t + 1][0], self.muhat[t + 1][0],
                               self.m[t][0]])
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
            expr = gp.LinExpr([V[self.NX - 1], V2[self.NX - 1], self.dt, -self.dt, -1.],
                              [self.m[t + 1][self.NX - 1], self.m[t + 1][self.NX - 2], self.mu[t + 1][self.NX - 1],
                               self.muhat[t + 1][self.NX - 1], self.m[t][self.NX - 1]])
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
            expr = gp.LinExpr([V[0], V1[1], self.dt, -1.],
                              [self.mhat[t + 1][0], self.mhat[t + 1][1], self.muhat[t + 1][0], self.mhat[t][0]])
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)
            expr = gp.LinExpr([V[self.NX - 1], V2[self.NX - 2], self.dt, -1.],
                              [self.mhat[t + 1][self.NX - 1], self.mhat[t + 1][self.NX - 2],
                               self.muhat[t + 1][self.NX - 1], self.mhat[t][self.NX - 1]])
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=0.)

        for j in range(1, self.NX - 1):
            expr = gp.LinExpr([V[j], V1[j + 1], V2[j - 1], self.dt, -self.dt],
                              [self.m[0][j], self.m[0][j + 1], self.m[0][j - 1], self.mu[0][j], self.muhat[0][j]])
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indens[j])
            expr = gp.LinExpr([V[j], V1[j + 1], V2[j - 1], self.dt],
                              [self.mhat[0][j], self.mhat[0][j + 1], self.mhat[0][j - 1], self.muhat[0][j]])
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indenshat[j])

        expr = gp.LinExpr([V[0], V1[1], self.dt, -self.dt],
                          [self.m[0][0], self.m[0][1], self.mu[0][0], self.muhat[0][0]])
        model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indens[0])
        expr = gp.LinExpr([V[self.NX - 1], V2[self.NX - 2], self.dt, -self.dt],
                          [self.m[0][self.NX - 1], self.m[0][self.NX - 2], self.mu[0][self.NX - 1],
                           self.muhat[0][self.NX - 1]])
        model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indens[self.NX - 1])
        expr = gp.LinExpr([V[0], V1[1], self.dt], [self.mhat[0][0], self.mhat[0][1], self.muhat[0][0]])
        model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indenshat[0])
        expr = gp.LinExpr([V[self.NX - 1], V2[self.NX - 2], self.dt],
                          [self.mhat[0][self.NX - 1], self.mhat[0][self.NX - 2], self.muhat[0][self.NX - 1]])
        model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=indenshat[self.NX - 1])

    def calculate_constraint1(self, x):

        # x + offset_agent : m
        # x + offset_agent + offset_mesure : mu
        # x + offset + 2*offset_mesure : mhat
        # x + offset + 3*offset_mesure : mu hat

        return (self.V[j] * x[offset_agent + (t+1)*(self.NX-1) + j] +
                            self.V1[j + 1] * x[offset_agent + (t+1)*(self.NX-1) + j + 1] +
                            self.V2[j - 1] * x[offset_agent + (t+1)*(self.NX-1) + j - 1] +
                            self.dt * x[offset_agent + offset_mesure + (t+1)*(self.NX-1) + j] -
                            self.dt * x[offset_agent + 3*offset_mesure + (t+1)*(self.NX-1) + j] -
                            x[offset_agent + t*(self.NX-1) + j])

    def calculate_constraint1hat(self, x):

       return (self.V[j] * x[offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1+ j] +
                self.V1[j + 1] * x[offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1+ j + 1] +
                self.V2[j - 1] * x[offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1+ j - 1] +
                self.dt * x[offset_agent + 3 * offset_mesure + (t + 1) * self.NX - 1+ j] -
                x[offset_agent + 2 * offset_mesure + t * self.NX - 1+ j])

    def calculate_constraint2a(self, x):  # Special case j = 0

        return (self.V[0] * x[offset_agent + (i + 1) * self.NX - 1+ 0] +
                          self.V1[1] * x[offset_agent + (i + 1) * self.NX - 1+ 1] +
                          self.dt * x[offset_agent + 3 * offset_mesure + (i + 1) * self.NX - 1+ 0] -  x[offset_agent + i * self.NX - 1+ 0])

    def calculate_constraint2ahat(self, x):

        return (self.V[0] * x[offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1 + 0] +  # mhat[t+1][0]
                self.V1[1] * x[offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1 + 1] +  # mhat[t+1][1]
                self.dt * x[offset_agent + 3 * offset_mesure + (t + 1) * self.NX - 1 + 0] -  # muhat[t+1][0]
                x[offset_agent + 2 * offset_mesure + t * self.NX - 1  + 0])  # mhat[t][0]

    def calculate_constraint2b(self, x):  # Special case for j = self.NX-1

        return (self.V[self.NX - 1 ] * x[offset_agent + (i + 1) * self.NX - 1+ self.NX - 1 ] +
                self.V2[self.NX - 1 ] * x[offset_agent + (i + 1) * self.NX - 1+ self.NX - 1- 1] +
                self.dt * x[offset_agent + offset_mesure + (i + 1) * self.NX - 1+ self.NX - 1 ] -
                self.dt * x[offset_agent + 3 * offset_mesure + (i + 1) * self.NX - 1+ self.NX - 1 ] -
                x[offset_agent + i * self.NX - 1+ self.NX - 1 ])

    def calculate_constraint2bhat(self, x):

        return (
                self.V[self.NX - 1 ] * x[
            offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1  + self.NX - 1 ] +  # mhat[t+1][NX-1]
                self.V2[self.NX - 1  - 1] * x[
                    offset_agent + 2 * offset_mesure + (t + 1) * self.NX - 1  + (self.NX - 1  - 1)] +  # mhat[t+1][NX-2]
                self.dt * x[offset_agent + 3 * offset_mesure + (t + 1) * self.NX - 1  + self.NX - 1 ] -  # muhat[t+1][NX-1]
                x[offset_agent + 2 * offset_mesure + t * self.NX - 1  + self.NX - 1 ]  # mhat[t][NX-1]
        )

    def calculate_constraint3(self, x):
        return (self.V[j] * x[offset_agent + j] +  # x1[0][j]
                self.V1[j + 1] * x[offset_agent + j + 1] +  # x1[0][j+1]
                self.V2[j - 1] * x[offset_agent + j - 1] +  # x1[0][j-1]
                self.dt * x[offset_agent + offset_mesure + j] -  # x2[0][j]
                self.dt * x[offset_agent + 3 * offset_mesure + j] -  # x3[0][j]
                indens[j])

    def calculate_constraint3hat(self, x):

        return (self.V[j] * x[offset_agent + 2 * offset_mesure + j] +  # mhat[0][j]
                self.V1[j + 1] * x[offset_agent + 2 * offset_mesure + (j + 1)] +  # mhat[0][j+1]
                self.V2[j - 1] * x[offset_agent + 2 * offset_mesure + (j - 1)] +  # mhat[0][j-1]
                self.dt * x[offset_agent + 3 * offset_mesure + j] -  # muhat[0][j]
                indenshat[j])

    def calculate_constraint4a(self, x):
        return (
                self.V[0] * x[offset_agent + 0] +  # x1[0][0]
                self.V1[1] * x[offset_agent + 1] +  # x1[0][1]
                self.dt * x[offset_agent + offset_mesure + 0] -  # x2[0][0]
                self.dt * x[offset_agent + 3 * offset_mesure + 0] -  # x3[0][0]
                indens[0])

    def calculate_constraint4ahat(self, x):
        return (
                self.V[0] * x[offset_agent + 2 * offset_mesure + 0] +  # mhat[0][0]
                self.V1[1] * x[offset_agent + 2 * offset_mesure + 1] +  # mhat[0][1]
                self.dt * x[offset_agent + 3 * offset_mesure + 0] -  # muhat[0][0]
                indenshat[0])

    def calculate_constraint4b(self, x):
        return (
                self.V[self.NX - 1] * x[offset_agent + (self.NX - 1)] +  # x1[0][self.NX - 1]
                self.V2[self.NX - 2] * x[offset_agent + (self.NX - 2)] +  # x1[0][self.NX - 2]
                self.dt * x[offset_agent + offset_mesure + (self.NX - 1)] -  # x2[0][self.NX - 1]
                self.dt * x[offset_agent + 3 * offset_mesure + (self.NX - 1)] -  # x3[0][self.NX - 1]
                indens[self.NX - 1])

    def calculate_constraint4bhat(self, x):
        return (self.V[self.NX - 1] * x[offset_agent + 2 * offset_mesure + self.NX - 1] +  # mhat[0][self.NX-1]
                self.V2[self.NX - 2] * x[offset_agent + 2 * offset_mesure + (self.NX - 2)] +  # mhat[0][self.NX-2]
                self.dt * x[offset_agent + 3 * offset_mesure + self.NX - 1] -  # muhat[0][self.NX-1]
                indenshat[self.NX - 1])

    def bestResponse(self, peakPr, offpeakPr, cPrice, fPrice, subsidy):
        # best response function
        # peakPr : peak price vector
        # offpeakPr : offpeak price vector
        # fPr : vector of fuel prices
        runGain = gp.LinExpr()
        entryGain = gp.LinExpr()
        exitGain = gp.LinExpr()
        curval = 0

        for t in range(self.Nt-1):
            H = self.dX*self.dt*np.exp(-self.rho*(self.T[t+1]))*(pcoef*self.gain(peakPr[t+1],cPrice[t+1],fPrice[:,t+1],subsidy)+opcoef*self.gain(offpeakPr[t+1],cPrice[t+1], fPrice[:,t+1],subsidy))
            runGain.addTerms(H,self.m[t])
            curval = curval + np.sum(H*self.m_[t+1,:])
            H = -self.fCost*self.dX*self.dt*np.exp(-(self.rho+self.gamma)*(self.T[t+1]))*np.ones(self.NX)
            entryGain.addTerms(H,self.muhat[t])
            curval = curval + np.sum(H*self.muhat_[t+1,:])
            H = self.sCost*self.dX*self.dt*np.exp(-(self.rho+self.gamma)*(self.T[t+1]))*np.ones(self.NX)
            exitGain.addTerms(H,self.mu[t])
            curval = curval + np.sum(H*self.mu_[t+1,:])

        obj = runGain + entryGain + exitGain

        self.model.setObjective(obj, GRB.MAXIMIZE)
        self.model.update()
        self.model.optimize()
        sol_m = [[self.m[t][j].X for j in range(self.NX)] for t in range(self.Nt-1)]
        sol_mhat = [[self.mhat[t][j].X for j in range(self.NX)] for t in range(self.Nt-1)]
        sol_mu = [[self.mu[t][j].X for j in range(self.NX)] for t in range(self.Nt-1)]
        sol_muhat = [[self.muhat[t][j].X for j in range(self.NX)] for t in range(self.Nt-1)]

        ob_func = obj.getValue() - curval

        return ob_func, curval, np.array(sol_m), np.array(sol_mhat), np.array(sol_mu), np.array(sol_muhat)


    def update(self,weight,m,mhat,mu,muhat):
        # density update with given weight
        self.m_[1:,:] = (1.-weight)*self.m_[1:,:]+weight*m
        self.mhat_[1:,:] = (1.-weight)*self.mhat_[1:,:]+weight*mhat
        self.mu_[1:,:] = (1.-weight)*self.mu_[1:,:]+weight*mu
        self.muhat_[1:,:] = (1.-weight)*self.muhat_[1:,:]+weight*muhat

    def capacity(self):
        return np.sum(self.m_,axis=1)*self.dX
    def pot_capacity(self):
        return np.sum(self.mhat_,axis=1)*self.dX
    def exit_measure(self):
        return np.sum(self.mu_,axis=1)*self.dX
    def entry_measure(self):
        return np.sum(self.muhat_,axis=1)*self.dX

class Conventional(Agent):
# Base class for conventional producer

    epsilon = 0.5 # parameter of the supply function
    def __init__(self,name,cp,ap):
        Agent.__init__(self,name,cp,ap)
        self.cTax = ap['emissions']
        self.fuel = ap['fuel']
        self.cFuel = ap['cFuel']
        kappa = ap['mean reversion']
        theta = ap['long term mean']
        stdIn = ap['standard deviation']
        delta = stdIn*np.sqrt(2.*kappa/theta)
        V = 1.+delta*delta*self.X*self.dt/(self.dX*self.dX)
        V1 = -delta*delta*self.X*self.dt/(2*self.dX*self.dX)+kappa*(theta-self.X)*self.dt/(2*self.dX)
        V2 = -delta*delta*self.X*self.dt/(2*self.dX*self.dX)-kappa*(theta-self.X)*self.dt/(2*self.dX)
        alpha = (theta/stdIn)*(theta/stdIn)
        bet = theta/stdIn/stdIn
        indens = ap['initial capacity']*bet*gamma.pdf(bet*self.X,alpha)
        indenshat = ap['potential capacity']*bet*gamma.pdf(bet*self.X,alpha)
        self.indens = indens
        self.V = V
        self.V1 = V1
        self.V2 = V2
        self.indenshat = indenshat
        self.indens = indens
        self.indenshat = indenshat
        self.preCalc(indens,indenshat,V,V1,V2)

    def G(self,x):
        return (self.epsilon/2+(x-self.epsilon))*(x>self.epsilon)+x*x/2/self.epsilon*(x>0)*(x<=self.epsilon)

    def F(self,x):
        return (x-self.epsilon>0)+x/self.epsilon*(x>0)*(x<=self.epsilon)

    def gain(self,p, cp, fp, sub):
        return (convcoef*self.G(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp) - self.rCost)

    def offer(self,p, cp, fp, t):
        return sum(self.F(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp)*self.m_[t,:])*self.dX

    def ioffer(self,p,cp,fp,t):
        return sum(self.G(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp)*self.m_[t,:])*self.dX

    def full_offer(self,price, cPrice, fPrice):
        # agent supply for given price level
        res = np.zeros(self.Nt)
        for t in range(self.Nt):
            res[t] = self.offer(price[t], cPrice[t], fPrice[:,t], t)
        return res


class Renewable(Agent):
    def __init__(self,name,cp,ap):
        Agent.__init__(self,name,cp,ap)
        kappa = ap['mean reversion']
        theta = ap['long term mean']
        stdIn = ap['standard deviation']
        delta = stdIn*np.sqrt(2.*kappa/(theta*(1-theta)-stdIn*stdIn))
        V = 1.+delta*delta*self.X*(1-self.X)*self.dt/(self.dX*self.dX)
        V1 = -delta*delta*self.X*(1.-self.X)*self.dt/(2*self.dX*self.dX)+kappa*(theta-self.X)*self.dt/(2*self.dX)
        V2 = -delta*delta*self.X*(1.-self.X)*self.dt/(2*self.dX*self.dX)-kappa*(theta-self.X)*self.dt/(2*self.dX)

        alpha = theta*(theta*(1.-theta)/stdIn/stdIn-1.)
        bet = (1.-theta)*(theta*(1.-theta)/stdIn/stdIn-1.)
        indens = ap['initial capacity']*beta.pdf(self.X,alpha,bet)
        indenshat = ap['potential capacity']*beta.pdf(self.X,alpha,bet)
        self.indens = indens
        self.V = V
        self.V1 = V1
        self.V2 = V2
        self.indenshat = indenshat
        self.indens = indens
        self.indenshat = indenshat
        self.preCalc(indens,indenshat,V,V1,V2)

    def gain(self,p,cp,fp,sub):
        return convcoef*(p+sub)*self.X - self.rCost

    def offer(self,t):
        return sum(self.X*self.m_[t,:])*self.dX

    def full_offer(self):
        # agent supply for given price level
        res = np.zeros(self.Nt)
        for t in range(self.Nt):
            res[t] = self.offer(t)
        return res


class Simulation:
    # main class for performing the simulation
    def __init__(self, cagents, ragents, cp):
        # agents: list of agent class instances
        # cp : common parameters
        self.cagents = cagents # conventional
        self.ragents = ragents # renewable
        self.Nt = cp['Nt']
        self.T = np.linspace(cp['tmin'],cp['tmax'],self.Nt)
        self.pdemand = np.array(cp["demand"][:self.Nt])*cp["demand ratio"]/(pcoef*cp["demand ratio"]+opcoef)
        self.opdemand = np.array(cp["demand"][:self.Nt])/(pcoef*cp["demand ratio"]+opcoef)
        self.Prp = np.zeros(self.Nt)
        self.Prop = np.zeros(self.Nt)
        self.Nfuels = cp['Nfuels']
        self.carbonTax = np.interp(self.T,cp['carbon tax'][0],cp['carbon tax'][1])
        self.subsidy = np.interp(self.T,cp['res subsidy'][0],cp['res subsidy'][1])
        self.acoef = np.array(cp['Fsupply'][0])
        self.bcoef = np.array(cp['Fsupply'][1])
        self.fPrice = np.zeros((self.Nfuels,self.Nt))
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)  # Minimum value
            env.start()
            self.model = gp.Model(env=env)

    def psibar(self,x):
        return np.sum(self.bcoef*(x-self.acoef)**2/2.)

    def calcPrice(self):
        # compute price for given demand profile
        def opfunc(x,t):
            # x = [pp,pop,p1...pK]
            rdem = reduce(lambda a,b:a+b.offer(t),self.ragents,0)
            res = self.psibar(x[2:])
            for ag in self.cagents:
                res = res + pcoef*ag.ioffer(x[0],self.carbonTax[t],x[2:],t) + opcoef*ag.ioffer(x[1],self.carbonTax[t],x[2:],t)
            return res+pcoef*x[0]*(rdem-self.pdemand[t])+opcoef*x[1]*(rdem-self.opdemand[t])+pcoef*G0(x[0])+opcoef*G0(x[1])

        for j in range(self.Nt):
            x0 = np.zeros(self.Nfuels+2)
            x0[0] = self.Prp[j]
            x0[1] = self.Prop[j]
            x0[2:] = self.fPrice[:,j]
            bds = [(0,Pmax),(0,Pmax)]+[(0,None)]*self.Nfuels
            opres = minimize(lambda x:opfunc(x,j),x0,bounds=bds)
            self.Prp[j] = opres.x[0]
            self.Prop[j] = opres.x[1]
            self.fPrice[:,j] = opres.x[2:]

        return self.Prp, self.Prop, self.fPrice

    def run(self, Niter, tol, power=1., shift=1):
        # run the simulation with maximum Niter iterations
        # the program will stop when the total increase of objective function is less than tol
        # power and shift are coefficients in the weight update formula
        conv = np.zeros(Niter)
        start = time.time()
        for i in range(Niter):
            self.calcPrice()
            weight = np.power(1./(i+shift),power)
            print("Iteration",i)
            message = "Weight: {:.4f}".format(weight)
            obtot = 0

            for a in self.ragents:
                ob, val, m, mhat, mu, muhat = a.bestResponse(self.Prp, self.Prop, self.carbonTax, self.fPrice, self.subsidy)
                a.update(weight, m, mhat, mu, muhat)
                message = message+"; "+a.name+": {:.2f}".format(ob)
                obtot = obtot+ob

            for a in self.cagents:
                ob, val, m, mhat, mu, muhat = a.bestResponse(self.Prp,self.Prop,self.carbonTax,self.fPrice, self.subsidy)
                a.update(weight,m,mhat,mu,muhat)
                message = message+"; "+a.name+": {:.2f}".format(ob)
                obtot = obtot+ob

            message = message + "; Total: {:.2f}".format(obtot)
            conv[i] = obtot
            print(message)
            if(obtot<tol):
                Niter = i
                break
        self.calcPrice()
        end = time.time()

        #### Calculs revenus

        producers_revenues = 0 # Revenus des producteurs renewable + gas + charbon + baseline
        peak_offer = 0  # Offre d'électricité conventionnelle peak
        off_peak_offer = 0  # Offre d'électricité conventionnelle off peak
        res_offer = 0 # Offre de renouvelable

        # fuel_revenues : revenus pour l'offre de combustible

        for agent in self.cagents:
            producers_revenues += self.compute_agent(agent, self.Prp, self.Prop, self.fPrice)
            peak_offer += agent.full_offer(self.Prp, self.carbonTax, self.fPrice)
            off_peak_offer += agent.full_offer(self.Prop, self.carbonTax, self.fPrice)
            fuel_revenues = (pcoef * np.sum(peak_offer) + opcoef * np.sum(off_peak_offer)) * agent.cFuel * self.fPrice[agent.fuel]  # revenus du vendeur de combustible ajoutés
            producers_revenues += np.sum(fuel_revenues)

        for agent in self.ragents:
            producers_revenues += self.compute_agent(agent, self.Prp, self.Prop, self.fPrice)
            res_offer += agent.full_offer()

        producers_revenues += pcoef*np.sum(G0(self.Prp))+opcoef*np.sum(G0(self.Prop)) # Ajout revenus de marché Baseline

        peak_offer += G0(self.Prp) # Ajout offre baseline
        off_peak_offer += G0(self.Prop)

        consumer_surplus_p = (Pmax - self.Prp)*(peak_offer+res_offer)    # Surplus du consommateur : différence entre le price cap et le prix effectivement payé
                                                                                                       # * quantité d'électricité totale
        consumer_surplus_op = (Pmax - self.Prop)*(off_peak_offer+res_offer)

        mf_revenues = pcoef*np.sum(consumer_surplus_p) + opcoef*np.sum(consumer_surplus_op) + producers_revenues

        price_vector = np.concatenate([self.Prp, self.Prop, self.fPrice[0], self.fPrice[1]])

        return conv, end-start, Niter, price_vector, mf_revenues

    def write(self, scenario_name):
        # write simulation output into a file scenario_name
        output = {"time": self.cagents[0].T, "peak price": self.Prp, "offpeak price": self.Prop,
                  "peak demand": self.pdemand, "offpeak demand": self.opdemand}
        for a in self.cagents:
            output[a.name+" capacity"] = a.capacity()
            output[a.name+" potential capacity"] = a.pot_capacity()
            output[a.name+" exit measure"] = a.exit_measure()
            output[a.name+" entry measure"] = a.entry_measure()
            output[a.name+" peak supply"] = a.full_offer(self.Prp, self.carbonTax, self.fPrice)
            output[a.name+" offpeak supply"] = a.full_offer(self.Prop, self.carbonTax, self.fPrice)
        for a in self.ragents:
            output[a.name+" capacity"] = a.capacity()
            output[a.name+" potential capacity"] = a.pot_capacity()
            output[a.name+" exit measure"] = a.exit_measure()
            output[a.name+" entry measure"] = a.entry_measure()
            output[a.name+" peak supply"] = a.full_offer()
            output[a.name+" offpeak supply"] = a.full_offer()
        for f in range(self.Nfuels):
            output["Fuel " + str(f)] = self.fPrice[f,:]

        df = pd.DataFrame.from_dict(output)
        df.to_csv(scenario_name+'.csv')
        return output

    def compute_agent(self, agent, peakPr, offpeakPr, fPrice):

        objective = 0
        agent.preCalc(agent.indens, agent.indenshat, agent.V, agent.V1, agent.V2)

        ob, val, m, mhat, mu, muhat = agent.bestResponse(peakPr, offpeakPr, self.carbonTax, fPrice, self.subsidy)
        agent.update(1, m, mhat, mu, muhat)

        for t in range(len(peakPr) - 1):
            run_gain = agent.dX * agent.dt * np.exp(-agent.rho * agent.T[t + 1]) * (pcoef * agent.gain(peakPr[t + 1], self.carbonTax[t + 1], fPrice[:, t + 1], self.subsidy[t + 1])
                                                                              + opcoef * agent.gain(offpeakPr[t + 1], self.carbonTax[t + 1], fPrice[:, t + 1], self.subsidy[t + 1]))

            entry_cost = -agent.fCost * agent.dX * agent.dt * np.ones(agent.NX) * np.exp(-(agent.rho + agent.gamma) * agent.T[t + 1])

            exit_gain = agent.sCost * agent.dX * agent.dt * np.ones(agent.NX) * np.exp(-(agent.rho + agent.gamma) * agent.T[t + 1])

            objective += np.sum(run_gain * agent.m_[t + 1, :]) + np.sum(entry_cost * agent.muhat_[t + 1, :]) \
                         + np.sum(exit_gain * agent.mu_[t + 1, :])

        return objective

    def plannerProblem(self, peakPr, offpeakPr, fPrice):

        producers_revenues = 0
        peak_offer = 0
        off_peak_offer = 0
        res_offer = 0

        for agent in self.cagents:
            producers_revenues += self.compute_agent(agent, peakPr, offpeakPr, fPrice)
            peak_offer += agent.full_offer(peakPr, self.carbonTax, fPrice)
            off_peak_offer += agent.full_offer(offpeakPr, self.carbonTax, fPrice)
            fuel_revenues = (pcoef*np.sum(peak_offer) + opcoef*np.sum(off_peak_offer)) * agent.cFuel * fPrice[agent.fuel]  # revenus du vendeur de combustible ajoutés
            producers_revenues += np.sum(fuel_revenues)

        for agent in self.ragents:
            producers_revenues += self.compute_agent(agent, peakPr, offpeakPr, fPrice)
            res_offer += agent.full_offer()

        baseline_gain = pcoef * np.sum(G0(peakPr)) + opcoef * np.sum(G0(offpeakPr))
        producers_revenues += baseline_gain

        consumer_surplus_p = (peakPr)*(peak_offer+res_offer)
        consumer_surplus_op = (offpeakPr)*(off_peak_offer+res_offer)

        objective_planner = producers_revenues - pcoef*np.sum(consumer_surplus_p) - opcoef*np.sum(consumer_surplus_op)

        ### Pour paralléliser

        # objectives = Parallel(n_jobs=-2, prefer="threads")(delayed(self.compute_agent)(agent, peakPr, offpeakPr, fPrice) for agent in self.ragents + self.cagents)
        # objective_planner = sum(objectives)

        return objective_planner

    def optimizePrices(self, initial_prices):

        iteration_count = [0]
        def callback(x):
            iteration_count[0] += 1
            print("Iteration {}: Current function value: {:.6f}".format(iteration_count[0], objective(x)))

        def objective(prices):
            peakPr, offpeakPr = prices[:self.Nt], prices[self.Nt:2 * self.Nt]
            fPrice = np.reshape(prices[2 * self.Nt:], (self.Nfuels, self.Nt))
            return self.plannerProblem(peakPr, offpeakPr, fPrice)

        price_cap_constraint = LinearConstraint(np.eye(len(initial_prices)), lb=0, ub=Pmax)

        initial_objective = objective(initial_prices)
        print("Initial function value: {:.0f}".format(initial_objective))

        # result = minimize(objective, initial_prices, method='SLSQP', constraints=price_cap_constraint, callback=callback)
        # optimized_prices = result.x
        result = objective(initial_prices)
        optimized_prices = initial_prices

        optimized_peakPr = optimized_prices[:self.Nt]
        optimized_offpeakPr = optimized_prices[self.Nt:2 * self.Nt]
        optimized_fPrice_flat = optimized_prices[2 * self.Nt:]
        optimized_fPrice = np.reshape(optimized_fPrice_flat, (self.Nfuels, self.Nt))

        for agent in self.ragents + self.cagents:
            agent.preCalc(agent.indens, agent.indenshat, agent.V, agent.V1, agent.V2)
            ob, val, m, mhat, mu, muhat = agent.bestResponse(optimized_peakPr, optimized_offpeakPr, self.carbonTax, optimized_fPrice, self.subsidy)
            agent.update(1, m, mhat, mu, muhat)

        return result, optimized_peakPr, optimized_offpeakPr, optimized_fPrice


class PlannerProblem:
    def __init__(self, cagents, ragents, cp):

        self.cagents = cagents  # conventional
        self.ragents = ragents  # renewable
        self.agents = cagents + ragents
        self.Nt = cp['Nt']
        self.T = np.linspace(cp['tmin'], cp['tmax'], self.Nt)
        self.pdemand = np.array(cp["demand"]) * cp["demand ratio"] / (pcoef * cp["demand ratio"] + opcoef)
        self.opdemand = np.array(cp["demand"]) / (pcoef * cp["demand ratio"] + opcoef)
        self.Prp = np.zeros(self.Nt)
        self.Prop = np.zeros(self.Nt)
        self.Nfuels = cp['Nfuels']
        self.carbonTax = np.interp(self.T, cp['carbon tax'][0], cp['carbon tax'][1])
        self.subsidy = np.interp(self.T,cp['res subsidy'][0],cp['res subsidy'][1])
        self.acoef = np.array(cp['Fsupply'][0])
        self.bcoef = np.array(cp['Fsupply'][1])
        self.fPrice = np.zeros((self.Nfuels, self.Nt))

    def psibar(self,x):
        return np.sum(self.bcoef*(x-self.acoef)**2/2.)

    def calcPrice(self):
        # compute price for given demand profile
        def opfunc(x,t):
            # x = [pp,pop,p1...pK]
            rdem = reduce(lambda a,b:a+b.offer(t),self.ragents,0)
            res = self.psibar(x[2:])
            for ag in self.cagents:
                res = res + pcoef*ag.ioffer(x[0],self.carbonTax[t],x[2:],t) + opcoef*ag.ioffer(x[1], self.carbonTax[t], x[2:], t)
            return res+pcoef*x[0]*(rdem-self.pdemand[t])+opcoef*x[1]*(rdem-self.opdemand[t])+pcoef*G0(x[0])+opcoef*G0(x[1])

        for j in range(self.Nt):
            x0 = np.zeros(self.Nfuels+2)
            x0[0] = self.Prp[j]
            x0[1] = self.Prop[j]
            x0[2:] = self.fPrice[:,j]
            bds = [(0,Pmax),(0,Pmax)]+[(0,None)]*self.Nfuels
            opres = minimize(lambda x:opfunc(x,j),x0,bounds=bds)
            self.Prp[j]=opres.x[0]
            self.Prop[j] = opres.x[1]
            self.fPrice[:,j] = opres.x[2:]

        return self.Prp, self.Prop, self.fPrice

    def objective_function(self, variables):
        peakPr, offpeakPr, fPrice_array = self.calcPrice()

        obj_flow = 0
        offset = 0

        for agent in self.agents:
            num_vars_per_agent = 4 * (self.Nt - 1) * agent.NX # 4 mesures
            agent_vars = variables[offset:offset + num_vars_per_agent]
            m, mhat, mu, muhat = self.unpack_agent_vars(agent_vars, agent)
            offset += num_vars_per_agent

            for t in range(self.Nt - 1):
                peak_gain = agent.gain(peakPr[t + 1], self.carbonTax[t + 1], fPrice_array[:, t + 1], self.subsidy[t+1])
                offpeak_gain = agent.gain(offpeakPr[t + 1], self.carbonTax[t + 1], fPrice_array[:, t + 1], self.subsidy[t+1])
                H_run = agent.dX * agent.dt * np.exp(-agent.rho * (self.T[t + 1])) * (
                            pcoef * peak_gain + opcoef * offpeak_gain)
                H_entry = -agent.fCost * agent.dX * agent.dt * np.exp(
                    -(agent.rho + agent.gamma) * (self.T[t + 1])) * np.ones(agent.NX)
                H_exit = agent.sCost * agent.dX * agent.dt * np.exp(
                    -(agent.rho + agent.gamma) * (self.T[t + 1])) * np.ones(agent.NX)

                obj_flow += np.sum(H_run * m[t]) + np.sum(H_entry * mhat[t]) + np.sum(H_exit * mu[t])

        return -obj_flow

    def pack_agent_vars(self, m, mhat, mu, muhat): # Passage mesure à flatten

        m_flat = m.flatten()
        mhat_flat = mhat.flatten()
        mu_flat = mu.flatten()
        muhat_flat = muhat.flatten()

        variables = np.concatenate([m_flat, mu_flat, mhat_flat, muhat_flat])

        return variables

    def unpack_agent_vars(self, variables, agent):  # Passage flatten à mesure

        total_vars_per_agent = 4 * (self.Nt - 1) * agent.NX
        var_per_type = (self.Nt - 1) * agent.NX

        m_flat = variables[:var_per_type]
        mu_flat = variables[var_per_type:2 * var_per_type]
        mhat_flat = variables[2 * var_per_type:3 * var_per_type]
        muhat_flat = variables[3 * var_per_type:total_vars_per_agent]

        m = m_flat.reshape((agent.Nt - 1, agent.NX))
        mhat = mhat_flat.reshape((agent.Nt - 1, agent.NX))
        mu = mu_flat.reshape((agent.Nt - 1, agent.NX))
        muhat = muhat_flat.reshape((agent.Nt - 1, agent.NX))

        return m, mhat, mu, muhat

    def populate_constraints(self):
        constraints_agents = []
        offset = 0

        for agent in self.agents:
            agent_constraints = []
            offset_agent = offset
            offset_mesure = agent.NX
            offset += agent.Nt * agent.NX

            for t in range(agent.Nt - 2):
                for j in range(1, agent.NX - 1):
                    agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint1})

                    agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint1hat})


            for t in range(agent.Nt - 2):
                # Constraints for each type, passing current t, agent, and offsets
                agent_constraints.append({'type': 'eq', 'fun':  agent.calculate_constraint2a})
                agent_constraints.append({'type': 'eq', 'fun':  agent.calculate_constraint2ahat})
                agent_constraints.append({'type': 'eq', 'fun':  agent.calculate_constraint2b})
                agent_constraints.append({'type': 'eq', 'fun':  agent.calculate_constraint2bhat})

            for j in range(1, agent.NX - 1):
                agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint3})
                agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint3hat})

            # Constraints for edge cases at the end of variable sets
            agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint4a})
            agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint4ahat})
            agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint4b})
            agent_constraints.append({'type': 'eq', 'fun': agent.calculate_constraint4bhat})
            constraints_agents.extend(agent_constraints)

        return constraints_agents

    def run(self):
        total_variable_count = sum(4 * (agent.Nt - 1) * agent.NX for agent in self.agents)
        initial_guess = np.zeros(total_variable_count)

        constraints = self.populate_constraints()

        x0 = initial_guess

        def objective_wrapper(variables):
            return self.objective_function(variables)

        bounds = [(0, None)] * total_variable_count


        result = minimize(
            objective_wrapper,
            x0,
            method='trust-constr',
            constraints=constraints,
            bounds=bounds,
            callback=callback,
            options={'disp': True, 'sparse_jacobian': True, 'verbose': 3, 'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 20000}
        )

        if result.success:
            print("Optimization succeeded.")
        else:
            print("Optimization failed:", result.message)

        return result


#### Optimization tools
def getpar(fname):
    with open(fname) as f:
        data = f.read()
    return eval(data)
