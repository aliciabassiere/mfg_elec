import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog
import matplotlib
import matplotlib.pyplot as plt
import time
from cvxopt import solvers, matrix, sparse
from scipy.optimize import root_scalar, minimize
from scipy.sparse import bsr_matrix
from scipy.stats import gamma
from scipy.stats import beta
from scipy.linalg import block_diag
import random
import pandas as pd
from functools import reduce

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
Pmax = 150


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
        self.tmax = cp['tmax']
        self.dens = np.zeros(self.Nt*self.NX)


    def preCalc(self,indens,V,V1,V2):
        # Some technical preliminary computations
        A=np.zeros((self.NX,self.NX))
        A.flat[::self.NX+1] = V
        A.flat[1::self.NX+1] = V1
        A.flat[self.NX::self.NX+1] = V2
        self.A_ub = np.zeros(((self.Nt-1)*self.NX,(self.Nt-1)*self.NX))
        for i in range(self.Nt-1):
            self.A_ub[((i)*self.NX):((i+1)*self.NX), ((i)*self.NX):((i+1)*self.NX)] = A
        for i in range(self.Nt-2):
            self.A_ub[((i+1)*self.NX):((i+2)*self.NX), ((i)*self.NX):((i+1)*self.NX)] = -np.diag(np.ones(self.NX))
        self.b_ub=np.zeros((self.Nt-1)*self.NX)
        self.b_ub[:self.NX] = indens
        self.dens[:self.NX]=indens

    def bestResponse(self, peakPr, offpeakPr, cPrice, fPrice):
        # best response function
        # peakPr : peak price vector
        # offpeakPr : offpeak price vector
        # fPr : vector of fuel prices
        # TODO : revert to interior point when the objective function returned by highs-ds is negative
        H=np.zeros(self.Nt*self.NX)
        for i in range(self.Nt):
            H[i*self.NX:(i+1)*self.NX]=(self.dX*self.dt*np.exp(-self.rho*(self.T[i]))*
                                        (pcoef*self.gain(peakPr[i],cPrice[i], fPrice[:,i],i)+
                                         opcoef*self.gain(offpeakPr[i],cPrice[i], fPrice[:,i],i)))
        try:
            res = linprog(-H[self.NX:],bsr_matrix(self.A_ub),self.b_ub,method="highs-ds")
        except:
            print(self.name+'bestResponse: Unexpected error')
        if(res.status):
            print(self.name,res.message,'Reverting to interior-point')
            res = linprog(-H[self.NX:],bsr_matrix(self.A_ub),self.b_ub,method="interior-point",options={"sparse":True,"rr":False})
        br = res.x
        val = np.dot(H[self.NX:],self.dens[self.NX:])
        ob_func = np.dot(H[self.NX:],br)- val
        cons = self.b_ub-np.dot(self.A_ub,br)
        cons1 = np.sum(cons*(cons<0))
        cons2 = np.sum(br*(br<0))
        if(cons1+cons2<-100):
            print(self.name,'Constraint violation, reverting to interior point: ',cons1, cons2)
            res = linprog(-H[self.NX:],bsr_matrix(self.A_ub),self.b_ub,method="interior-point",options={"sparse":True,"rr":False})
            br = res.x
            val = np.dot(H[self.NX:],self.dens[self.NX:])
            ob_func = np.dot(H[self.NX:],br)- val
        return ob_func, val, br

    def update(self,weight,dens1):
        # density update with given weight
        self.dens[self.NX:] = (1.-weight)*self.dens[self.NX:]+weight*dens1




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
        V=1.+delta*delta*self.X*self.dt/(self.dX*self.dX)
        V1 =-delta*delta*self.X[1:]*self.dt/(2*self.dX*self.dX)+kappa*(theta-self.X[1:])*self.dt/(2*self.dX)
        V2=-delta*delta*self.X[:-1]*self.dt/(2*self.dX*self.dX)-kappa*(theta-self.X[:-1])*self.dt/(2*self.dX)
        alpha = (theta/stdIn)*(theta/stdIn)
        bet = theta/stdIn/stdIn
        indens = ap['initial capacity']*bet*gamma.pdf(bet*self.X,alpha)
        self.maxdens = ap['total capacity']*bet*gamma.pdf(bet*self.X,alpha)
        self.preCalc(indens,V,V1,V2)
    def G(self,x):
        # Gain function
        return (self.epsilon/2+(x-self.epsilon))*(x>self.epsilon)+x*x/2/self.epsilon*(x>0)*(x<=self.epsilon)
    def F(self,x):
        # Supply function
        return (x-self.epsilon>0)+x/self.epsilon*(x>0)*(x<=self.epsilon)
    def full_offer(self,price, cPrice, fPrice):
        # agent supply for given price level
        res = np.zeros(self.Nt)
        for i in range(self.Nt):
            res[i] = self.offer(price[i], cPrice[i], fPrice[:,i],i)
        return res

class ConventionalExit(Conventional):
    def __init__(self,name,cp,ap):
        Conventional.__init__(self,name,cp,ap)
    def gain(self,p, cp, fp,t):
        return (convcoef*self.G(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp) - self.rCost -
                (self.rho+self.gamma)*self.fCost*np.exp(-self.gamma*self.T[t]))
    def offer(self,p,cp, fp, t):
        return sum(self.F(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp)*self.dens[t*self.NX:(t+1)*self.NX])*self.dX
    def ioffer(self,p,cp,fp,t):
        return sum(self.G(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp)*self.dens[t*self.NX:(t+1)*self.NX])*self.dX
    def capacity(self):
        return np.sum(np.reshape(self.dens,(self.Nt,self.NX)),axis=1)*self.dX



class ConventionalEntry(Conventional):
    def __init__(self,name,cp,ap):
        Conventional.__init__(self,name,cp,ap)
        self.frate = ap['fixed cost decrease rate']
    def gain(self,p,cp,fp,t):
        return (-convcoef*self.G(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp) + self.rCost +
                (self.rho+self.frate)*self.fCost*np.exp(-self.frate*self.T[t])+
                (self.gamma-self.frate)*self.fCost*np.exp(-(self.rho+self.gamma)*(self.tmax-self.T[t])-self.frate*self.T[t]))
    def offer(self,p,cp,fp,t):
        return sum(self.F(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp)*(self.maxdens-self.dens[t*self.NX:(t+1)*self.NX]))*self.dX
    def ioffer(self,p,cp,fp,t):
        return sum(self.G(p-self.cFuel*fp[self.fuel]-self.X-self.cTax*cp)*(self.maxdens-self.dens[t*self.NX:(t+1)*self.NX]))*self.dX
    def capacity(self):
        return np.sum(self.maxdens)*self.dX - np.sum(np.reshape(self.dens,(self.Nt,self.NX)),axis=1)*self.dX



class Renewable(Agent):
    def __init__(self,name,cp,ap):
        Agent.__init__(self,name,cp,ap)
        self.frate = ap['fixed cost decrease rate']
        kappa = ap['mean reversion']
        theta = ap['long term mean']
        stdIn = ap['standard deviation']
        delta = stdIn*np.sqrt(2.*kappa/(theta*(1-theta)-stdIn*stdIn))
        V=1.+delta*delta*self.X*(1-self.X)*self.dt/(self.dX*self.dX)
        V1 = -delta*delta*self.X[1:]*(1.-self.X[1:])*self.dt/(2*self.dX*self.dX)+kappa*(theta-self.X[1:])*self.dt/(2*self.dX)
        V2 = -delta*delta*self.X[:-1]*(1.-self.X[:-1])*self.dt/(2*self.dX*self.dX)-kappa*(theta-self.X[:-1])*self.dt/(2*self.dX)
        alpha = theta*(theta*(1.-theta)/stdIn/stdIn-1.)
        bet = (1.-theta)*(theta*(1.-theta)/stdIn/stdIn-1.)
        indens = ap['initial capacity']*beta.pdf(self.X,alpha,bet)
        self.maxdens = ap['total capacity']*beta.pdf(self.X,alpha,bet)
        self.preCalc(indens,V,V1,V2)
    def gain(self,p,cp,fp,t):
        return (-convcoef*p*self.X + self.rCost + self.rho*self.fCost +
                (self.rho+self.frate)*self.fCost*np.exp(-self.frate*self.T[t])+
                (self.gamma-self.frate)*self.fCost*np.exp(-(self.rho+self.gamma)*(self.tmax-self.T[t])-self.frate*self.T[t]))
    def offer(self,t):
        return sum(self.X*(self.maxdens-self.dens[t*self.NX:(t+1)*self.NX]))*self.dX
    def capacity(self):
        return np.sum(self.maxdens)*self.dX - np.sum(np.reshape(self.dens,(self.Nt,self.NX)),axis=1)*self.dX
    def full_offer(self):
        # agent supply for given price level
        res = np.zeros(self.Nt)
        for i in range(self.Nt):
            res[i] = self.offer(i)
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
        self.pdemand = np.array(cp["demand"])*cp["demand ratio"]/(pcoef*cp["demand ratio"]+opcoef)
        self.opdemand = np.array(cp["demand"])/(pcoef*cp["demand ratio"]+opcoef)
        self.Prp = np.zeros(self.Nt)
        self.Prop = np.zeros(self.Nt)
        self.Nfuels = cp['Nfuels']
        self.fTax = np.interp(self.T,cp['carbon tax'][0],cp['carbon tax'][1])
        self.acoef = np.array(cp['Fsupply'][0])
        self.bcoef = np.array(cp['Fsupply'][1])
        self.fPrice = np.zeros((self.Nfuels,self.Nt))

    def psibar(self,x):
        return np.sum(self.bcoef*(x-self.acoef)**2/2.)

    def calcPrice(self):
        # compute price for given demand profile
        def opfunc(x,t):
            # x = [pp,pop,p1...pK]
            rdem = reduce(lambda a,b:a+b.offer(t),self.ragents,0)
            res = self.psibar(x[2:])
            for ag in self.cagents:
                res = res + pcoef*ag.ioffer(x[0],self.fTax[t],x[2:],t) + opcoef*ag.ioffer(x[1],self.fTax[t],x[2:],t)
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

    def run(self,Niter,tol,power=1., shift=1):
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
                ob, val, dens1 = a.bestResponse(self.Prp,self.Prop,self.fTax,self.fPrice)
                a.update(weight,dens1)
                message = message+"; "+a.name+": {:.2f}".format(ob)
                obtot = obtot+ob
            for a in self.cagents:
                ob, val, dens1 = a.bestResponse(self.Prp,self.Prop,self.fTax,self.fPrice)
                a.update(weight,dens1)
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
        return conv, end-start, Niter

    def write(self, scenario_name):
        # write simulation output into a file scenario_name
        output = {"time": self.cagents[0].T, "peak price":self.Prp,"offpeak price":self.Prop,
                  "peak demand":self.pdemand, "offpeak demand":self.opdemand}
        for a in self.cagents:
            output[a.name+" capacity"] = a.capacity()
            output[a.name+" peak supply"] = a.full_offer(self.Prp,self.fTax, self.fPrice)
            output[a.name+" offpeak supply"] = a.full_offer(self.Prop,self.fTax,self.fPrice)
        for a in self.ragents:
            output[a.name+" capacity"] = a.capacity()
            output[a.name+" peak supply"] = a.full_offer()
            output[a.name+" offpeak supply"] = a.full_offer()
        for i in range(self.Nfuels):
            output["Fuel "+str(i)] = self.fPrice[i,:]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(scenario_name+'.csv')
        return output


def getpar(fname):
    # service function: extract parameter dictionary from file
    with open(fname) as f:
        data = f.read()
    return eval(data)
