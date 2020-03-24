import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

class AnkleModel:
    """
    TA muscle model adapted from Benoussaad et al. (2013)
    """
    def __init__(self, HR, Fmax, Emin):
        self.Tact = 0.01 #s (Activation constant time)
        self.Tdeact = 0.04 #s (Relaxation constant time)
        self.J = 0.0197 # kg.m^2 (Inertia of the foot around ankle)
        self.d = 3.7 # cm (moment arm of TA wrt ankle)
        self.tendon_length = 22.3 #cm   Model assumes constant tendon length
        self.resting_length_muscle_tendon = 32.1 #cm
        self.av = 1.33
        self.fv1 = 0.18
        self.fv2 = 0.023
        self.vmax = -0.9 #m/s
        self.Fmax = 600 #N
        self.W = 0.56 #shape parameter of f_fl
        self.a = [2.1, -0.08, -7.97, 0.19, -1.79]
        #optimal_length_CE is the "optimal length of the fiber at which the maximal force can be generated"
        self.optimal_length_CE = 10.0 #where do we find this value???
        self.m = 1.0275 #kg (mass of foot)
        self.COM = 11.45 #cm (center of mass location with respect to the ankle)

    def F_m(self, x, x_ext):
        """
        :param x: state vector
        :param x_ext: external states
        :return: TA muscular force induced by electrical stimulation (Fm)
        """
        return x[0]*self.Fmax*self.f_fl(x, x_ext)*self.f_fv(x, x_ext)

    def f_fl(self, x, x_ext):
        """
       :param x: state vector
       :param x_ext: external states
       :return: force (f_fl) due to TA muscle length
       """
        return np.exp(-pow((self.CE_length(x, x_ext)-self.optimal_length_CE)/(self.W*self.optimal_length_CE), 2))

        
    def f_fv(self, x, x_ext):
        """
       :param x: state vector
       :param x_ext: external states
       :return: force (f_fv) due to muscle contraction speed
       """
        v_ce = self.d*(x_ext[3]-x[2]) # muscle contraction speed
        if v_ce < 0:
            return (1-v_ce/self.vmax)/(1+v_ce/(self.vmax*self.fv1))
        else:
            return (1+self.av*(v_ce/self.fv2))/(1+(v_ce/self.fv2))
        
    def CE_length(self, x, x_ext):
        """
        :param x: state vector
        :param x_ext: external states
        :return: length of contractile element
        """
        #model assumes constant tendon length
        lmt = self.resting_length_muscle_tendon + self.d*(x_ext[2]-x[1])
        return lmt - self.tendon_length

    def T_grav(self, x):
        """
        :param x: state vector
        :return: torque of the foot around the ankle due to gravity
        """
        g = 9.81 #gravity acceleration
        return self.m*self.COM*np.cos(x[1])*g

    def T_acc(self, x, x_ext):
        """
        :param x: state vector
        :param x_ext: external states
        :return: torque due to movement of the ankle
        """
        return self.m*self.COM*(x_ext[0]*np.sin(x[1]-x_ext[1]*np.cos(x[1])))

    def T_ela(self, x):
        """
        :param x: state vector
        :return: passive elastic torque of the foot around the ankle due to passive muscles and tissues
        """
        return np.exp(self.a[0]+self.a[1]*x[1])-np.exp(self.a[2]+self.a[3]*x[1])+self.a[4]

    def get_derivative(self, t, u, x, x_ext):
        """
        :param t: time
        :param u: normalized muscle excitation (0 <= E <=1 )
        :param x: state variables [normalized activation level of TA muscle; foot-ankle angle; rotational velocity of foot]
        :param x_ext: external states
        :return: time derivatives of state variables
        """
        B = 1 #viscosity parameter
        x1_dot = (u-x[0])*(u/self.Tact-(1-u)/self.Tdeact)
        x2_dot = x[2]
        x3_dot = ((1/self.J)*self.F_m(x, x_ext)*self.d) + self.T_grav(x) + self.T_acc(x, x_ext) + self.T_ela(x) + B

        return [x1_dot, x2_dot, x3_dot]