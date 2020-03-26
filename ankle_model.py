import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp,quad
from scipy.interpolate import interp1d
from scipy import signal
from sympy import integrate,Symbol
#from FESsignal import trapezoid_wave

class AnkleModel:
    """
    TA muscle model adapted from Benoussaad et al. (2013)
    """
    def __init__(self):
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
        self.optimal_length_CE = 7.5 #cm - source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3130447/
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

    def get_derivative(self, t, x, x_ext_1,x_ext_2,x_ext_3,x_ext_4, u):
        """
        :param t: time
        :param u: normalized muscle excitation (0 <= E <=1 )
        :param x: state variables [normalized activation level of TA muscle; foot-ankle angle; rotational velocity of foot]
        :param x_ext: external states
        :return: time derivatives of state variables
        """
        u_val = u(t)
        x_ext = [x_ext_1(t),x_ext_2(t),x_ext_3(t),x_ext_4(t)]

        if x[0]>1:
            x[0] =1
        B = 1 #viscosity parameter
        x1_dot = (u_val-x[0])*(u_val/self.Tact-(1-u_val)/self.Tdeact)
        x2_dot = x[2]
        x3_dot = ((1/self.J)*self.F_m(x, x_ext)*self.d) + self.T_grav(x) + self.T_acc(x, x_ext) + self.T_ela(x) + B

        return [x1_dot, x2_dot, x3_dot]

def set_x_ext():
    """
   :param x_ext: external states (linear horizontal acc, linear vertical acc, absolute orientation of shank, absolute rotational velocity)
   :return: x_ext: external states returned
   """
    x_ext_1 = np.array([
        [62.69100630769873, -0.33311750699695075],
        [63.335143489702986, -0.2933497639834579],
        [64.60837963156355, -0.2614353147583439],
        [64.93170140774467, -0.23758302351810867],
        [65.73373992230252, -0.19779439408496602],
        [66.37537073394877, -0.16596349053845172],
        [67.01950791595304, -0.12619574752495932],
        [67.6661514683153, -0.07849116504448839],
        [68.6260913154267, -0.03868164919169592],
        [69.58352479218013, -0.006808972805881108],
        [70.38806967709596, 0.04091649609423964],
        [70.87681189690463, 0.08860019215506076],
        [71.36054137599733, 0.12041020928192481],
        [72.0021721876436, 0.1522411128284391],
        [72.80421070220142, 0.19202974226158176],
        [73.76916329002881, 0.24771293704833086],
        [74.25289276912152, 0.27952295417519535],
        [75.21032624587494, 0.3113956305610093],
        [75.8469443168052, 0.32735285517356605],
        [76.80187142320062, 0.35128869209240143],
        [77.4384894941309, 0.3672459167049582],
        [78.23802163833075, 0.39909770667112276],
        [79.19545511508416, 0.43097038305693647],
        [80.1503822214796, 0.45490621997577185],
        [81.10530932787502, 0.478842056894607],
        [81.74694013952127, 0.5106729604411213],
        [82.86227494882826, 0.5425665232465855],
        [84.45632649651195, 0.5903964242449564],
        [85.41125360290738, 0.6143322611637916],
        [86.53160115293035, 0.6620995029032126],
        [87.48903462968377, 0.6939721792890265],
        [88.29357951459959, 0.7416976481891477],
        [89.09060528844145, 0.7656125986883331],
        [89.72972972972973, 0.7895066627678686],
        [90.69217594719913, 0.8372530180876396],
        [91.80751075650612, 0.8691465808931036],
        [92.9228455658131, 0.9010401436985673],
        [93.88027904256651, 0.9329128200843815],
        [94.99060111115752, 0.9489327039558884],
        [96.41171310413968, 0.9491206817327378],
        [97.5145160616567, 0.9413300472033088],
        [98.61230627845774, 0.9176657337399226],
        [99.39429383015163, 0.8939596474372367],
        [100.16876227077154, 0.8464430427336149]
    ])
    x = 0.40625 * ((x_ext_1[:, 0] - 62.69100630769873) / (100 - 62.69100630769873))
    y = x_ext_1[:, 1]
    f1 = interp1d(x, y, fill_value="extrapolate")

    x_ext_2 = np.array([
        [62.029853630259396, -0.7086614173228343],
        [62.386358146949405, -0.6771653543307082],
        [62.91241968987005, -0.6062992125984248],
        [62.92691174339403, -0.6456692913385823],
        [63.62832713395487, -0.5511811023622043],
        [64.17177914110428, -0.5275590551181097],
        [65.27607361963189, -0.5275590551181097],
        [65.6499686005507, -0.5433070866141727],
        [66.77165354330708, -0.5905511811023617],
        [67.14554852422587, -0.6062992125984248],
        [67.34119124679967, -0.6377952755905508],
        [67.91072895029224, -0.6850393700787397],
        [68.47736824308002, -0.7244094488188972],
        [68.67301096565383, -0.7559055118110233],
        [69.42949615960582, -0.8110236220472438],
        [69.9932370416888, -0.8425196850393698],
        [70.55987633447658, -0.8818897637795272],
        [70.55987633447658, -0.8818897637795272],
        [71.49461378677357, -0.9212598425196847],
        [72.60470508671077, -0.9370078740157477],
        [73.34090140572918, -0.9370078740157477],
        [74.63504178542098, -0.9527559055118107],
        [75.56398241630836, -0.9763779527559051],
        [76.49292304719577, -0.9999999999999998],
        [77.24071300903337, -1.0314960629921257],
        [78.16385681851118, -1.0393700787401572],
        [79.45509878749817, -1.0472440944881887],
        [81.10574368388, -1.0314960629921257],
        [82.7621854016714, -1.0314960629921257],
        [83.4983817206898, -1.0314960629921257],
        [84.22878121829862, -1.0157480314960627],
        [85.13743297425243, -0.9842519685039368],
        [85.86783247186125, -0.9685039370078736],
        [86.96343171827446, -0.9448818897637792],
        [87.88077870634268, -0.9370078740157477],
        [89.34447611226508, -0.9133858267716533],
        [90.43717694797351, -0.8818897637795272],
        [91.1733732669919, -0.8818897637795272],
        [92.27476933481474, -0.8740157480314957],
        [93.56311289309694, -0.8740157480314957],
        [94.85145645137915, -0.8740157480314957],
        [95.59634800251195, -0.8976377952755903],
        [96.70933771315394, -0.9212598425196847],
        [97.46002608569633, -0.9606299212598421],
        [98.20781604753394, -0.9921259842519683],
        [98.76865851891212, -1.0157480314960627]
    ])
    x = 0.40625 * ((x_ext_2[:, 0]-62.029853630259396) / (100-62.029853630259396))
    y = x_ext_2[:, 1]
    f2 = interp1d(x, y, fill_value="extrapolate")

    # CHANGE THESE VALUES
    x_ext_3 = np.array([
        [2.8359375, -11.587030716723547],
        [2.8515625, -11.416382252559725],
        [2.8671875, -11.16040955631399],
        [2.875, -10.989761092150168],
        [2.8828125, -10.392491467576786],
        [2.890625, -10.051194539249146],
        [2.8984375, -9.624573378839589],
        [2.90625, -9.197952218430032],
        [2.9140625, -8.941979522184297],
        [2.921875, -8.51535836177474],
        [2.9296875, -8.088737201365184],
        [2.9375, -7.406143344709896],
        [2.9453125, -6.723549488054605],
        [2.953125, -5.69965870307167],
        [2.9609374999999996, -5.187713310580204],
        [2.953125, -5.443686006825935],
        [2.9687500000000004, -4.846416382252556],
        [2.9687500000000004, -4.590443686006825],
        [2.9687500000000004, -4.249146757679181],
        [2.9765625, -3.993174061433443],
        [2.9921875, -2.9692832764505113],
        [3, -2.030716723549485],
        [3.0078125, -1.4334470989761066],
        [3.015625, -0.7508532423208152],
        [3.0234375, 0.017064846416385393],
        [3.03125, 0.784982935153586],
        [3.0390625, 1.4675767918088756],
        [3.046875, 2.150170648464165],
        [3.0546875, 3.0034129692832767],
        [3.0625, 3.515358361774746],
        [3.0546875, 3.3447098976109224],
        [3.0625, 3.771331058020479],
        [3.0703125, 4.539249146757681],
        [3.0781249999999996, 5.221843003412971],
        [3.0859375000000004, 5.477815699658704],
        [3.09375, 6.245733788395906],
        [3.1015625, 6.501706484641639],
        [3.109375, 7.184300341296931],
        [3.1171875, 7.440273037542664],
        [3.125, 8.122866894197955],
        [3.1328125, 8.378839590443686],
        [3.1484375, 8.976109215017065],
        [3.15625, 9.31740614334471],
        [3.1640625, 9.573378839590445],
        [3.1796875, 9.744027303754267],
        [3.1953125, 9.829351535836178],
        [3.2109375000000004, 9.658703071672356],
        [3.2265625, 9.488054607508534],
        [3.2421875, 9.31740614334471]
    ])
    x = x_ext_3[:, 0] - 0.40625
    y = x_ext_3[:, 1]
    f3 = interp1d(x, y, fill_value="extrapolate")

    x_ext_4 = np.array([
        [62.456408020924144, 14.712292938099438],
        [63.56931124673059, 31.00261551874462],
        [64.52266782911944, 45.662598081953035],
        [65.640802092415, 55.448997384481345],
        [66.91586748038362, 70.11769834350486],
        [68.51002615518746, 88.04707933740201],
        [70.11072362685266, 97.84655623365305],
        [71.7101133391456, 109.27201394943341],
        [73.14864864864865, 120.69311246730612],
        [74.58979947689626, 128.86224934612042],
        [76.03356582388841, 133.77942458587628],
        [77.64210985178727, 133.82301656495213],
        [79.56843940714909, 138.75326939843077],
        [80.85658238884047, 137.16216216216228],
        [82.46643417611159, 135.5797733217089],
        [84.08020924149957, 129.11944202266793],
        [85.85745422842197, 119.41150828247609],
        [87.31299040976461, 109.69485614646916],
        [89.09285091543157, 96.73496076721892],
        [90.39014821272885, 83.76198779424595],
        [91.2048823016565, 70.77593722755017],
        [92.34001743679163, 59.424585876198876],
        [93.47776809067133, 44.821272885789085],
        [94.4507410636443, 35.091543156059345],
        [95.75065387968614, 18.86660854402794],
        [97.21142109851787, 2.64603312990414],
        [98.66434176111596, -3.818657367044409]
    ])
    x = 0.40625 * ((x_ext_4[:, 0]-62.456408020924144) / (100-62.456408020924144))
    y = x_ext_4[:, 1]
    f4 = interp1d(x, y, fill_value="extrapolate")

    return f1, f2, f3, f4

def input(t):
    # return 0.3
    # return np.sin(t) / 2 + 0.5
    return np.sin(3*t)/3+0.6

def trapezoid_wave(t, width=0.40625, slope=2, amp=1):
    a = slope*width*signal.sawtooth((2*np.pi*(t))/width, width=0)/4.
    #a = slope * width * signal.sawtooth((2 * np.pi * (t + 0.18)) / width, width=0.5) / 4.
    if a>amp/2.:
        a = amp/2.
    elif a<-amp/2.:
        a = -amp/2.
    return a + amp/2.
def toe_clearance(states,times):
    t = Symbol('t')
    x_ext_integral = integrate(t**2,t)
    print(type(x_ext_integral))
    x_ext_array = [x_ext_integral(t) for t in times]
    length_foot = 0.26
    alphaF_min = states[:,1]
    return x_ext_array - np.sin(-alphaF_min)*length_foot
#0.3m
(x_ext_1,x_ext_2,x_ext_3,x_ext_4) = set_x_ext()
ankle = AnkleModel()
# sol = solve_ivp(ankle.get_derivative,[0,6],[0.8,8,-4],rtol = 1e-5, atol = 1e-8,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,input))
#sol = solve_ivp(ankle.get_derivative,[0,3],[0.5,-15,0],rtol = 1e-5, atol = 1e-8,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,trapezoid_wave))
sol = solve_ivp(ankle.get_derivative,[0,0.40625],[0.8,-15,10],rtol = 1e-5, atol = 1e-8,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,trapezoid_wave))
times = sol.t
states = sol.y.T
toe_clear = toe_clearance(states,times)
plt.plot(times,toe_clear)
plt.show()
# plt.subplot(3, 1, 1)
# plt.plot(times,states[:,0])
# plt.xlabel('Time (s)')
# plt.ylabel('Activation')
# plt.subplot(3, 1, 2)
# plt.plot(times,states[:,1])
# plt.xlabel('Time (s)')
# plt.ylabel('Angle')
# plt.subplot(3, 1, 3)
# plt.plot(times,states[:,2])
# plt.xlabel('Time (s)')
# plt.ylabel('Rotational velocity')
# plt.show()