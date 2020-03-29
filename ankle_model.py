import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp,quad
from scipy.interpolate import interp1d
from scipy import signal
#from FESsignal import trapezoid_wave

class AnkleModel:
    """
    TA muscle model adapted from Benoussaad et al. (2013)
    """
    def __init__(self):
        self.Tact = 0.01 #s (Activation constant time)
        self.Tdeact = 0.04 #s (Relaxation constant time)
        self.J = 0.0197 # kg.m^2 (Inertia of the foot around ankle)
        self.d = 0.037 # m (moment arm of TA wrt ankle)
        self.tendon_length = .223 #m   Model assumes constant tendon length
        self.resting_length_muscle_tendon = .321 #m
        self.av = 1.33
        self.fv1 = 0.18
        self.fv2 = 0.023
        self.vmax = -0.9 #m/s
        self.Fmax = 600 #N
        self.W = 0.56 #shape parameter of f_fl
        self.a = [2.1, -0.08, -7.97, 0.19, -1.79]
        #optimal_length_CE is the "optimal length of the fiber at which the maximal force can be generated"
        self.optimal_length_CE = .075 #m - source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3130447/
        self.m = 1.0275 #kg (mass of foot)
        self.COM = .1145 #m (center of mass location with respect to the ankle)

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
        g = -9.81 #gravity acceleration
        return self.m*self.COM*np.cos(np.deg2rad(x[1]))*g

    def T_acc(self, x, x_ext):
        """
        :param x: state vector
        :param x_ext: external states
        :return: torque due to movement of the ankle
        """
        return self.m*self.COM*(x_ext[0]*np.sin(np.deg2rad(x[1]))-x_ext[1]*np.cos(np.deg2rad(x[1])))

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
        #
        # if x[0]>1:
        #     x[0] =1
        B = 0.82 #viscosity parameter
        x1_dot = (u_val-x[0])*(u_val/self.Tact-(1-u_val)/self.Tdeact)
        x2_dot = x[2]
        x3_dot = 1/self.J*(self.F_m(x,x_ext)*self.d + self.T_grav(x) + self.T_acc(x, x_ext) + self.T_ela(x) + B*(x_ext_4(t)-x[2]))
        print(x3_dot)
        return [x1_dot, x2_dot, x3_dot]

def set_x_ext():
    """
   :param x_ext: external states (linear horizontal acc, linear vertical acc, absolute orientation of shank, absolute rotational velocity)
   :return: x_ext: external states returned
   """
    x_ext_1 = np.array([
        [55.966414637202874, -0.12765779690045553],
        [56.738376707464795, -0.1831112410710558],
        [57.209574334767524, -0.19098542127908447],
        [57.82363507247587, -0.24645975186933455],
        [58.44772129161619, -0.2701867245916705],
        [59.54551150841722, -0.2938510380550565],
        [60.635782614144276, -0.34132586991937863],
        [61.898993274572874, -0.34115877856217924],
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
    x = 0.40625 * ((x_ext_1[:, 0] - 55.966414637202874) / (100 - 55.966414637202874))
    y = 9.81*x_ext_1[:, 1]
    f1 = interp1d(x, y, fill_value="extrapolate")

    x_ext_2_new = np.array([
        [7.893473645030628, -1.5458536265137672],
        [7.8966528421431255, -1.2806950588330763],
        [7.896449914667859, -1.0838554078252156],
        [7.899629111780357, -0.8186968401445238],
        [7.907982959512134, -0.5494797229585147],
        [7.912887040164392, -0.2829683054427172],
        [7.9192454343893885, 0.24734882991866702],
        [7.925942041073162, 0.4495998802669501],
        [7.938016225851479, 0.4590698291126891],
        [7.9466406435502766, 0.4658340782882169],
        [7.958850113312105, 0.34407759312871544],
        [7.974644635136961, 0.09380037363418459],
        [7.981882381754777, -0.22885431203849427],
        [7.987665814799852, -0.8153147155567595],
        [7.989728910798389, -1.142027950734755],
        [7.993516890336684, -1.467388336077644],
        [7.999232680890005, -1.98823552259329],
        [8.003155945411812, -2.4448223419414195],
        [8.010528977013138, -2.8987034616193394],
        [8.014519884026699, -3.420903497970089],
        [8.018443148548505, -3.8774903173182205],
        [8.019119573466057, -4.533622487344422],
        [8.023042837987864, -4.990209306692553],
        [8.030483512080945, -5.509703643373091],
        [8.036469872601288, -6.293003697899216],
        [8.038735896075089, -6.816556584085074],
        [8.04803673869144, -7.465924504935748],
        [8.05189236072149, -7.856898107281259],
        [8.054158384195292, -8.380450993467113],
        [8.05662733514436, -9.10084353066083],
        [8.062140198222416, -9.424851066168616],
        [8.071102828379988, -9.74615290200619],
        [8.086626780237825, -9.733977253490238],
        [8.093188101938088, -9.400499769136715],
        [8.101271379702844, -8.868829783940226],
        [8.100933167244067, -8.540763698927124],
        [8.103909436881299, -8.078765480238571],
        [8.108813517533557, -7.812254062722774],
        [8.120481847361344, -7.409104811861313],
        [8.128700410109609, -7.008661260670065],
        [8.137054257841385, -6.7394441434840555],
        [8.149128442619702, -6.7299741946383165],
        [8.156298546745763, -6.987015663308375],
        [8.161676124840307, -7.17979676481092],
        [8.16691841795134, -7.241351432308223],
        [8.172363638537641, -7.499745750813386],
        [8.187955232887234, -7.553183319300057],
        [8.194854767046271, -7.5477719199596365],
        [8.205204068284829, -7.539654820949002],
        [8.210378718904108, -7.535596271443685],
        [8.213828485983626, -7.532890571773475],
        [8.220457450175644, -7.26502630442257],
        [8.223839574763408, -7.19670738774974],
        [8.240817840193984, -6.920726021388202],
        [8.244267607273501, -6.918020321717991],
        [8.25316259493932, -7.1737089405529435],
        [8.26723223322442, -7.425339009882579],
        [8.277852104429998, -7.679674778882429],
        [8.283297325016298, -7.938069097387592],
        [8.286950019571083, -8.132203048725241],
        [8.295777364745147, -8.322278450557574],
        [8.299362416808176, -8.450799184892604],
        [8.313229127618008, -8.505589603214379],
        [8.318606705712554, -8.698370704716924],
        [8.323848998823587, -8.759925372214227],
        [8.329226576918131, -8.95270647371677],
        [8.336396681044192, -9.20974794238683],
        [8.343837355137271, -9.729242279067368],
        [8.351075101755088, -10.051896964740047],
        [8.354998366276893, -10.508483784088178],
        [8.359124558273965, -11.161910254444168],
        [8.361255296764256, -11.554236706624785]
    ])
    x=x_ext_2_new[:,0] - 7.893473645030628
    y = -x_ext_2_new[:,1]
    f2_new = interp1d(x, y, fill_value="extrapolate")

    x_ext_2 = np.array([
        [56.18810685474131, -1.3385826771653542],
        [56.71996521907154, -1.2834645669291336],
        [56.73155886189072, -1.3149606299212597],
        [57.43877107386116, -1.2362204724409447],
        [57.97932467030577, -1.2047244094488188],
        [58.51408144534079, -1.1574803149606296],
        [59.05463504178542, -1.1259842519685037],
        [59.411139558475426, -1.0944881889763776],
        [59.58069658470605, -1.0551181102362202],
        [60.12125018115066, -1.0236220472440942],
        [60.454567412202294, -0.9291338582677162],
        [60.47485628713588, -0.9842519685039368],
        [61.16467803487754, -0.8582677165354327],
        [61.51828414086275, -0.8188976377952752],
        [61.86609342543838, -0.7637795275590548],
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
    x = 0.40625 * ((x_ext_2[:, 0]-56.18810685474131) / (100-56.18810685474131))
    y = -9.81*x_ext_2[:, 1]
    f2 = interp1d(x, y, fill_value="extrapolate")

    x_ext_1_new = np.array([
        [7.92172337678564, -13.492149431510597],
        [7.92159843405106, -12.879930032068671],
        [7.9296364166423725, -12.266044729499015],
        [7.929553121485986, -11.857898463204393],
        [7.933530465203448, -11.346882678772253],
        [7.933384698679772, -10.632626712756682],
        [7.937382866186331, -10.22364749489821],
        [7.939298654783224, -9.61101161967435],
        [7.945337553621258, -9.20161592603393],
        [7.945275082253967, -8.895506226312982],
        [7.947232518429054, -8.486943484236434],
        [7.9491274832368495, -7.772271042438916],
        [7.949044188080463, -7.364124776144294],
        [7.95300070800883, -6.75107242513851],
        [7.952875765274249, -6.138853025696591],
        [7.958935487901378, -5.831493898629837],
        [7.960872100287368, -5.320894589979623],
        [7.960788805130981, -4.912748323685015],
        [7.968930906667778, -4.809045853983626],
        [7.972929074174337, -4.400066636125146],
        [7.976968889259089, -4.19516055141397],
        [7.9850276956394985, -3.683311815417966],
        [7.989088334513348, -3.580442297280449],
        [7.9952105285077675, -3.5791928699346514],
        [8.00337345383366, -3.5775269668069214],
        [8.00337345383366, -3.5775269668069214],
        [8.015617841822499, -3.5750281121153193],
        [8.017616925575778, -3.3705385031860757],
        [8.021635916871434, -3.0635958519012547],
        [8.025675731956186, -2.8586897671900715],
        [8.033755362125692, -2.4488775977677335],
        [8.03975261338553, -1.8354087709800098],
        [8.043729957102995, -1.3243929865478634],
        [8.045708217067178, -1.017866811044975],
        [8.051705468327016, -0.40439798425725826],
        [8.057827662321436, -0.4031485569114608],
        [8.063949856315855, -0.4018991295656562],
        [8.072112781641748, -0.40023322643792625],
        [8.07215442921994, -0.6043063595852232],
        [8.082483028611886, -1.214443380117487],
        [8.08258714755737, -1.7246262129857541],
        [8.088771812919079, -2.0294864853609127],
        [8.099121236100121, -2.7416600724668285],
        [8.099121236100121, -2.7416600724668285],
        [8.107388280371497, -3.2501770022073586],
        [8.113572945733207, -3.5550372745825243],
        [8.113572945733207, -3.5550372745825243],
        [8.121694223480905, -3.349298238307476],
        [8.127795593686228, -3.2460122443880266],
        [8.131814584981884, -2.9390695931032056],
        [8.135875223855733, -2.8362000749656815],
        [8.135812752488443, -2.5300903752447184],
        [8.139873391362293, -2.4272208571072014],
        [8.139790096205907, -2.019074590812579],
        [8.143788263712466, -1.610095372954106],
        [8.147786431219025, -1.201116155095626],
        [8.15586606138853, -0.7913039856732738],
        [8.15986422889509, -0.38232476781479363],
        [8.170005414185166, -0.07413268918416094],
        [8.180167423264338, 0.13202282287280553],
        [8.184311357294574, -0.17325392528428551],
        [8.196659864228895, -0.6809379034609506],
        [8.198825538294948, -1.2927408271209373],
        [8.207092582566323, -1.8012577568614816],
        [8.21329807171713, -2.2081545958102993],
        [8.2216067635667, -2.9207446586981405],
        [8.221690058723086, -3.3288909249927485],
        [8.223855732789138, -3.940693848652735],
        [8.223876556578235, -4.042730415226394],
        [8.226083878222482, -4.858606472033692],
        [8.234371746282953, -5.469159968347881],
        [8.23451751280663, -6.18341593436346],
        [8.236620715505394, -6.489109158302476],
        [8.240806297113823, -6.998459039606892],
        [8.240847944692016, -7.202532172754189],
        [8.240972887426597, -7.8147515721961085],
        [8.247178376577402, -8.221648411144933],
        [8.249281579276165, -8.527341635083957],
        [8.247386614468368, -9.242014076881468],
        [8.247407438257465, -9.34405064345512],
        [8.261775852734164, -9.749281579276207],
        [8.27204198075882, -10.0533089000875],
        [8.27204198075882, -10.0533089000875],
        [8.273999416933906, -9.64474615801096],
        [8.273936945566616, -9.33863645828999],
        [8.277893465494982, -8.725584107284199],
        [8.28189163300154, -8.316604889425726],
        [8.287888884261381, -7.703136062638002],
        [8.287847236683188, -7.499062929490691],
        [8.289763025280081, -6.886427054266839],
        [8.289638082545501, -6.27420765482492],
        [8.293594602473867, -5.661155303819129],
        [8.297592769980426, -5.252176085960635],
        [8.303569197451168, -4.536670692599266],
        [8.311565532464288, -3.7187122568823128],
        [8.311482237307901, -3.3105659905876905],
        [8.31348132106118, -3.106076381658454],
        [8.31941610095373, -2.186497855149767],
        [8.321331889550622, -1.5738619799259155],
        [8.321206946816043, -0.9616425804839892],
        [8.32110282787056, -0.45145974761572205],
        [8.327120902919496, 0.05997251259834968],
        [8.333118154179335, 0.6734413393860734],
        [8.334992295198035, 1.490150347757229],
        [8.336949731373121, 1.8987130898337838],
        [8.340927075090583, 2.409728874265916],
        [8.340843779934197, 2.817875140560524],
        [8.342801216109283, 3.2264378826370788],
        [8.3426970971638, 3.736620715505346],
        [8.346611969513972, 4.553746199658441],
        [8.350547665653242, 5.2688351172378916],
        [8.356565740702179, 5.780267377451956]
    ])
    x = x_ext_1_new[:,0] - 7.92172337678564
    y = -x_ext_1_new[:, 1]
    f1_new = interp1d(x, y, fill_value="extrapolate")

    x_ext_4 = np.array([
        [56.57541412380124, -73.25196163905832],
        [57.04489973844812, -56.9790758500435],
        [58.160418482999134, -43.94071490845681],
        [59.27332170880558, -27.65039232781163],
        [60.38622493461203, -11.360069747166449],
        [61.18003487358325, 1.6695727986050883],
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
    x = 0.40625 * ((x_ext_4[:, 0]-56.57541412380124) / (100-56.57541412380124))
    y = x_ext_4[:, 1]
    f4 = interp1d(x, y, fill_value="extrapolate")

    x_ext_4_new = np.array([
        [7.907216494845361, -0.9355670103092777],
        [7.922680412371134, -0.849656357388314],
        [7.93298969072165, -0.6812714776632269],
        [7.93298969072165, -0.5146048109965626],
        [7.938144329896907, -0.013745704467352127],
        [7.93298969072165, -0.26460481099656086],
        [7.943298969072165, 0.15378006872852445],
        [7.943298969072165, 0.23711340206185838],
        [7.943298969072165, 0.32044673539519053],
        [7.943298969072165, 0.40378006872852357],
        [7.943298969072165, 0.6537800687285245],
        [7.948453608247423, 0.9046391752577332],
        [7.958762886597938, 1.0730240549828194],
        [7.958762886597938, 1.3230240549828194],
        [7.963917525773196, 1.573883161512029],
        [7.969072164948454, 1.9080756013745717],
        [7.979381443298969, 2.159793814432991],
        [7.989690721649485, 2.4115120274914092],
        [7.989690721649485, 2.744845360824743],
        [8.010309278350515, 2.9149484536082477],
        [8.02061855670103, 2.833333333333334],
        [8.036082474226804, 3.085910652920963],
        [8.04639175257732, 3.337628865979382],
        [8.061855670103093, 3.673539518900344],
        [8.061855670103093, 4.006872852233678],
        [8.082474226804123, 4.010309278350516],
        [8.092783505154639, 3.6786941580756016],
        [8.108247422680414, 3.9312714776632305],
        [8.118556701030927, 4.266323024054983],
        [8.1340206185567, 4.3522336769759455],
        [8.144329896907216, 4.603951890034365],
        [8.149484536082474, 4.938144329896907],
        [8.164948453608247, 5.190721649484535],
        [8.175257731958762, 4.942439862542955],
        [8.195876288659793, 4.612542955326461],
        [8.216494845360826, 4.5326460481099655],
        [8.231958762886599, 4.201890034364261],
        [8.231958762886599, 3.7852233676975953],
        [8.237113402061855, 3.286082474226805],
        [8.231958762886599, 3.4518900343642613],
        [8.242268041237114, 2.9536082474226806],
        [8.247422680412372, 2.621134020618557],
        [8.257731958762886, 2.3728522336769764],
        [8.262886597938145, 2.123711340206186],
        [8.268041237113403, 1.9579037800687296],
        [8.268041237113403, 1.4579037800687296],
        [8.278350515463918, 1.1262886597938158],
        [8.278350515463918, 0.7096219931271497],
        [8.293814432989691, 0.5455326460481116],
        [8.304123711340207, 0.21391752577319778],
        [8.31958762886598, 0.049828178694159675],
        [8.34020618556701, -0.030068728522335775]
    ])
    x = x_ext_4_new[:, 0] - 7.907216494845361
    y = 180/np.pi*x_ext_4_new[:, 1]
    f4_new = interp1d(x, y, fill_value="extrapolate")

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
    x = x_ext_3[:, 0] - 2.8359375
    y = x_ext_3[:, 1]
    # f3 = interp1d(x, y, fill_value="extrapolate")
    sol = solve_ivp(shank_derivative, [0, 0.40625], [0], first_step=0.01, max_step=0.01, args=(f4,))
    f3 = interp1d(sol.t, sol.y.T[:,0], fill_value="extrapolate")
    plt.figure()
    plt.plot(sol.t, f4(sol.t))
    plt.plot(sol.t, sol.y.T[:, 0])
    plt.show(block=False)
    return f1, f2, f3, f4
    # return f1_new,f2_new,f3_new,f4_new
def input(t):
    # return 0.3
    # return np.sin(t) / 2 + 0.5
    return np.sin(3*t)/3+0.6

def trapezoid_wave(t, width=0.65, slope=7, amp=1):
    a = slope*width*signal.sawtooth((2*np.pi*(t+0.12))/width, width=0.5)/4.
    #a = slope * width * signal.sawtooth((2 * np.pi * (t + 0.18)) / width, width=0.5) / 4.
    if a>amp/2.:
        a = amp/2.
    elif a<-amp/2.:
        a = -amp/2.
    return a + amp/2.
def toe_clearance(states,x_ext_2):
    solve = solve_ivp(differential_eqn,[0,.40625],[0,0],first_step = 0.01, max_step = 0.01, args=(x_ext_2,))
    solutions = solve.y.T
    print(f"length of toe clear sol: {len(solutions)}")
    length_foot = .26
    alphaF_min = states[:,1]
    print(f"length of alpha min: {len(alphaF_min)}")
    print(solutions[:,0])
    print("ankle to toe")
    print(np.sin(np.deg2rad(-alphaF_min))*length_foot)
    # plt.figure()
    # plt.plot(solve.t,x_ext_2(solve.t))
    # plt.plot(solve.t, solutions[:, 1])
    # plt.plot(solve.t,solutions[:,0])
    # plt.show(block=False)
    return solutions[:,0] - np.sin(np.deg2rad(-alphaF_min))*length_foot

def differential_eqn(t,x,x_ext):
    x1_dot = x[1]
    x2_dot = x_ext(t)
    return [x1_dot, x2_dot]

def shank_derivative(t,x,x_ext_4):
    x1_dot = x_ext_4(t)
    return x1_dot

def zero_input(t):
    return 0.5

#0.3m
(x_ext_1,x_ext_2,x_ext_3,x_ext_4) = set_x_ext()
ankle = AnkleModel()
# sol = solve_ivp(ankle.get_derivative,[0,6],[0.8,8,-4],rtol = 1e-5, atol = 1e-8,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,input))
#sol = solve_ivp(ankle.get_derivative,[0,3],[0.5,-15,0],rtol = 1e-5, atol = 1e-8,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,trapezoid_wave))
# sol = solve_ivp(ankle.get_derivative,[0,0.406],[0.8,-15,20], first_step = 0.01, max_step = 0.01,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,trapezoid_wave))
sol = solve_ivp(ankle.get_derivative,[0,0.40625],[0,-30,0], first_step = 0.01, max_step = 0.01,args=(x_ext_1,x_ext_2,x_ext_3,x_ext_4,trapezoid_wave))
times = sol.t
states = sol.y.T
toe_clear = toe_clearance(states,x_ext_2)
plt.figure()
plt.plot(times,toe_clear)
plt.show(block=False)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(times,states[:,0])
plt.xlabel('Time (s)')
plt.ylabel('Activation')
plt.subplot(3, 1, 2)
plt.plot(times,states[:,1])
plt.xlabel('Time (s)')
plt.ylabel('Angle')
plt.subplot(3, 1, 3)
plt.plot(times,states[:,2])
plt.xlabel('Time (s)')
plt.ylabel('Rotational velocity')
plt.show(block=False)
plt.show()