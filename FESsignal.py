from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def trapezoid_wave(t, width=2., slope=1, amp=1.):
    a = slope*width*signal.sawtooth(2*np.pi*(t+0.12)/width, width=0.5)/4.
    a[a>amp/2.] = amp/2.
    a[a<-amp/2.] = -amp/2.
    return a + amp/2.

t = np.linspace(0, 0.40625, 501)
l = "width=2, slope=2, amp=1"
plt.plot(t,trapezoid_wave(t,width=0.65,slope=7,amp=1), label=l)

plt.legend()
plt.show()