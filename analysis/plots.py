import matplotlib.pyplot as plt
import numpy as np
import constants as cons
import FF 
import diffDecayOmegaPi as dDR


s_values=np.linspace(.93,1.8,100)
y_values_1=np.zeros(len(s_values))
y_values_2=np.zeros(len(s_values))
y_values_3=np.zeros(len(s_values))

for i in range (0,len(s_values)):
    y_values_1[i]=dDR.AFB(s_values[i]**2,0.005)
    y_values_2[i]=dDR.AFB(s_values[i]**2,-0.005)
    y_values_3[i]=dDR.AFB(s_values[i]**2,0.0)

plt.figure(figsize=(6,6))
plt.plot(s_values, y_values_1, label=r'$\epsilon_T=0.005$', color="blue")
plt.plot(s_values, y_values_2, label=r'$\epsilon_T=-0.005$', color="green")
plt.plot(s_values, y_values_3, label=r'$\epsilon_T=0.0$', color="black")
plt.ylabel(r'$A_{FB}(s)$')
plt.xlabel('s[GeV]')
plt.title("Forward-Backward assymetry")
plt.xlim([.9,1.8])
plt.ylim([-.03,.03])

plt.legend()
#plt.show()



FV_values=np.zeros(len(s_values))
FT1_values=np.zeros(len(s_values))
FT2_values=np.zeros(len(s_values))
FT3_values=np.zeros(len(s_values))

for i in range(len(s_values)):
    FV_values[i]=np.abs(FF.FV(s_values[i]**2))
    FT1_values[i]=np.abs(FF.FT1(s_values[i]**2))
    FT2_values[i]=np.abs(FF.FT2(s_values[i]**2))
    FT3_values[i]=np.abs(FF.FT3(s_values[i]**2))


plt.figure(figsize=(6,6))
plt.plot(s_values, FV_values, label=r'$F_V$', color="blue")
plt.plot(s_values, FT1_values, label=r'$F_{T1}$', color="green")
plt.plot(s_values, FT2_values, label=r'$F_{T2}$', color="black")
plt.plot(s_values, FT3_values, label=r'$F_{T3}$', color="yellow")

#plt.ylabel(r'$\frac{1}{\Gamma}\frac{d\Gamma}{ds}$')
plt.ylabel(r'$F(s)$')
plt.xlabel(r'$\sqrt{s}$[GeV]')
plt.title("Differential decay rate")
plt.xlim([.9,1.8])
plt.ylim([0.0,11.5])

plt.legend()
#plt.show()



FV_arg_values=np.zeros(len(s_values))
FT1_arg_values=np.zeros(len(s_values))
FT2_arg_values=np.zeros(len(s_values))
FT3_arg_values=np.zeros(len(s_values))

for i in range(len(s_values)):
    FV_arg_values[i]=np.angle(FF.FV(s_values[i]**2),deg=True) 
    FT1_arg_values[i]=np.angle(FF.FT1(s_values[i]**2),deg=True) 
    FT2_arg_values[i]=np.angle(FF.FT2(s_values[i]**2),deg=True) 
    FT3_arg_values[i]=np.angle(FF.FT3(s_values[i]**2),deg=True) 


plt.figure(figsize=(6,6))
plt.plot(s_values, FV_arg_values, label=r'$F_V$', color="blue")
plt.plot(s_values, FT1_arg_values, label=r'$F_{T1}$', color="green")
plt.plot(s_values, FT2_arg_values, label=r'$F_{T2}$', color="black")
plt.plot(s_values, FT3_arg_values, label=r'$F_{T3}$', color="yellow")

#plt.ylabel(r'$\frac{1}{\Gamma}\frac{d\Gamma}{ds}$')
plt.ylabel(r'$F(s)$')
plt.xlabel(r'$\sqrt{s}$[GeV]')
plt.title("Differential decay rate")
plt.xlim([.9,1.8])
plt.ylim([-180,180])

plt.legend()
#plt.show()

diffDecay_1=np.zeros(len(s_values))
diffDecay_2=np.zeros(len(s_values))
diffDecay_3=np.zeros(len(s_values))

def N(s):
    return 32 * np.pi**2 * cons.mTau**3/(cons.GF**2 * cons.Vud**2 * (cons.mTau**2 - s)**2 * (cons.mTau**2 + 2 * s))


for i in range(len(s_values)):
    diffDecay_1[i]=dDR.dGamma(s_values[i]**2,5e-4) * N(s_values[i]**2)
    diffDecay_2[i]=dDR.dGamma(s_values[i]**2,-5e-4) * N(s_values[i]**2)
    diffDecay_3[i]=dDR.dGamma(s_values[i]**2,0.0) * N(s_values[i]**2)

plt.figure(figsize=(6,6))
plt.plot(s_values, diffDecay_1, label=r'$\epsilon_T=0.005$', color="blue")
plt.plot(s_values, diffDecay_2, label=r'$\epsilon_T=-0.005$', color="green")
plt.plot(s_values, diffDecay_3, label=r'$\epsilon_T=0.0$', color="black")
#plt.ylabel(r'$\frac{1}{\Gamma}\frac{d\Gamma}{ds}$')
plt.ylabel(r'$\nu(s)$')
plt.xlabel('s[GeV]')
plt.title("Differential decay rate")
plt.xlim([.9,1.8])
plt.ylim([0.0,.05])

plt.legend()
plt.show()