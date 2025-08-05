import numpy as np
from scipy.integrate import quad

import FF 
import constants as cons


########  Constantes ######

GF=1.1663788e-5
Vud=0.97373
Vus=0.22534
SEW=1.0201
mTau=1.77686 

seg=(1.0e25)/6.58
#TTau=290.3e-15 * seg
TTau = cons.TTau



'''def lam(a,b,c):
    x= a**2 + b**2 + c**2 - 2 * ( a * b + a * c + c * b)

    return np.where(x<0,0,x)

def XV(s,Fv):
    return (mTau**2 +2 *s) * np.abs(Fv(s))**2

def XT(s,Fv,FT2,FT3):
    return 12 * mTau * np.real( np.conjugate(Fv(s)) * (FT3(s) - FT2(s)))

def XAT(s,A1,A2,A3,FT1,FT2,FT3,mV,mP):
    DeltaVP=mV**2 - mP**2
    SigmaVP=mV**2 + mP**2

    l1=4 * mV**2 * np.real(np.conjugate(FT3(s)) * (A2(s)-A3(s) - 6 * A1(s) * (DeltaVP + s)/lam(s,mV**2,mP**2)))
    l2=-np.real(np.conjugate(FT1(s)) * (2 * A1(s) * (s-DeltaVP) + (A3(s)-A2(s)) * lam(s,mV**2,mP**2)))
    l3=2 * np.real(np.conjugate(FT2(s)) * ( (A2(s)-A3(s)) * (s-SigmaVP)-2 * A1(s) * (s+DeltaVP) * (s-SigmaVP -4 * mV**2)/lam(s,mV**2,mP**2)))
    
    return (3 * mTau/(2 * mV**2)) * (l1+l2+l3)

def X2A(s,A1,A2,A3,mV,mP):

    DeltaVP=mV**2 - mP**2
    SigmaVP=mV**2 + mP**2

    l1=4 * np.abs(A1(s))**2 * ( 2 * mTau**2 + s + 6 * mV**2 * s * (mTau**2 + 2 * s)/lam(s,mV**2,mP**2) )
    l2=np.abs(A2(s))**2 * ( (2 * mTau**2 + s) * lam(s,mV**2,mP**2) + 6 * s * mP**2 * mTau**2)
    l3=np.abs(A3(s))**2 * ( (2 * mTau**2 + s) * lam(s,mV**2,mP**2) + 6 * s * mV**2 * mTau**2)
    l4=4 * np.real(np.conjugate(A1(s)) * A3(s) ) * (s-DeltaVP) *(2 * mTau**2+s)
    l5=-4*np.real(np.conjugate(A1(s)) * A2(s)) * (DeltaVP * (2 * mTau**2 +s) - s * (mTau**2-s))
    l6=-2*np.real(np.conjugate(A2(s)*A3(s))) * (3 * mTau**2 * (DeltaVP**2 -s **2) + 3 * s * (s**2 -2 * s * SigmaVP +DeltaVP**2)+ (mTau**2-s)*lam(s,mV**2,mP**2))

    return (l1+l2+l3+l4+l5+l6)/(4 * mV**2 * s)

def X2T(s,FT1,FT2,FT3,mV,mP):
    
    DeltaVP=mV**2 - mP**2
    SigmaVP=mV**2 + mP**2   

    l1= np.abs(FT1(s))**2 * s * (2 * mTau**2 + s) * lam(s,mV**2,mP**2)**2 
    l2= 4 * np.abs(FT2(s))**2 * (2 * mTau**2 + s) * ( 12 * mP**2 * mV**2 * s + (4 * mV**2 +s) * lam(s,mV**2,mP**2))
    l3=16 * np.abs(FT3(s))**2 * mV**2 * (2 * mTau**2 + s) * (lam(s,mV**2,mP**2) +3 * mV**2 * s)
    l4=4* s * (2 * mTau**2 +s) * lam(s,mV**2,mP**2) * ( 2* np.real(np.conjugate(FT3(s) * FT1(s)))* mV**2 + np.real(np.conjugate(FT2(s))* FT1(s)) * (SigmaVP+s))
    l5=-16 * mV**2 * np.real( np. conjugate(FT3(s)) * FT2(s)) * (lam(s,mV**2,mP**2) * (mTau**2 -s) + 3 * (mTau**2+s) * DeltaVP**2 - 3 * s**2 * ( mTau**2 +SigmaVP))

    return (l1+l2+l3+l4+l5)/(2 * s* mV**2 * lam(s,mV**2,mP**2))'''

def lam(a, b, c):
    x = a**2 + b**2 + c**2 - 2 * (a * b + a * c + c * b)
    return np.where(x < 0, 0, x)

def XV(s, Fv):
    return (mTau**2 + 2 * s) * np.abs(Fv(s))**2

def XT(s, Fv, FT2, FT3):
    return 12 * mTau * np.real(np.conjugate(Fv(s)) * (FT3(s) - FT2(s)))

def XAT(s, A1, A2, A3, FT1, FT2, FT3, mV, mP):
    DeltaVP = mV**2 - mP**2
    SigmaVP = mV**2 + mP**2
    lam_val = lam(s, mV**2, mP**2)
    # Avoid division by zero
    l1 = 4 * mV**2 * np.real(np.conjugate(FT3(s)) * (A2(s) - A3(s) - 6 * A1(s) * (DeltaVP + s) / np.where(lam_val > 0, lam_val, 1e-10)))
    l2 = -np.real(np.conjugate(FT1(s)) * (2 * A1(s) * (s - DeltaVP) + (A3(s) - A2(s)) * lam_val))
    l3 = 2 * np.real(np.conjugate(FT2(s)) * ((A2(s) - A3(s)) * (s - SigmaVP) - 2 * A1(s) * (s + DeltaVP) * (s - SigmaVP - 4 * mV**2) / np.where(lam_val > 0, lam_val, 1e-10)))
    return (3 * mTau / (2 * mV**2)) * (l1 + l2 + l3)

def X2A(s, A1, A2, A3, mV, mP):
    DeltaVP = mV**2 - mP**2
    SigmaVP = mV**2 + mP**2
    lam_val = lam(s, mV**2, mP**2)
    # Avoid division by zero
    l1 = 4 * np.abs(A1(s))**2 * (2 * mTau**2 + s + 6 * mV**2 * s * (mTau**2 + 2 * s) / np.where(lam_val > 0, lam_val, 1e-10))
    l2 = np.abs(A2(s))**2 * ((2 * mTau**2 + s) * lam_val + 6 * s * mP**2 * mTau**2)
    l3 = np.abs(A3(s))**2 * ((2 * mTau**2 + s) * lam_val + 6 * s * mV**2 * mTau**2)
    l4 = 4 * np.real(np.conjugate(A1(s)) * A3(s)) * (s - DeltaVP) * (2 * mTau**2 + s)
    l5 = -4 * np.real(np.conjugate(A1(s)) * A2(s)) * (DeltaVP * (2 * mTau**2 + s) - s * (mTau**2 - s))
    l6 = -2 * np.real(np.conjugate(A2(s) * A3(s))) * (3 * mTau**2 * (DeltaVP**2 - s**2) + 3 * s * (s**2 - 2 * s * SigmaVP + DeltaVP**2) + (mTau**2 - s) * lam_val)
    return (l1 + l2 + l3 + l4 + l5 + l6) / (4 * mV**2 * s)

def X2T(s, FT1, FT2, FT3, mV, mP):
    DeltaVP = mV**2 - mP**2
    SigmaVP = mV**2 + mP**2
    lam_val = lam(s, mV**2, mP**2)
    # Avoid division by zero
    l1 = np.abs(FT1(s))**2 * s * (2 * mTau**2 + s) * lam_val**2
    l2 = 4 * np.abs(FT2(s))**2 * (2 * mTau**2 + s) * (12 * mP**2 * mV**2 * s + (4 * mV**2 + s) * lam_val)
    l3 = 16 * np.abs(FT3(s))**2 * mV**2 * (2 * mTau**2 + s) * (lam_val + 3 * mV**2 * s)
    l4 = 4 * s * (2 * mTau**2 + s) * lam_val * (2 * np.real(np.conjugate(FT3(s) * FT1(s))) * mV**2 + np.real(np.conjugate(FT2(s)) * FT1(s)) * (SigmaVP + s))
    l5 = -16 * mV**2 * np.real(np.conjugate(FT3(s)) * FT2(s)) * (lam_val * (mTau**2 - s) + 3 * (mTau**2 + s) * DeltaVP**2 - 3 * s**2 * (mTau**2 + SigmaVP))
    return (l1 + l2 + l3 + l4 + l5) / (2 * s * mV**2 * np.where(lam_val > 0, lam_val, 1e-10))


def dGamma(s,Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epshR,epshT,Vckm):

    #GGeff= (GF**2 * SEW * mTau * Vud**2 * (1-s/mTau**2)**2 * lam(s,mV**2,mP**2)**(3/2) * (1 + epsilonL + epsilonR)**2)/(1536. * np.pi**3 * s**2)
    GGeff= (GF**2 * SEW * mTau * Vckm**2 * (1-s/mTau**2)**2 * lam(s,mV**2,mP**2)**(3/2))/(1536. * np.pi**3 * s**2)
    #epshT=epsilonT/(1+epsilonL+epsilonR)
    #epshR=epsilonR/(1+epsilonL+epsilonR)

    return GGeff * (XV(s,Fv) + 2 * epshT * XT(s,Fv,FT2,FT3) + (1-2 * epshR) * X2A(s,A1,A2,A3,mV,mP) -2 * epshT * (1-2*epshR) * XAT(s,A1,A2,A3,FT1,FT2,FT3,mV,mP) + 4 * epshT**2 * X2T(s,FT1,FT2,FT3,mV,mP) )

'''def Gamma(Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epsilonL,epsilonR,epsilonT):

    s0=(mV + mP)**2
    smax=mTau**2

    result, error = quad(lambda s: dGamma(s,Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epsilonL,epsilonR,epsilonT),s0,smax)

    return result'''

def Gamma(Fv, A1, A2, A3, FT1, FT2, FT3, mV, mP, epshR, epshT, Vckm):
    s0 = (mV + mP)**2
    smax = mTau**2
    result, error = quad(lambda s: dGamma(s, Fv, A1, A2, A3, FT1, FT2, FT3, mV, mP, epshR, epshT, Vckm), s0 + 1e-6, smax, epsrel=1e-4)
    return result

def BRVP(Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epshR,epshT, Vckm):
    return TTau * Gamma(Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epshR,epshT,Vckm)

#def dNVP(s,Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epsilonL,epsilonR,epsilonT):
 #   return dGamma(s,Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epsilonL,epsilonR,epsilonT)/Gamma(Fv,A1,A2,A3,FT1,FT2,FT3,mV,mP,epsilonL,epsilonR,epsilonT)
def dNVP(s, Fv, A1, A2, A3, FT1, FT2, FT3, mV, mP, epshR, epshT, Vckm):
    gamma = Gamma(Fv, A1, A2, A3, FT1, FT2, FT3, mV, mP, epshR, epshT, Vckm)
    return dGamma(s, Fv, A1, A2, A3, FT1, FT2, FT3, mV, mP, epshR, epshT, Vckm) / np.where(gamma > 0, gamma, 1e-10)

def A0(s):
    return 0

def A0OmegaK(s):
    return 0
def A0RhoK(s):
    return 0
def A0KsPi(s):
    return 0

# For tau->Omega+Pi+nu
print( BRVP(Fv=FF.FV,A1=A0,A2=A0,A3=A0,FT1=FF.FT1,FT2=FF.FT2,FT3=FF.FT3,mV=cons.mOmega,mP=cons.mPi,epshR=0.0,epshT=0.0,Vckm=Vud) )
# For tau->Omega+K+nu
print( BRVP(Fv=FF.FVOmegaK,A1=FF.A1OmegaK,A2=FF.A2OmegaK,A3=FF.A3OmegaK,FT1=FF.FT1OmegaK,FT2=FF.FT2OmegaK,FT3=FF.FT3OmegaK,mV=cons.mOmega,mP=cons.mK,epshR=0.0,epshT=0.0,Vckm=Vus) )
# For tau->Rho+K+nu
print( BRVP(Fv=FF.FVRhoK,A1=FF.A1RhoK,A2=FF.A2RhoK,A3=FF.A3RhoK,FT1=FF.FT1RhoK,FT2=FF.FT2RhoK,FT3=FF.FT3RhoK,mV=cons.mRho,mP=cons.mK,epshR=0.0,epshT=0.0,Vckm=Vus) )
# For tau->Ks+Pi+nu
print( BRVP(Fv=FF.FVKsPi,A1=FF.A1KsPi,A2=FF.A2KsPi,A3=FF.A3KsPi,FT1=FF.FT1KsPi,FT2=FF.FT2KsPi,FT3=FF.FT3KsPi,mV=cons.mKs,mP=cons.mPi,epshR=0.0,epshT=0.0,Vckm=Vus) )