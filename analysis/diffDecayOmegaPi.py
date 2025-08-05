import FF as FF
import constants as cons

import numpy as np
from scipy.integrate import quad


DeltaOmegaPi= cons.mOmega**2 - cons.mPi**2


def lam(a,b,c):
    return a**2 + b**2 + c**2 - 2 * ( a * b + a * c + c * b) 

def d2Gamma(s,theta,epsilonT):

    A = (cons.GF**2 * cons.Vud**2 * cons.SEW/(2048 * np.pi**3 * s**2 * cons.mTau**3)) * (s - cons.mTau**2)**2 * lam(s,cons.mOmega**2, cons.mPi**2)**(3/2) * ( (cons.mTau**2 + s) * np.abs(FF.FV(s))**2 + 16 * cons.mTau * epsilonT * np.real(FF.FV(s) * (np.conjugate(FF.FT3(s) - FF.FT2(s))))  )
    B= - (cons.GF**2 * cons.Vud**2 * cons.SEW/(128 * np.pi**3 * s**2 * cons.mTau**2)) * (s - cons.mTau**2)**2 * lam(s,cons.mOmega**2,cons.mPi**2) * epsilonT * ( (s - DeltaOmegaPi) * np.real(FF.FV(s) * np.conjugate(FF.FT2(s))) + (s + DeltaOmegaPi) * np.real(FF.FV(s) * np.conjugate(FF.FT3(s))) ) 
    C= (cons.GF**2 * cons.SEW * cons.Vud**2/(2048 * np.pi**3 * s**2 * cons.mTau**3)) * (s - cons.mTau**2)**2 * lam(s,cons.mOmega**2, cons.mPi**2)**(3/2) * np.abs(FF.FV(s))**2

    return A + B * np.cos(theta) + C * np.cos(theta)**2

def dGamma(s,epsilonT):
    return ( (cons.GF**2 * cons.Vud**2 * cons.SEW)/(1536 * np.pi**3 * s**2 * cons.mTau**3) ) * (cons.mTau**2 - s )**2 * lam(s,cons.mOmega**2,cons.mPi**2)**(3/2) * ( (cons.mTau**2 + 2 * s) * np.abs(FF.FV(s))**2 + 24 * cons.mTau * epsilonT * np.real( FF.FV(s) * np.conjugate(FF.FT3(s) - FF.FT2(s))))
    
def AFB(s,epsilonT):
    
    A = (cons.GF**2 * cons.Vud**2 * cons.SEW/(2048 * np.pi**3 * s**2 * cons.mTau**3)) * (s - cons.mTau**2)**2 * lam(s,cons.mOmega**2, cons.mPi**2)**(3/2) * ( (cons.mTau**2 + s) * np.abs(FF.FV(s))**2 + 16 * cons.mTau * epsilonT * np.real(FF.FV(s) * (np.conjugate(FF.FT3(s) - FF.FT2(s))))  )
    B= - (cons.GF**2 * cons.Vud**2 * cons.SEW/(128 * np.pi**3 * s**2 * cons.mTau**2)) * (s- cons.mTau**2)**2 * lam(s,cons.mOmega**2,cons.mPi**2) * epsilonT * ( (s - DeltaOmegaPi) * np.real(FF.FV(s) * np.conjugate(FF.FT2(s))) + (s + DeltaOmegaPi) * np.real(FF.FV(s) * np.conjugate(FF.FT3(s))) ) 
    C= (cons.GF**2  * cons.Vud**2 * cons.SEW/(2048 * np.pi**3 * s**2 * cons.mTau**3)) * (s - cons.mTau**2)**2 * lam(s,cons.mOmega**2, cons.mPi**2)**(3/2) * np.abs(FF.FV(s))**2

    return 3 * B/(6 * A + 2 * C)

def Gamma(epsilonT):

    s0=(cons.mOmega + cons.mPi)**2
    smax=cons.mTau**2

    result, error = quad(lambda s: dGamma(s,epsilonT),s0,smax)

    return result
