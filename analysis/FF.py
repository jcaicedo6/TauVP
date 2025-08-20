import constants as cons
import numpy as np

DeltaOmegaPi=cons.mOmega**2 - cons.mPi**2
SigmaOmegaPi=cons.mOmega**2 + cons.mPi**2

DeltaOmegaK=cons.mOmega**2 - cons.mK0**2
SigmaOmegaK=cons.mOmega**2 + cons.mK0**2

DeltaRhoK=cons.mRho**2 - cons.mK0**2
SigmaRhoK=cons.mRho**2 + cons.mK0**2

DeltaKsPi=cons.mKs**2 - cons.mPi**2
SigmaKsPi=cons.mKs**2 + cons.mPi**2

DeltaKsK=cons.mKs**2 - cons.mK0**2
SigmaKsK=cons.mKs**2 + cons.mK0**2

# Sigma for omega pi
#def sigma(s,mP):
 #   return np.piecewise(s,[s< 4 * mP**2, s> 4 * mP**2],[0.0, lambda s: np.sqrt(1.0 - 4.0 * mP**2/s)])

def sigma(s, mP):
    """
    Calculates the phase space factor for a particle decay, robust against negative sqrt arguments.
    """
    # Clip the argument of the sqrt at zero to prevent invalid values
    sqrt_arg = np.maximum(0.0, 1.0 - 4.0 * mP**2 / s)
    
    # Calculate the result
    result = np.sqrt(sqrt_arg)
    
    # The result is only valid for s >= 4 * mP**2.
    # Set the invalid values (where s < 4 * mP**2) to 0.0
    return np.where(s >= 4.0 * mP**2, result, 0.0)


# Sigma for omega K
# m1 = mK or mKs, m2 = mPi or mEta or mRho 
#def sigmaOmegaK(s,m1, m2):
 #   return np.piecewise(s,[s< (m1 +m2)**2, s> (m1 +m2)**2],[0.0, lambda s: np.sqrt((s - (m1 + m2)**2) * (s - (m1 - m2)**2))/s])


def sigmaOmegaK(s, m1, m2):
    """
    Calculates the phase space factor for two particles, robust against negative sqrt arguments.
    """
    # The argument of the square root
    #sqrt_arg = (s - (m1 + m2)**2) * (s - (m1 - m2)**2)
    
    # Clip the argument of the sqrt at zero to prevent invalid values
    #clipped_arg = np.maximum(0.0, sqrt_arg)
    
    # Calculate the result
    #result = np.sqrt(clipped_arg) / s
    
    # The result is only valid for s >= (m1 + m2)**2.
    # Set the invalid values to 0.0
    #return np.where(s >= (m1 + m2)**2, result, 0.0)
    expr = (s - (m1 + m2)**2) * (s - (m1 - m2)**2)
    return (1/s) * np.sqrt(np.maximum(expr, 1e-12)) * np.heaviside(s - (m1 + m2)**2, 0.5)

def sigmaKsK(s, m1, m2):
    """
    Calculates the phase space factor for two particles, robust against negative sqrt arguments.
    """
    # The argument of the square root
    #sqrt_arg = (s - (m1 + m2)**2) * (s - (m1 - m2)**2)
    
    # Clip the argument of the sqrt at zero to prevent invalid values
    #clipped_arg = np.maximum(0.0, sqrt_arg)
    
    # Calculate the result
    #result = np.sqrt(clipped_arg) / 2
    
    # The result is only valid for s >= (m1 + m2)**2.
    # Set the invalid values to 0.0
    #return np.where(s >= (m1 + m2)**2, result, 0.0)
    expr = (s - (m1 + m2)**2) * (s - (m1 - m2)**2)
    return 0.5 * np.sqrt(np.maximum(expr, 1e-12)) * np.heaviside(s - (m1 + m2)**2, 0.5)



def Drho(s):

    GamRho = (s * cons.mRho/(96 * np.pi * cons.F**2)) * (sigma(s,cons.mPi)**3  + .5 * sigma(s,cons.mK0)**3) 

    return 1.0/( cons.mRho**2 - s - 1j * cons.mRho * GamRho)

def Drhop(s):

    GamRhop = (s * cons.GammaRhoprime/cons.mRhoprime**2) * (sigma(s,cons.mPi)**3  + .5 * sigma(s,cons.mK0)**3)/(sigma(cons.mRhoprime**2,cons.mPi)**3  + .5 * sigma(cons.mRhoprime**2,cons.mK0)**3) 

    return 1.0/( cons.mRhoprime**2 - s - 1j * cons.mRhoprime * GamRhop)

def DK(s):
    return 1.0/( cons.mK0**2 - s)

def Dpi(s):

    return 1.0/( cons.mPi**2 - s)

def DKs(s):

    GamKs = (s * cons.Gamma0Ks/cons.mKs**2) * (sigmaOmegaK(s,cons.mK0,cons.mPi)**3  + sigmaOmegaK(s,cons.mK0,cons.mEta)**3) * (1.0/(sigmaOmegaK(cons.mKs**2,cons.mK0,cons.mPi)**3  + sigmaOmegaK(cons.mKs**2,cons.mK0,cons.mEta)**3))

    return 1.0/( cons.mKs**2 - s - 1j * cons.mKs * GamKs)

def DKsp(s):

    GamKsp = (s * cons.Gamma0Ksp/cons.mKsp**2) * (sigmaOmegaK(s,cons.mK0,cons.mPi)**3  + sigmaOmegaK(s,cons.mK0,cons.mEta)**3) * (1.0/(sigmaOmegaK(cons.mKsp**2,cons.mK0,cons.mPi)**3  + sigmaOmegaK(cons.mKsp**2,cons.mK0,cons.mEta)**3))

    return 1.0/( cons.mKsp**2 - s - 1j * cons.mKsp * GamKsp)

def DK1H(s):

    GamK1H = (s * cons.Gamma0K1H/cons.mK1H**2) * (sigmaOmegaK(s,cons.mK0,cons.mRho)**3  + sigmaOmegaK(s,cons.mKs,cons.mPi)**3) * (1.0/(sigmaOmegaK(cons.mK1H**2,cons.mK0,cons.mRho)**3  + sigmaOmegaK(cons.mK1H**2,cons.mKs,cons.mPi)**3))

    return 1.0/( cons.mK1H**2 - s - 1j * cons.mK1H * GamK1H)

def DK1L(s):

    GamK1L = (s * cons.Gamma0K1L/cons.mK1L**2) * (sigmaOmegaK(s,cons.mK0,cons.mRho)**3  + sigmaOmegaK(s,cons.mKs,cons.mPi)**3) * (1.0/(sigmaOmegaK(cons.mK1L**2,cons.mK0,cons.mRho)**3  + sigmaOmegaK(cons.mK1L**2,cons.mKs,cons.mPi)**3))

    return 1.0/( cons.mK1L**2 - s - 1j * cons.mK1L * GamK1L)

def Da1(s):

    Gama1 = (s * cons.Gamma0a1/cons.ma1**2) * (sigmaKsK(s,cons.mPi,cons.mRho)**3  + (sigmaKsK(s,cons.mK0, cons.mKs)**3 / 2)) * (1.0/(sigmaKsK(cons.ma1**2,cons.mPi,cons.mRho)**3  + (sigmaKsK(cons.ma1**2,cons.mK0,cons.mKs)**3 /2)))

    return 1.0/( cons.ma1**2 - s - 1j * cons.ma1 * Gama1)

def FV(s):

    l1 = (2 * np.sqrt(2)/(cons.F * cons.mOmega) ) * (2 * cons.d3 * cons.FV + cons.ds * cons.FV1) + (4 * np.sqrt(2) * cons.FV/(cons.F * cons.mOmega) ) * (cons.d12 * cons.mPi**2 + cons.d3 * (s + DeltaOmegaPi)) * Drho(s) 
    l2 = (2 * np.sqrt(2) * cons.FV1/(cons.F * cons.mOmega)) * ( cons.dmM + cons.ds * s ) * Drhop(s)
    
    return l1 + l2

def FVOmegaK(s):

    l1 = (np.sqrt(2)/(cons.FK * cons.mOmega)) * (2 *cons.d3OK * cons.FVOK + cons.dsOK * cons.FV1OK) + (2 * np.sqrt(2) * cons.FVOK/(cons.FK * cons.mOmega)) * (cons.d12 * cons.mK0**2 + cons.d3OK * (s + DeltaOmegaK)) * DKs(s)
    l2 = (np.sqrt(2) * cons.FV1OK/(cons.FK * cons.mOmega)) * (cons.dmOK * cons.mK0**2 + cons.dMOK * cons.mOmega**2 + cons.dsOK * s) * DKsp(s)

    return l1 + l2

def FVRhoK(s):

    l1 = (np.sqrt(2)/(cons.FK * cons.mRho)) * (2 * cons.d3OK * cons.FVOK + cons.dsOK * cons.FV1OK) + (2 * np.sqrt(2) * cons.FVOK/(cons.FK * cons.mRho)) * (cons.d12 * cons.mK0**2 + cons.d3OK * (s + DeltaRhoK)) * DKs(s)
    l2 = (np.sqrt(2) * cons.FV1OK/(cons.FK * cons.mRho)) * (cons.dmOK * cons.mK0**2 + cons.dMOK * cons.mRho**2 + cons.dsOK * s) * DKsp(s)

    return l1 + l2

def FVKsPi(s):

    l1 = (np.sqrt(2)/(cons.FOK * cons.mKs)) * (2 * cons.d3OK * cons.FVOK + cons.dsOK * cons.FV1OK) + (np.sqrt(2) * cons.FVOK/(cons.FOK * cons.mKs)) * (cons.d12 * cons.mPi**2 + cons.d3OK * (s + DeltaKsPi)) * DKs(s)
    l2 = (np.sqrt(2) * cons.FV1OK/(cons.FOK * cons.mKs)) * (cons.dmOK * cons.mPi**2 + cons.dMOK * cons.mKs**2 + cons.dsOK * s) * DKsp(s)

    return l1 + l2

def FVKsK(s):

    l1 = (2 * np.sqrt(2) / (cons.FK * cons.mKs)) * (2 * cons.d3OK * cons.FVOK + cons.dsOK * cons.FV1OK) + (4 * np.sqrt(2) * cons.FVOK/(cons.FK * cons.mKs)) * (cons.d12 * cons.mK0**2 + cons.d3OK * (s + DeltaKsK)) * Drho(s)
    l2 = (2 * np.sqrt(2) * cons.FV1OK/(cons.FK * cons.mKs)) * (cons.dmOK * cons.mK0**2 + cons.dMOK * cons.mKs**2 + cons.dsOK * s) * Drhop(s)

    return l1 + l2

def FT1(s):

    l1 = (4 * cons.FVT/(cons.F * cons.mOmega * cons.mRho**2)) * (cons.d12 * cons.mPi**2 + cons.d3 * (DeltaOmegaPi + cons.mRho**2)) * Drho(s)
    l2 = -(4 * cons.FV1T/(cons.F * cons. mOmega * cons.mRhoprime**2)) * ( cons.dd * cons.mOmega**2 - (cons.db + 4 * cons.df) * cons.mPi**2 - (cons.da-cons.db) * cons.mRhoprime**2 ) * Drhop(s)
    
    return l1 + l2

def FT1OmegaK(s):

    l1 = (4 * cons.FVT/(cons.FK * cons.mOmega * cons.mKs**2)) * (cons.d12 * cons.mK0**2 + cons.d3OK * (DeltaOmegaK + cons.mKs**2)) * DKs(s)
    l2 = -(4 * cons.FV1T/(cons.FK * cons.mOmega * cons.mKsp**2)) * ( cons.dd * cons.mOmega**2 - (cons.db + 4 * cons.df) * cons.mK0**2 - (cons.da-cons.db) * cons.mKsp**2 ) * DKsp(s)
    
    return l1 + l2

def FT1RhoK(s):

    l1 = (4 * cons.FVT/(cons.FK * cons.mRho * cons.mKs**2)) * (cons.d12 * cons.mK0**2 + cons.d3OK * (DeltaRhoK + cons.mKs**2)) * DKs(s)
    l2 = -(4 * cons.FV1T/(cons.FK * cons.mRho * cons.mKsp**2)) * ( cons.dd * cons.mRho**2 - (cons.db + 4 * cons.df) * cons.mK0**2 - (cons.da-cons.db) * cons.mKsp**2 ) * DKsp(s)
    
    return l1 + l2

def FT1KsPi(s):

    l1 = (4 * cons.FVT/(cons.FOK * cons.mKs * cons.mKs**2)) * (cons.d12 * cons.mPi**2 + cons.d3OK * (DeltaKsPi + cons.mKs**2)) * DKs(s)
    l2 = -(4 * cons.FV1T/(cons.FOK * cons.mKs * cons.mKsp**2)) * ( cons.dd * cons.mKs**2 - (cons.db + 4 * cons.df) * cons.mPi**2 - (cons.da-cons.db) * cons.mKsp**2 ) * DKsp(s)
    
    return l1 + l2

def FT1KsK(s):

    l1 = (4 * cons.FVT/(cons.FK * cons.mKs * cons.mRho**2)) * (cons.d12 * cons.mK0**2 + cons.d3OK * (DeltaKsK + cons.mRho**2)) * Drho(s)
    l2 = -(4 * cons.FV1T/(cons.FK * cons.mKs * cons.mRhoprime**2)) * ( cons.dd * cons.mKs**2 - (cons.db + 4 * cons.df) * cons.mK0**2 - (cons.da-cons.db) * cons.mRhoprime**2 ) * Drhop(s)
    
    return l1 + l2

def FT2(s):

    l1 =  -(2 * cons.FVT/(cons.F * cons.mOmega * cons.mRho**2)) * ( (cons.d12 * cons.mPi**2 - cons.d3 * (SigmaOmegaPi - cons.mRho**2)) + ( cons.d12 * cons.mPi**2 * (s + DeltaOmegaPi) - cons.d3 * (s * SigmaOmegaPi - DeltaOmegaPi**2 + cons.mRho**2 * ( 2 * cons.mOmega**2 + s + DeltaOmegaPi))) * Drho(s) )
    l2 =  -(2 * cons.FV1T/(cons.F * cons.mOmega * cons.mRhoprime**2)) * ((cons.dd * cons.mOmega**2 + (cons.db + 4 * cons.df) * cons.mPi**2 + (cons.da - cons.db) * cons.mRhoprime**2) * (1 + (s - SigmaOmegaPi) * Drhop(s)) + 2 * cons.mOmega**2 * ((cons.db + cons.dd + 4 * cons.df) * cons.mPi**2 + (cons.da - cons.db - cons.dd) * cons.mRhoprime**2) * Drhop(s))

    return l1 + l2

def FT2OmegaK(s):

    l1 =  -(2 * cons.FVT/(cons.FK * cons.mOmega * cons.mKs**2)) * ( (cons.d12 * cons.mK0**2 - cons.d3OK * (SigmaOmegaK - cons.mKs**2)) + ( cons.d12 * cons.mK0**2 * (s + DeltaOmegaK) - cons.d3OK * (s * SigmaOmegaK - DeltaOmegaK**2 + cons.mKs**2 * ( 2 * cons.mOmega**2 + s + DeltaOmegaK))) * DKs(s) )
    l2 =  -(2 * cons.FV1T/(cons.FK * cons.mOmega * cons.mKsp**2)) *  ((cons.dd * cons.mOmega**2 + (cons.db + 4 * cons.df) * cons.mKs**2 + (cons.da - cons.db) * cons.mKsp**2) * (1 + (s - SigmaOmegaK) * DKsp(s)) + 2 * cons.mOmega**2 * ((cons.db + cons.dd + 4 * cons.df) * cons.mK0**2 + (cons.da - cons.db - cons.dd) * cons.mKsp**2) * DKsp(s))

    return l1 + l2

def FT2RhoK(s):

    l1 =  -(2 * cons.FVT/(cons.FK * cons.mRho * cons.mKs**2)) * ( (cons.d12 * cons.mK0**2 - cons.d3OK * (SigmaRhoK - cons.mKs**2)) + ( cons.d12 * cons.mK0**2 * (s + DeltaRhoK) - cons.d3OK * (s * SigmaRhoK - DeltaRhoK**2 + cons.mKs**2 * ( 2 * cons.mRho**2 + s + DeltaRhoK))) * DKs(s) )
    l2 =  -(2 * cons.FV1T/(cons.FK * cons.mRho * cons.mKsp**2)) *  ((cons.dd * cons.mRho**2 + (cons.db + 4 * cons.df) * cons.mKs**2 + (cons.da - cons.db) * cons.mKsp**2) * (1 + (s - SigmaRhoK) * DKsp(s)) + 2 * cons.mRho**2 * ((cons.db + cons.dd + 4 * cons.df) * cons.mK0**2 + (cons.da - cons.db - cons.dd) * cons.mKsp**2) * DKsp(s))

    return l1 + l2

def FT2KsPi(s):

    l1 =  -(2 * cons.FVT/(cons.FOK * cons.mKs * cons.mKs**2)) * ( (cons.d12 * cons.mPi**2 - cons.d3OK * (SigmaKsPi - cons.mKs**2)) + ( cons.d12 * cons.mPi**2 * (s + DeltaKsPi) - cons.d3OK * (s * SigmaKsPi - DeltaKsPi**2 + cons.mKs**2 * ( 2 * cons.mKs**2 + s + DeltaKsPi))) * DKs(s) )
    l2 =  -(2 * cons.FV1T/(cons.FOK * cons.mKs * cons.mKsp**2)) *  ((cons.dd * cons.mKs**2 + (cons.db + 4 * cons.df) * cons.mKs**2 + (cons.da - cons.db) * cons.mKsp**2) * (1 + (s - SigmaKsPi) * DKsp(s)) + 2 * cons.mKs**2 * ((cons.db + cons.dd + 4 * cons.df) * cons.mPi**2 + (cons.da - cons.db - cons.dd) * cons.mKsp**2) * DKsp(s))

    return l1 + l2

def FT2KsK(s):

    l1 = -(2 * cons.FVT/(cons.FK * cons.mKs * cons.mRho**2)) * ((cons.d12 * cons.mK0**2 - cons.d3OK * (SigmaKsK - cons.mRho**2)) + (cons.d12 * cons.mK0**2 * (s + DeltaKsK) - cons.d3OK * ( s * SigmaKsK - DeltaKsK**2 + cons.mRho**2 * (2 * cons.mKs**2 + s + DeltaKsK))) * Drho(s))
    l2 = -(2 * cons.FV1T/(cons.FK * cons.mKs * cons.mRhoprime**2)) * ((cons.dd * cons.mKs**2 + (cons.db + 4 * cons.df) * cons.mK0**2 + (cons.da-cons.db) * cons.mRhoprime**2) * (1 + (s - SigmaKsK) * Drhop(s)) + 2 * cons.mKs**2 * ((cons.db + cons.dd + 4 * cons.df) * cons.mK0**2 + (cons.da - cons.db - cons.dd) * cons.mRhoprime**2) * Drhop(s))

    return l1 + l2  

def FT3(s):

    l1= - (2 * cons.FVT/(cons.F * cons.mOmega * cons.mRho**2)) * ( (cons.d12 * cons.mPi**2 - cons.d3 * (SigmaOmegaPi + cons.mRho**2)) + (cons.d12 * cons.mPi**2 * ( s + DeltaOmegaPi - 2 * cons.mRho**2) - cons.d3 * ( s * SigmaOmegaPi - DeltaOmegaPi**2 + cons.mRho**2 * (s - SigmaOmegaPi))) * Drho(s) )
    l2= - (2 * cons.FV1T/(cons.F * cons.mOmega * cons.mRhoprime**2)) * ( (cons.dd * cons.mOmega**2 + ( cons.db + 4 * cons.df) * cons.mPi**2 - (cons.da - cons.db) * cons.mRhoprime**2) * (1 + (s - SigmaOmegaPi) * Drhop(s)) + 2 * cons.mPi**2 * ( (cons.db + cons.dd + 4 * cons.df) * cons.mOmega**2 - (cons.da + 4 * cons.df) * cons.mRhoprime**2) * Drhop(s) )

    return l1 + l2

def FT3OmegaK(s):

    l1= - (2 * cons.FVT/(cons.FK * cons.mOmega * cons.mKs**2)) * ( (cons.d12 * cons.mK0**2 - cons.d3OK * (SigmaOmegaK + cons.mKs**2)) + (cons.d12 * cons.mK0**2 * (s + DeltaOmegaK - 2 * cons.mKs**2) - cons.d3OK * ( s * SigmaOmegaK - DeltaOmegaK**2 + cons.mKs**2 * (s - SigmaOmegaK))) * DKs(s) )
    l2= - (2 * cons.FV1T/(cons.FK * cons.mOmega * cons.mKsp**2)) * ( (cons.dd * cons.mOmega**2 + ( cons.db + 4 * cons.df) * cons.mK0**2 - (cons.da - cons.db) * cons.mKsp**2) * (1 + (s - SigmaOmegaK) * DKsp(s)) + 2 * cons.mK0**2 * ( (cons.db + cons.dd + 4 * cons.df) * cons.mOmega**2 - (cons.da + 4 * cons.df) * cons.mKsp**2) * DKsp(s) )

    return l1 + l2

def FT3RhoK(s):

    l1= - (2 * cons.FVT/(cons.FK * cons.mRho * cons.mKs**2)) * ( (cons.d12 * cons.mK0**2 - cons.d3OK * (SigmaRhoK + cons.mKs**2)) + (cons.d12 * cons.mK0**2 * (s + DeltaRhoK - 2 * cons.mKs**2) - cons.d3OK * ( s * SigmaRhoK - DeltaRhoK**2 + cons.mKs**2 * (s - SigmaRhoK))) * DKs(s) )
    l2= - (2 * cons.FV1T/(cons.FK * cons.mRho * cons.mKsp**2)) * ( (cons.dd * cons.mRho**2 + ( cons.db + 4 * cons.df) * cons.mK0**2 - (cons.da - cons.db) * cons.mKsp**2) * (1 + (s - SigmaRhoK) * DKsp(s)) + 2 * cons.mK0**2 * ( (cons.db + cons.dd + 4 * cons.df) * cons.mRho**2 - (cons.da + 4 * cons.df) * cons.mKsp**2) * DKsp(s) )

    return l1 + l2

def FT3KsPi(s):

    l1= - (2 * cons.FVT/(cons.FOK * cons.mKs * cons.mKs**2)) * ( (cons.d12 * cons.mPi**2 - cons.d3OK * (SigmaKsPi + cons.mKs**2)) + (cons.d12 * cons.mPi**2 * (s + DeltaKsPi - 2 * cons.mKs**2) - cons.d3OK * ( s * SigmaKsPi - DeltaKsPi**2 + cons.mKs**2 * (s - SigmaKsPi))) * DKs(s) )
    l2= - (2 * cons.FV1T/(cons.FOK * cons.mKs * cons.mKsp**2)) * ( (cons.dd * cons.mKs**2 + ( cons.db + 4 * cons.df) * cons.mPi**2 - (cons.da - cons.db) * cons.mKsp**2) * (1 + (s - SigmaKsPi) * DKsp(s)) + 2 * cons.mPi**2 * ( (cons.db + cons.dd + 4 * cons.df) * cons.mKs**2 - (cons.da + 4 * cons.df) * cons.mKsp**2) * DKsp(s) )

    return l1 + l2

def FT3KsK(s):

    l1 = - (2 * cons.FVT/(cons.FK * cons.mKs * cons.mRho**2)) * ((cons.d12 * cons.mK0**2 - cons.d3OK * (SigmaKsK + cons.mRho**2)) + (cons.d12 * cons.mK0**2 * (s + DeltaKsK - 2 * cons.mRho**2) - cons.d3OK * ( s * SigmaKsK - DeltaKsK**2 + cons.mRho**2 * (s - SigmaKsK))) * Drho(s))
    l2 = - (2 * cons.FV1T/(cons.FK * cons.mKs * cons.mRhoprime**2)) * ((cons.dd * cons.mKs**2 + (cons.db + 4 * cons.df) * cons.mK0**2 - (cons.da - cons.db) * cons.mRhoprime**2) * (1 + (s - SigmaKsK) * Drhop(s)) + 2 * cons.mK0**2 * ((cons.db + cons.dd + 4 * cons.df) * cons.mKs**2 - (cons.da + 4 * cons.df) * cons.mRhoprime**2) * Drhop(s) )

    return l1 + l2

def A1OmegaK(s):

    A1_a = - cons.FVOK * (DeltaOmegaK + s) - 2 * cons.Gv * (SigmaOmegaK - s) 
    A1_b = (cons.lam_0 * cons.mK0**4 + (cons.mOmega**2 - s) * (cons.lam_p * cons.mOmega**2 - cons.lam_pp * s)) - cons.mK0**2 * (cons.lam_0 * cons.mOmega**2 + cons.lam_p * cons.mOmega**2 + cons.lam_0 * s + cons.lam_pp * s)
    A1_c = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return np.sqrt(2) * 1j * (1.0 / (4 * cons.mOmega * cons.FK)) * (A1_a + (2 * np.sqrt(2) * cons.FA * A1_b * A1_c))

def A2OmegaK(s):
    
    A2_a = - (cons.Gv * cons.mOmega * DK(s))/cons.FK  
    A2_b = (np.sqrt(2) * cons.FA * cons.mOmega) / (cons.FK )* (cons.lam_p + cons.lam_pp)
    A2_c = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return 1j * np.sqrt(2) * (A2_a + (A2_b * A2_c))

def A3OmegaK(s):
    
    A3_a = (cons.FVOK - 2 * cons.Gv) / (2 * cons.FK * cons.mOmega) 
    A3_b = - (cons.Gv * cons.mOmega * DK(s)) / cons.FK
    A3_c = (np.sqrt(2) * cons.FA) / (cons.FK * cons.mOmega) * (cons.lam_0 * cons.mK0**2 + cons.lam_pp  * cons.mOmega**2 - cons.lam_pp *s)
    A3_d = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return 1j * np.sqrt(2) * (A3_a + A3_b + (A3_c * A3_d))


def A1RhoK(s):

    A1_a = - cons.FVOK * (DeltaRhoK + s) - 2 * cons.Gv * (SigmaRhoK - s)
    A1_b = (cons.lam_0 * cons.mK0**4 + (cons.mRho**2 - s) * (cons.lam_p * cons.mRho**2 - cons.lam_pp * s)) - cons.mK0**2 * (cons.lam_0 * cons.mRho**2 + cons.lam_p * cons.mRho**2 + cons.lam_0 * s + cons.lam_pp * s)
    A1_c = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))
    
    return  np.sqrt(2) * 1j * (1.0 / (4 * cons.mRho * cons.FK)) * (A1_a + 2 * np.sqrt(2) * cons.FA * (A1_b * A1_c))

def A2RhoK(s):
    
    A2_a = - (cons.Gv * cons.mRho * DK(s))/cons.FK  
    A2_b = (np.sqrt(2) * cons.FA * cons.mRho) / cons.FK * (cons.lam_p + cons.lam_pp)
    A2_c = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return 1j * np.sqrt(2) * (A2_a + (A2_b * A2_c))

def A3RhoK(s):
    
    A3_a = (cons.FVOK - 2 * cons.Gv) / (2 * cons.FK * cons.mRho) 
    A3_b = - (cons.Gv * cons.mRho * DK(s)) / cons.FK
    A3_c = np.sqrt(2) * cons.FA / (cons.FK * cons.mRho) * (cons.lam_0 * cons.mK0**2 + cons.lam_pp * cons.mRho**2 - cons.lam_pp *s)
    A3_d = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return 1j * np.sqrt(2) * (A3_a + A3_b + (A3_c * A3_d))

def A1KsPi(s):
    
    A1_a = - cons.FVOK * (DeltaKsPi + s) - 2 * cons.Gv * (SigmaKsPi - s)
    A1_b = (cons.lam_0 * cons.mPi**4 + (cons.mKs**2 - s) * (cons.lam_p * cons.mKs**2 - cons.lam_pp * s)) - cons.mPi**2 * (cons.lam_0 * cons.mKs**2 + cons.lam_p * cons.mKs**2 + cons.lam_0 * s + cons.lam_pp * s)
    A1_c = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))
    
    return  np.sqrt(2) * 1j * (1.0 / (4 * cons.mKs * cons.FOK)) * (A1_a + (2 * np.sqrt(2) * cons.FA * A1_b * A1_c))

def A2KsPi(s):
    
    A2_a = - cons.Gv * cons.mKs * DK(s)/cons.FOK  
    A2_b = np.sqrt(2) * cons.FA * cons.mKs / cons.FOK * (cons.lam_p + cons.lam_pp)
    A2_c = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return 1j * np.sqrt(2) * (A2_a + (A2_b * A2_c))

def A3KsPi(s):
    
    A3_a = (cons.FVOK - 2 * cons.Gv) / (2 * cons.FOK * cons.mKs) 
    A3_b = - (cons.Gv * cons.mKs * DK(s)) / cons.FOK
    A3_c = (np.sqrt(2) * cons.FA )/ (cons.FOK * cons.mKs) * (cons.lam_0 * cons.mPi**2 + cons.lam_pp * cons.mKs**2 - cons.lam_pp *s)
    A3_d = (cons.c0**2 * DK1H(s) + cons.s0**2 * DK1L(s))

    return 1j * np.sqrt(2) * (A3_a + A3_b + (A3_c * A3_d))

def A1KsK(s):

    A1_a = cons.FVOK * (cons.mK0**2 - cons.mKs**2 - s) - 2 * cons.Gv * (cons.mK0**2 * cons.mKs**2 - s)
    A1_b = cons.lam_0 * cons.mK0**4 + (cons.mKs**2 - s) * (cons.lam_p * cons.mKs**2 - cons.lam_pp * s)
    A1_c = cons.mK0**2 * (cons.lam_0 * cons.mKs**2 + cons.lam_p * cons.mKs**2 + cons.lam_0 * s + cons.lam_pp * s)
    
    return  (-1.0 / (2 * np.sqrt(2) * cons.mKs * cons.FK)) * (A1_a + (2 * np.sqrt(2) * cons.FA * (A1_b - A1_c) * Da1(s)))

def A2KsK(s):

    return  (np.sqrt(2) * cons.Gv * cons.mKs / cons.FK) * Dpi(s) - (2 * cons.FA * cons.mKs / cons.FK) * (cons.lam_p + cons.lam_pp) * Da1(s)

def A3KsK(s):

    return  (- cons.FVOK + 2 * cons.Gv) / (np.sqrt(2) * cons.FK * cons.mKs) + (np.sqrt(2) * cons.Gv * cons.mKs / cons.FK) * Dpi(s) - (2 * cons.FA / (cons.FK * cons.mKs)) * (cons.lam_0 * cons.mK0**2 + cons.lam_pp * cons.mKs**2 - cons.lam_pp * s) * Da1(s)
