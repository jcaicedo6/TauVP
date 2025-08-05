import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad
import sys
import iminuit
from iminuit.util import describe
import functools
import diffDecayVP as dVP
import FF
import constants as cons

Vus = 0.22534
def verbose_wrapper(func, every=10):  # More frequent output for debugging
    counter = {'calls': 0}
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        counter['calls'] += 1
        val = func(*args, **kwargs)
        if counter['calls'] % every == 0:
            print(f"[{counter['calls']:5d}] NLL = {val:.4f}, args = {args}")
        return val
    return wrapped

# --- Experimental Data ---
br_exp_values = np.array([4.1e-4, 1.4e-3, 2.2e-3])  # [tau->Omega+K, tau->Rho+K, tau->Kst+pi]
br_exp_errors = np.array([0.4e-4, 0.5e-3, 0.5e-3])

df = pd.read_csv('InvM_OmegaK.txt', sep='\s+', header=0)
x_data_omegaK = np.array(df['x_val_mean'])
y_data_omegaK = np.array(df['y_val_mean'])
errors_omegaK_down = np.array([row['y_val_mean'] - row['y_sigma_down'] for _, row in df.iterrows()])
errors_omegaK_up = np.array([row['y_sigma_up'] - row['y_val_mean'] for _, row in df.iterrows()])

bins = len(x_data_omegaK)
hist, bin_edges = np.histogram(x_data_omegaK, bins=bins, weights=y_data_omegaK, density=False)
bin_widths = np.diff(bin_edges) # Widths in q (GeV)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#mass_min, mass_max = bin_edges[0], bin_edges[-1]    # q range
s0 = (cons.mOmega + cons.mK)**2     # Minimum q^2
smax = cons.mTau**2.                # Maximum q^2   

# --- Model Functions ---
def mass_nll(epshR, epshT):
    expected_counts = np.zeros(bins)
    for i in range(bins):
        m_min, m_max = bin_edges[i], bin_edges[i + 1]
        s_min, s_max = m_min**2, m_max**2
        if s_min < s0 + 1e-6:
            s_min = s0 + 1e-6  # Avoid kinematic boundar
        if s_max > smax:
            s_max = smax
        # Invarianr mass spectrum for Omega+K
        # Integrate dN/ds over q^2, apply Jacobian 2m for dN/dm
        result, _ = quad(lambda s: dVP.dNVP(s, Fv=FF.FVOmegaK, A1=dVP.A0OmegaK, A2=dVP.A0OmegaK, A3=dVP.A0OmegaK,
                         FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                         mV=cons.mOmega, mP=cons.mK, epshR=epshR, epshT=epshT, Vckm=Vus) * 2 * np.sqrt(s),
                         s_min, s_max, epsrel=1e-3, limit=50)
        # Scale by total events and bin width in mass
        expected_counts[i] = result * hist.sum() / (m_max - m_min)  # Normalize per mass bin width
    expected_counts = np.clip(expected_counts, 1e-10, None)

    mass_nll = 0
    for obs, exp, err_down, err_up in zip(hist, expected_counts, errors_omegaK_down, errors_omegaK_up):
        sigma = err_up if obs > exp else err_down
        if sigma == 0:
            sigma = 1e-10
        mass_nll += 0.5 * ((obs - exp) / sigma)**2 + 0.5 * np.log(2 * np.pi * sigma**2)

    print(f"Mass NLL = {mass_nll:.4f}, args = ({epshR:.4f}, {epshT:.4f})")
    return mass_nll

def br_chi2(epshR, epshT):
    br1_theory = dVP.BRVP(Fv=FF.FVOmegaK, A1=dVP.A0OmegaK, A2=dVP.A0OmegaK, A3=dVP.A0OmegaK,
                         FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                         mV=cons.mOmega, mP=cons.mK, epshR=epshR, epshT=epshT, Vckm=Vus)
    br2_theory = dVP.BRVP(Fv=FF.FVRhoK, A1=dVP.A0RhoK, A2=dVP.A0RhoK, A3=dVP.A0RhoK,
                         FT1=FF.FT1RhoK, FT2=FF.FT2RhoK, FT3=FF.FT3RhoK,
                         mV=cons.mRho, mP=cons.mK, epshR=epshR, epshT=epshT, Vckm=Vus)
    br3_theory = dVP.BRVP(Fv=FF.FVKsPi, A1=dVP.A0KsPi, A2=dVP.A0KsPi, A3=dVP.A0KsPi,
                         FT1=FF.FT1KsPi, FT2=FF.FT2KsPi, FT3=FF.FT3KsPi,
                         mV=cons.mKs, mP=cons.mPi, epshR=epshR, epshT=epshT, Vckm=Vus)
    print(f"BRs: br1={br1_theory:.2e}, br2={br2_theory:.2e}, br3={br3_theory:.2e}")
    return ((br1_theory - br_exp_values[0]) / br_exp_errors[0])**2 + \
           ((br2_theory - br_exp_values[1]) / br_exp_errors[1])**2 + \
           ((br3_theory - br_exp_values[2]) / br_exp_errors[2])**2

def combined_nll(epshR, epshT):
    return mass_nll(epshR, epshT) + br_chi2(epshR, epshT)

# --- Perform the Fit ---
#print("\n--- Running iminuit Minimization ---")
#print("Function Parameters:", describe(combined_nll))
#initial_nll_val = combined_nll(0.1, 0.1)
#print(f"Initial NLL (initial values): {initial_nll_val:.4f}")

# Plot combined likelihood to debug

    

m = Minuit(combined_nll, epshR=0.1, epshT=0.1)
m.limits['epshR'] = (-2.0, 2.0)
m.limits['epshT'] = (-0.5, 0.5)

# Run minimization
m.migrad()  # Run the fit
m.hesse()   # Compute errors

# --- Extract and Print Results ---
print("\n--- Fit Results ---")
print(m)
for param in m.parameters:
    value, error = m.values[param], m.errors[param]
    print(f"{param}: {value:.4f} +/- {error:.4f}")
print(f"\nMinimum NLL: {m.fval:.4f}")
print("Correlation matrix:")
print(m.covariance.correlation())

# Check BR contributions
print("\nBranching Ratio Fit vs. Experimental:")
epshR_fit, epshT_fit = m.values['epshR'], m.values['epshT']
br1_fit = dVP.BRVP(Fv=FF.FVOmegaK, A1=dVP.A0OmegaK, A2=dVP.A0OmegaK, A3=dVP.A0OmegaK,
                   FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                   mV=cons.mOmega, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
br2_fit = dVP.BRVP(Fv=FF.FVRhoK, A1=dVP.A0RhoK, A2=dVP.A0RhoK, A3=dVP.A0RhoK,
                   FT1=FF.FT1RhoK, FT2=FF.FT2RhoK, FT3=FF.FT3RhoK,
                   mV=cons.mRho, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
br3_fit = dVP.BRVP(Fv=FF.FVKsPi, A1=dVP.A0KsPi, A2=dVP.A0KsPi, A3=dVP.A0KsPi,
                   FT1=FF.FT1KsPi, FT2=FF.FT2KsPi, FT3=FF.FT3KsPi,
                   mV=cons.mKs, mP=cons.mPi, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
print(f"BR1: Fit = {br1_fit:.4e}, Exp = {br_exp_values[0]:.4e} ± {br_exp_errors[0]:.4e}")
print(f"BR2: Fit = {br2_fit:.4e}, Exp = {br_exp_values[1]:.4e} ± {br_exp_errors[1]:.4e}")
print(f"BR3: Fit = {br3_fit:.4e}, Exp = {br_exp_values[2]:.4e} ± {br_exp_errors[2]:.4e}")

# --- Visualization and Goodness of Fit ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# BR plot
ax1.errorbar([0, 1, 2], br_exp_values, yerr=br_exp_errors, fmt='bo', label='Experimental', capsize=5)
ax1.plot([0, 1, 2], [br1_fit, br2_fit, br3_fit], 'rs', label='Fit')
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels([r"$\tau \to \Omega K$", r"$\tau \to \rho K$", r"$\tau \to K^* \pi$"])
ax1.set_ylabel('Branching Ratio')
ax1.set_title('Branching Ratios: Experimental vs. Fit')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, max(br_exp_values) * 1.5)

# Mass spectrum plot
ax2.hist(bin_centers, bins=bin_edges, weights=hist, density=False, alpha=0.5, label='Data')
ax2.errorbar(bin_centers, hist, yerr=[errors_omegaK_down, errors_omegaK_up], fmt='none', ecolor='black', label='Errors')

mass_range = np.linspace(bin_edges[0], bin_edges[-1], 500)
q2_range = mass_range ** 2
dN_fit_q2 = np.array([dVP.dNVP(q2, Fv=FF.FVOmegaK, A1=dVP.A0OmegaK, A2=dVP.A0OmegaK, A3=dVP.A0OmegaK,
                         FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                         mV=cons.mOmega, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus) for q2 in q2_range])
# Convert dN/dq^2 to dN/dm with Jacobian
dN_fit_mass = dN_fit_q2 * 2 * mass_range  # ds/dm = 2m
dN_fit_mass *= hist.sum() / np.trapz(dN_fit_mass, mass_range)  # Normalize to total events
ax2.plot(mass_range, dN_fit_mass, 'r-', label='Fitted dN/dm')

# Compute chi-squared (approximate for binned data)
pdf_binned = np.array([quad(lambda m: dVP.dNVP(m**2, Fv=FF.FVOmegaK, A1=dVP.A0OmegaK, A2=dVP.A0OmegaK, A3=dVP.A0OmegaK,
                         FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                         mV=cons.mOmega, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus) * 2 * m,
                           bin_edges[i], bin_edges[i+1], epsrel=1e-4, limit=100)[0] * hist.sum() for i in range(bins)])
chi2 = np.sum(((hist - pdf_binned) ** 2 / (hist + 1e-10)) / (errors_omegaK_down**2 + errors_omegaK_up**2) / 2)
dof = bins - 2  # 2 parameters: epshR, epshT
print(f"\nChi-squared / dof: {chi2:.5f} / {dof} = {chi2/dof:.5f}")

ax2.set_xlabel('Invariant Mass (GeV)')
ax2.set_ylabel('Events')
ax2.set_title('Invariant Mass Spectrum: Data vs. Fit')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('Minuit_Combined_Fit_Results.png')
plt.show()

print("\nScript finished.")