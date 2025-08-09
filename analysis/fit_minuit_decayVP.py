import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.integrate import quad

# Your custom modules assumed available:
import diffDecayVP as dVP
import FF
import constants as cons

Vus = 0.22534  # CKM element

# Load data from file
df = pd.read_csv('InvM_OmegaK.txt', sep='\s+', header=0)
x_data_omegaK = np.array(df['x_val_mean'])
y_data_omegaK = np.array(df['y_val_mean'])
errors_down = np.array([row['y_val_mean'] - row['y_sigma_down'] for _, row in df.iterrows()])
errors_up = np.array([row['y_sigma_up'] - row['y_val_mean'] for _, row in df.iterrows()])
# Use asymmetric errors for robustness
data_errors = np.where(errors_down > 0, 0.5 * (errors_down + errors_up), errors_up)

# Calculate bin edges from bin centers
bin_edges = np.zeros(len(x_data_omegaK) + 1)
bin_edges[1:-1] = (x_data_omegaK[:-1] + x_data_omegaK[1:]) / 2
bin_edges[0] = x_data_omegaK[0] - (bin_edges[1] - x_data_omegaK[0])
bin_edges[-1] = x_data_omegaK[-1] + (x_data_omegaK[-1] - bin_edges[-2])

br_exp_values = np.array([4.1e-4, 1.4e-3, 2.2e-3])  # BR: OmegaK, RhoK, K*Pi
br_exp_errors = np.array([0.9e-4, 0.5e-3, 0.5e-3])

# Integration function
def integrate_dgamma(m, epshR, epshT, Fv, A1, A2, A3, FT1, FT2, FT3, mV, mP, Vckm):
    s = m**2
    return dVP.dGamma(s, Fv=Fv, A1=A1, A2=A2, A3=A3, FT1=FT1, FT2=FT2, FT3=FT3,
                      mV=mV, mP=mP, epshR=epshR, epshT=epshT, Vckm=Vckm) * 2 * m

# Expected counts per bin
def expected_counts_per_bin(epshR, epshT):
    expected = []
    for i in range(len(bin_edges) - 1):
        integral_bin, _ = quad(lambda m: integrate_dgamma(m, epshR, epshT, FF.FVOmegaK, FF.A1OmegaK, FF.A2OmegaK, FF.A3OmegaK,
                                                         FF.FT1OmegaK, FF.FT2OmegaK, FF.FT3OmegaK, cons.mOmega, cons.mK, Vus),
                              bin_edges[i], bin_edges[i+1], epsrel=1e-6, limit=200)
        expected.append(integral_bin)
    expected = np.array(expected)
    
    total_integral, _ = quad(lambda m: integrate_dgamma(m, epshR, epshT, FF.FVOmegaK, FF.A1OmegaK, FF.A2OmegaK, FF.A3OmegaK,
                                                       FF.FT1OmegaK, FF.FT2OmegaK, FF.FT3OmegaK, cons.mOmega, cons.mK, Vus),
                            bin_edges[0], bin_edges[-1], epsrel=1e-6, limit=200)
    if total_integral <= 0:
        return np.zeros_like(expected)
    probabilities = expected / total_integral
    mu = np.sum(y_data_omegaK) * probabilities
    return np.clip(mu, 1e-10, None)  # Avoid log(0) with a small positive threshold

def combined_nll_binned(epshR, epshT):
    mu = expected_counts_per_bin(epshR, epshT)
    n = y_data_omegaK

    #mu = np.clip(mu, 1e-30, None)  # Avoid log(0)
    if np.any(n < 0) or np.any(mu <= 0):
            return np.inf  # Reject invalid fits
    mass_nll = -np.sum(n * np.log(mu / n + 1) - mu + n)  # Poisson NLL with log1p for stabilit

    # Branching ratio chi^2
    br1 = dVP.BRVP(Fv=FF.FVOmegaK, A1=FF.A1OmegaK, A2=FF.A2OmegaK, A3=FF.A3OmegaK,
                   FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                   mV=cons.mOmega, mP=cons.mK, epshR=epshR, epshT=epshT, Vckm=Vus)
    br2 = dVP.BRVP(Fv=FF.FVRhoK, A1=FF.A1RhoK, A2=FF.A2RhoK, A3=FF.A3RhoK,
                   FT1=FF.FT1RhoK, FT2=FF.FT2RhoK, FT3=FF.FT3RhoK,
                   mV=cons.mRho, mP=cons.mK, epshR=epshR, epshT=epshT, Vckm=Vus)
    br3 = dVP.BRVP(Fv=FF.FVKsPi, A1=FF.A1KsPi, A2=FF.A2KsPi, A3=FF.A3KsPi,
                   FT1=FF.FT1KsPi, FT2=FF.FT2KsPi, FT3=FF.FT3KsPi,
                   mV=cons.mKs, mP=cons.mPi, epshR=epshR, epshT=epshT, Vckm=Vus)

    br_chi2 = ((br1 - br_exp_values[0]) / br_exp_errors[0])**2 + \
              ((br2 - br_exp_values[1]) / br_exp_errors[1])**2 + \
              ((br3 - br_exp_values[2]) / br_exp_errors[2])**2

    total_nll = mass_nll + br_chi2

    print(f"mass_nll={mass_nll:.2f}, total_nll={total_nll:.2f}, epshR={epshR:.4f}, epshT={epshT:.4f}")
    print(f"BRs: br1={br1:.2e}, br2={br2:.2e}, br3={br3:.2e}")

    return total_nll

print("\n--- Running iminuit Binned Minimization ---")

m = Minuit(combined_nll_binned, epshR=0.05, epshT=0.01)
m.limits['epshR'] = (-2, 2)
m.limits['epshT'] = (-2, 2)
m.errordef = Minuit.LIKELIHOOD

m.migrad()
m.hesse()

print("\n--- Fit Results ---")
print(m)
for p in m.parameters:
    print(f"{p}: {m.values[p]:.10f} ± {m.errors[p]:.6f}")
print(f"Minimum NLL: {m.fval:.4f}")
print("Correlation matrix:")
print(m.covariance.correlation())

# Check BR contributions with fit results
print("\nBranching Ratio Fit vs. Experimental:")
epshR_fit, epshT_fit = m.values['epshR'], m.values['epshT']
br1_fit = dVP.BRVP(Fv=FF.FVOmegaK, A1=FF.A1OmegaK, A2=FF.A2OmegaK, A3=FF.A3OmegaK,
                   FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
                   mV=cons.mOmega, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
br2_fit = dVP.BRVP(Fv=FF.FVRhoK, A1=FF.A1RhoK, A2=FF.A2RhoK, A3=FF.A3RhoK,
                   FT1=FF.FT1RhoK, FT2=FF.FT2RhoK, FT3=FF.FT3RhoK,
                   mV=cons.mRho, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
br3_fit = dVP.BRVP(Fv=FF.FVKsPi, A1=FF.A1KsPi, A2=FF.A2KsPi, A3=FF.A3KsPi,
                   FT1=FF.FT1KsPi, FT2=FF.FT2KsPi, FT3=FF.FT3KsPi,
                   mV=cons.mKs, mP=cons.mPi, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
print(f"BR1: Fit = {br1_fit:.4e}, Exp = {br_exp_values[0]:.4e} ± {br_exp_errors[0]:.4e}")
print(f"BR2: Fit = {br2_fit:.4e}, Exp = {br_exp_values[1]:.4e} ± {br_exp_errors[1]:.4e}")
print(f"BR3: Fit = {br3_fit:.4e}, Exp = {br_exp_values[2]:.4e} ± {br_exp_errors[2]:.4e}")

epshR_fit, epshT_fit = m.values['epshR'], m.values['epshT']

# Compute expected counts per bin from best fit
expected_fit = expected_counts_per_bin(epshR_fit, epshT_fit)

# --- Plot results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Branching Ratios plot
ax1.errorbar([0, 1, 2], br_exp_values, yerr=br_exp_errors, fmt='bo', label='Experimental', capsize=5)
br_fit = [
    dVP.BRVP(Fv=FF.FVOmegaK, A1=FF.A1OmegaK, A2=FF.A2OmegaK, A3=FF.A3OmegaK,
             FT1=FF.FT1OmegaK, FT2=FF.FT2OmegaK, FT3=FF.FT3OmegaK,
             mV=cons.mOmega, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus),
    dVP.BRVP(Fv=FF.FVRhoK, A1=FF.A1RhoK, A2=FF.A2RhoK, A3=FF.A3RhoK,
             FT1=FF.FT1RhoK, FT2=FF.FT2RhoK, FT3=FF.FT3RhoK,
             mV=cons.mRho, mP=cons.mK, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus),
    dVP.BRVP(Fv=FF.FVKsPi, A1=FF.A1KsPi, A2=FF.A2KsPi, A3=FF.A3KsPi,
             FT1=FF.FT1KsPi, FT2=FF.FT2KsPi, FT3=FF.FT3KsPi,
             mV=cons.mKs, mP=cons.mPi, epshR=epshR_fit, epshT=epshT_fit, Vckm=Vus)
]
ax1.plot([0, 1, 2], br_fit, 'rs', label='Fit')
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels([r"$\tau \to \omega K$", r"$\tau \to \rho K$", r"$\tau \to K^* \pi$"])
ax1.set_ylabel('Branching Ratio')
ax1.set_title('Branching Ratios: Experimental vs. Fit')
ax1.legend()
ax1.grid(True)
ax1.set_ylim(0, max(br_exp_values)*1.5)

# Mass spectrum plot
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths = np.diff(bin_edges)

# Average symmetric error bars for the data
#data_errors = 0.5 * (errors_down + errors_up)

# Plot data points with error bars
ax2.errorbar(bin_centers, y_data_omegaK, yerr=data_errors, fmt='o', color='blue', label='Data', capsize=4)

# Plot data histogram as a step plot
ax2.step(bin_edges, np.append(y_data_omegaK, y_data_omegaK[-1]), where='post', color='blue', alpha=0.3, label='Data histogram')

# Plot fit expectation as step plot
expected_fit = expected_counts_per_bin(epshR_fit, epshT_fit)
# Add goodness-of-fit check
chi2_mass = np.sum(((y_data_omegaK - expected_fit) / data_errors)**2) / (len(y_data_omegaK) - 2)
print(f"\nReduced Chi^2 (mass spectrum): {chi2_mass:.2f}")

ax2.step(bin_edges, np.append(expected_fit, expected_fit[-1]), where='post', color='red', label='Fit')

ax2.set_xlabel('Invariant Mass (GeV)')
ax2.set_ylabel('Events')
ax2.set_title('Invariant Mass Spectrum: Data vs. Fit')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('Minuit_Binned_Fit_Results.png')
plt.show()

print("\nScript finished.")
