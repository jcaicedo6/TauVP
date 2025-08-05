import numpy as np
from iminuit import Minuit
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Define experimental data ---
# Experimental Branching Ratios
br_exp_values = np.array([0.12, 0.06])
br_exp_errors = np.array([0.01, 0.005])

# Mock invariant mass data
num_mass_events = 10000
true_epsR = 0.5
true_epsT = 0.1
true_epshT = 0.2
true_epshR = 0.3
mock_mass_data = []
for _ in range(num_mass_events):
    if np.random.rand() < (0.8 * (1 + 0.1 * true_epshR + 0.2 * true_epshT)):
        mock_mass_data.append(np.random.normal(5.28, 0.01))
    else:
        mock_mass_data.append(np.random.uniform(5.0, 5.5))
mass_data = np.array(mock_mass_data)

# --- 2. Define model functions ---
# Fixed parameters
mean = 5.28  # Signal mean
sigma = 0.01  # Signal width
mass_min, mass_max = 5.0, 5.5  # Mass range for uniform background
N_total = num_mass_events  # Total number of mass events

# Signal fraction for mass PDF
def signal_fraction(epshR, epshT):
    return 0.8 * (1 + 0.1 * epshR + 0.2 * epshT)

# Mass PDF: Gaussian signal + uniform background
def mass_pdf(x, epshR, epshT):
    sig_frac = signal_fraction(epshR, epshT)
    sig_pdf = norm.pdf(x, mean, sigma)
    bkg_pdf = uniform.pdf(x, mass_min, mass_max - mass_min)
    return sig_frac * sig_pdf + (1 - sig_frac) * bkg_pdf

# Theoretical branching ratios
def br1_theory(epsR, epsT):
    return 0.1 * (1 + epsR + epsT**2)

def br2_theory(epsR, epsT):
    return 0.05 * (1 + epsR**2 + epsT**2)

# Negative log-likelihood for mass data
def mass_nll(epshR, epshT):
    pdf_vals = mass_pdf(mass_data, epshR, epshT)
    # Avoid log(0) by clipping very small values
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# Chi-squared for branching ratios
def br_chi2(epsR, epsT):
    br1_fit = br1_theory(epsR, epsT)
    br2_fit = br2_theory(epsR, epsT)
    chi2 = ((br1_fit - br_exp_values[0]) / br_exp_errors[0])**2 + \
           ((br2_fit - br_exp_values[1]) / br_exp_errors[1])**2
    return chi2

# Combined negative log-likelihood
def combined_nll(epsR, epsT, epshR, epshT):
    return mass_nll(epshR, epshT) + br_chi2(epsR, epsT)

# --- 3. Perform the fit ---
# Initialize Minuit
m = Minuit(combined_nll, epsR=0.5, epsT=0.1, epshR=0.3, epshT=0.2)
m.limits['epsR'] = (-1.0, 1.0)
m.limits['epsT'] = (-1.0, 1.0)
m.limits['epshR'] = (-1.0, 1.0)
m.limits['epshT'] = (-1.0, 1.0)

# Run minimization
m.migrad()  # Run the fit
m.hesse()   # Compute errors

# --- 4. Extract and print results ---
print("\n--- Fit Results ---")
print(f"Converged: {m.valid}")
print(f"Minimum NLL: {m.fval:.4f}")
print("\nFit Parameters:")
for param in m.parameters:
    value, error = m.values[param], m.errors[param]
    print(f"{param}: {value:.4f} +/- {error:.4f}")

# Correlation matrix
print("\nCorrelation Matrix:")
corr = m.covariance.correlation()
param_names = m.parameters
for i in range(len(param_names)):
    row = [corr[i, j] for j in range(len(param_names))]
    print(f"{param_names[i]}: {row}")

# Check branching ratio contributions
print("\nBranching Ratio Fit vs. Experimental:")
print(f"BR1: Fit = {br1_theory(m.values['epsR'], m.values['epsT']):.4f}, Exp = {br_exp_values[0]:.4f} ± {br_exp_errors[0]:.4f}")
print(f"BR2: Fit = {br2_theory(m.values['epsR'], m.values['epsT']):.4f}, Exp = {br_exp_values[1]:.4f} ± {br_exp_errors[1]:.4f}")

# --- 5. Visualization --

# Plot branching ratios
# --- 5. Visualization and Goodness of Fit ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot branching ratios
x = np.array([0, 1])
ax1.errorbar(x, br_exp_values, yerr=br_exp_errors, fmt='bo', label='Experimental', capsize=5)
br_fit = [br1_theory(m.values['epsR'], m.values['epsT']), 
          br2_theory(m.values['epsR'], m.values['epsT'])]
ax1.plot(x, br_fit, 'rs', label='Fit')
ax1.set_xticks(x)
ax1.set_xticklabels(['Decay 1 BR', 'Decay 2 BR'])
ax1.set_ylabel('Branching Ratio')
ax1.set_title('Branching Ratios: Experimental vs. Fit')
ax1.legend()
ax1.set_ylim(0, max(br_exp_values) * 1.5)

# Plot mass spectrum and compute chi-squared
bins = 100
hist, bin_edges = np.histogram(mass_data, bins=bins, range=(mass_min, mass_max))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]
hist_density = hist / (num_mass_events * bin_width)  # Normalize to density
hist_errors = np.sqrt(hist) / (num_mass_events * bin_width)  # Poisson errors
ax2.hist(mass_data, bins=bins, range=(mass_min, mass_max), density=True, alpha=0.5, label='Data')
x_mass = np.linspace(mass_min, mass_max, 1000)
pdf_total = mass_pdf(x_mass, m.values['epshR'], m.values['epshT'])
sig_frac = signal_fraction(m.values['epshR'], m.values['epshT'])
pdf_sig = norm.pdf(x_mass, mean, sigma) * sig_frac
pdf_bkg = uniform.pdf(x_mass, mass_min, mass_max - mass_min) * (1 - sig_frac)
ax2.plot(x_mass, pdf_total, 'r-', label='Total Fit')
ax2.plot(x_mass, pdf_sig, 'g--', label='Signal')
ax2.plot(x_mass, pdf_bkg, 'y--', label='Background')
ax2.set_xlabel('Invariant Mass')
ax2.set_ylabel('Density')
ax2.set_title('Invariant Mass Spectrum: Data vs. Fit')
ax2.legend()

# Compute chi-squared
pdf_binned = mass_pdf(bin_centers, m.values['epshR'], m.values['epshT']) * num_mass_events * bin_width
chi2 = np.sum(((hist - pdf_binned)**2 / (hist + 1e-10)) / (num_mass_events * bin_width)**2)
dof = bins - 4  # Number of bins minus number of free parameters
print(f"\nChi-squared / dof: {chi2:.5f} / {dof} = {chi2/dof:.5f}")

plt.tight_layout()
plt.savefig('Minuit_Combined_Fit_Results.png')
plt.show()

print("\nScript finished.")