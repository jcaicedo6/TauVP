import ROOT
import numpy as np
from array import array # Used for ROOT C-style arrays

# Suppress ROOT's default info messages
# ROOT.gROOT.SetBatch(True) # Set to False if you want interactive canvases to pop up and pause script
ROOT.gROOT.SetBatch(False) # Keep it False for interactive plotting like matplotlib
# ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;") # Suppress some ROOT messages (optional, keep it on for debugging)

# --- 1. Define your model functions (these will now be translated to RooFit objects) ---
# Note: In RooFit, you define these as RooFormulaVar or combine basic PDFs.

# --- 2. Define your experimental data ---
# Experimental Branching Ratios
br_exp_values = np.array([0.12, 0.06])
br_exp_errors = np.array([0.01, 0.005])

# Invariant Mass Data (example: randomly generated, in reality from your experiment)
np.random.seed(42)
num_mass_events = 10000 # Updated to 10000 as per your last code snippet
true_epsR = 0.5
true_epsT = 0.1
true_epshT = 0.2
true_epshR = 0.3
mock_mass_data = []
for _ in range(num_mass_events):
    # This simulation code is still in numpy, as it's just for generating mock data.
    # The actual PDF evaluation will be done by RooFit's components.
    if np.random.rand() < (0.8 * (1 + 0.1 * true_epshR + 0.2 * true_epshT)):
        mock_mass_data.append(np.random.normal(5.28, 0.01))
    else:
        mock_mass_data.append(np.random.uniform(5.0, 5.5))
mass_data = np.array(mock_mass_data)

# --- 3. Define the Model in RooFit ---

# Parameters (THESE WILL REMAIN FREE - EPSILONS)
epsR = ROOT.RooRealVar("epsR", "epsilon R", 0.5, -1.0, 1.0)
epsT = ROOT.RooRealVar("epsT", "epsilon T", 0.1, -1.0, 1.0)
epshT = ROOT.RooRealVar("epshT", "epsilon hT", 0.2, -1.0, 1.0)
epshR = ROOT.RooRealVar("epshR", "epsilon hR", 0.3, -1.0, 1.0)

# Observables
mass = ROOT.RooRealVar("mass", "Invariant Mass", 5.0, 5.5)

# --- Mass Spectrum Model ---
# FIXED: Mean and Sigma of signal
mean = ROOT.RooRealVar("mean", "Mean of signal", 5.28, 5.2, 5.3)
mean.setConstant(True) # <<<--- SET TO CONSTANT
sigma = ROOT.RooRealVar("sigma", "Width of signal", 0.01, 0.001, 0.1)
sigma.setConstant(True) # <<<--- SET TO CONSTANT

# Signal PDF (Gaussian)
signal_pdf = ROOT.RooGaussian("signal_pdf", "Signal PDF", mass, mean, sigma)

# Background PDF (Uniform)
background_pdf = ROOT.RooUniform("background_pdf", "Background PDF", mass)

# Total number of mass events (fixed, as this is the dataset size)
N_total_mass_events = ROOT.RooRealVar("N_total_mass_events", "Total mass events", num_mass_events)
N_total_mass_events.setConstant(True) # This was already correct

# Signal fraction for mass PDF (derived from epshR, epshT)
raw_signal_fraction_formula = "0.8 * (1 + 0.1 * @0 + 0.2 * @1)"
signal_fraction_mass_pdf = ROOT.RooFormulaVar("signal_fraction_mass_pdf", raw_signal_fraction_formula, ROOT.RooArgList(epshR, epshT))

# Now, define the yields as RooFormulaVar objects based on the total events and the signal_fraction
n_signal_mass = ROOT.RooFormulaVar("n_signal_mass", "@0 * @1", ROOT.RooArgList(N_total_mass_events, signal_fraction_mass_pdf))
n_background_mass = ROOT.RooFormulaVar("n_background_mass", "@0 * (1 - @1)", ROOT.RooArgList(N_total_mass_events, signal_fraction_mass_pdf))

# Combined PDF for mass spectrum
mass_pdf_model = ROOT.RooAddPdf("mass_pdf_model", "Mass PDF Model",
                                ROOT.RooArgList(signal_pdf, background_pdf),
                                ROOT.RooArgList(n_signal_mass, n_background_mass))


# --- Branching Ratio Model (Using RooGaussian constraints) ---
# FIXED: Experimental BRs as RooRealVar (constant, observed values)
br1_obs = ROOT.RooRealVar("br1_obs", "Observed BR1", br_exp_values[0])
br1_obs.setConstant(True) # <<<--- SET TO CONSTANT
br2_obs = ROOT.RooRealVar("br2_obs", "Observed BR2", br_exp_values[1])
br2_obs.setConstant(True) # <<<--- SET TO CONSTANT

# FIXED: Errors for BRs (constant, observed errors) - RooRealVar for RooGaussian sigma
br1_err = ROOT.RooRealVar("br1_err", "Error BR1", br_exp_errors[0], 0.0001, 1.0)
br1_err.setConstant(True) # <<<--- SET TO CONSTANT
br2_err = ROOT.RooRealVar("br2_err", "Error BR2", br_exp_errors[1], 0.0001, 1.0)
br2_err.setConstant(True) # <<<--- SET TO CONSTANT

# Theoretical BRs as RooFormulaVar (depend on epsR, epsT) - these are fine
br1_theory = ROOT.RooFormulaVar("br1_theory", "0.1 * (1 + @0 + @1*@1)", ROOT.RooArgList(epsR, epsT))
br2_theory = ROOT.RooFormulaVar("br2_theory", "0.05 * (1 + @0*@0 + @1*@1)", ROOT.RooArgList(epsR, epsT))

# Gaussian constraints for BRs - these are fine
constraint_br1 = ROOT.RooGaussian("constraint_br1", "BR1 Constraint", br1_obs, br1_theory, br1_err)
constraint_br2 = ROOT.RooGaussian("constraint_br2", "BR2 Constraint", br2_obs, br2_theory, br2_err)

# Create a list of external constraints - this is fine
external_constraints = ROOT.RooArgSet(constraint_br1, constraint_br2)


# --- Create RooFit Datasets ---

# Mass Data: Unbinned
mass_data_set = ROOT.RooDataSet("mass_data_set", "Mass Data", ROOT.RooArgSet(mass))
for val in mass_data:
    mass.setVal(val)
    mass_data_set.add(ROOT.RooArgSet(mass))

# --- Perform the Fit ---

print("\n--- Running RooFit Minimization ---")
fit_result = mass_pdf_model.fitTo(mass_data_set,
                                  ROOT.RooFit.Extended(True),
                                  ROOT.RooFit.Save(),
                                  ROOT.RooFit.PrintLevel(1), # Set verbosity for the minimizer
                                  ROOT.RooFit.SumW2Error(False),
                                  ROOT.RooFit.Hesse(True),
                                  ROOT.RooFit.Minimizer("Minuit2", "Migrad"),
                                  ROOT.RooFit.Strategy(2),
                                  ROOT.RooFit.ExternalConstraints(external_constraints)
                                 )

# --- 5. Extract and Print Results ---
print("\n--- RooFit Fit Results ---")
fit_result.Print() # Prints a summary of the fit result

# Extract fitted values and errors
print("\nFit Parameters:")
for param in fit_result.floatParsFinal():
    print(f"{param.GetName()}: {param.getVal():.4f} +/- {param.getError():.4f}")

print(f"\nMinimum NLL (from fit result): {fit_result.minNll():.4f}")

# Print Correlation Matrix
print("\nCorrelation Matrix from RooFit:")
fit_result.correlationMatrix().Print()


# --- 6. Goodness of Fit and Visualization with ROOT ---

# Create a canvas
c = ROOT.TCanvas("fit_canvas", "Combined Fit Results", 1200, 600)
c.Divide(2, 1)

# --- Plotting BRs ---
c.cd(1)
br_x = array('d', [0, 1])
br_names_str = ["Decay 1 BR", "Decay 2 BR"]
br_exp_val_arr = array('d', br_exp_values)
br_exp_err_arr = array('d', br_exp_errors)
br_x_err = array('d', [0, 0])

g_br_exp = ROOT.TGraphErrors(len(br_x), br_x, br_exp_val_arr, br_x_err, br_exp_err_arr)
g_br_exp.SetTitle("Branching Ratios: Experimental vs. Fit;Decay;Branching Ratio")
g_br_exp.SetMarkerStyle(20)
g_br_exp.SetMarkerSize(1.2)
g_br_exp.SetLineColor(ROOT.kBlue)
g_br_exp.SetMinimum(0)
g_br_exp.SetMaximum(max(br_exp_values) * 1.5)
g_br_exp.Draw("AP")

# Plot fitted BRs (using fitted parameters)
br1_fit_val = br1_theory.getVal()
br2_fit_val = br2_theory.getVal()

g_br_fit = ROOT.TGraph(len(br_x))
g_br_fit.SetPoint(0, 0, br1_fit_val)
g_br_fit.SetPoint(1, 1, br2_fit_val)
g_br_fit.SetMarkerStyle(21)
g_br_fit.SetMarkerColor(ROOT.kRed)
g_br_fit.SetLineColor(ROOT.kRed)
g_br_fit.Draw("P SAME")

axis = g_br_exp.GetXaxis()
axis.SetBinLabel(axis.FindBin(0), br_names_str[0])
axis.SetBinLabel(axis.FindBin(1), br_names_str[1])
axis.SetNdivisions(2)

legend_br = ROOT.TLegend(0.6, 0.7, 0.9, 0.85)
legend_br.AddEntry(g_br_exp, "Experimental", "lep")
legend_br.AddEntry(g_br_fit, "Fit", "p")
legend_br.Draw()

# --- Plotting Invariant Mass Spectrum ---
c.cd(2)

mass_frame = mass.frame(ROOT.RooFit.Title("Invariant Mass Spectrum: Data vs. Fit"))

mass_data_set.plotOn(mass_frame)

mass_pdf_model.plotOn(mass_frame)
mass_pdf_model.plotOn(mass_frame, ROOT.RooFit.Components("signal_pdf"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen+2))
mass_pdf_model.plotOn(mass_frame, ROOT.RooFit.Components("background_pdf"), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kOrange+2))

mass_frame.Draw()

c.Update()
c.SaveAs("RooFit_Combined_Fit_Results.png")

# Keep ROOT objects alive (important for interactive sessions or if you don't save immediately)
_keep_objects = [c, g_br_exp, g_br_fit, legend_br, mass_frame, mass_data_set, fit_result]
_keep_params = [epsR, epsT, epshR, epshT, mean, sigma, N_total_mass_events,
                signal_fraction_mass_pdf, n_signal_mass, n_background_mass,
                br1_obs, br2_obs, br1_err, br2_err, br1_theory, br2_theory,
                constraint_br1, constraint_br2, mass_pdf_model]


print("\nScript finished. If not in batch mode, close canvas to exit.")

if not ROOT.gROOT.IsBatch():
    input("Press Enter to exit...")