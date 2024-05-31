import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def compute_binodal(yvec, N):
    # Function that takes in the vector of y values and the value of N then outputs the the binodal curve in the form of
    # phi_right_vec, phi_left_vec, and chi_vec

    hyvec = 1.0 / yvec * np.arctanh(yvec)
    ytrial = np.logspace(np.log10(min(yvec)), np.log10(max(yvec)), 2000)
    ftrial = 1.0 / ytrial * np.arctanh(ytrial)

    # Create interpolation function
    hinv = interp1d(ftrial, ytrial, kind='linear', bounds_error=False)

    zvec = hinv(hyvec / N - (1 / N - 1))
    print(zvec)
    Bvec = 2 * zvec / (yvec + zvec)
    chi_vec = ((1 / N - 1) * Bvec * yvec - np.log((1 - zvec) / (1 + zvec))) / (Bvec ** 2 * yvec)
    phi_right_vec = zvec * (1 + yvec) / (zvec + yvec)
    phi_left_vec = zvec * (1 - yvec) / (zvec + yvec)

    return chi_vec, phi_right_vec, phi_left_vec

def approx_binodal(N):
    # Function that takes in the value of N then outputs the the approximate, asymptotic binodal curve in the form of
    # phi_right_app, phi_left_app, and chi_app
    phic = 1 / (1 + np.sqrt(N))
    chic = 0.5 * (1 + 1 / np.sqrt(N)) ** 2
    betac = 2 * phic

    phi_right_app = 1 - np.logspace(-12, np.log10(1 - betac), 10000)
    chi_app = ((1 / N - 1) * phi_right_app - np.log(1 - phi_right_app)) / phi_right_app ** 2
    phi_left_app = phi_right_app / (1 - phi_right_app) ** N * np.exp(-2 * N * chi_app * phi_right_app)

    return chi_app, phi_right_app, phi_left_app



#Run and plot the function outputs
N = 100  # Define N

# Define yvec so that it samples values close to 1
yvec = np.concatenate([np.logspace(-3, np.log10(0.1), 5000),
                       np.linspace(0.1001, 0.8999, 5000),
                       np.flip(1 - np.logspace(-20, np.log10(0.1), 5000))])

# Call the functions
chi_vec, phi_right_vec, phi_left_vec = compute_binodal(yvec, N)

# Find asymptotic part of curve:
chi_app, phi_right_app, phi_left_app = approx_binodal(N)

# Plot both implicit result and asymptotic formula
plt.figure(figsize=(10, 5))
plt.plot(phi_right_vec, chi_vec, color='blue')
plt.plot(phi_left_vec, chi_vec, color='blue' )
plt.plot(phi_right_app, chi_app, color='red', linestyle='--')
plt.plot(phi_left_app, chi_app, color='red', linestyle='--' )
plt.xscale('log')
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\chi$')
plt.xlim(1e-15, 1)
plt.ylim(0.5, 1.5)
string_N = f"N = {N}"
plt.title(string_N)
plt.legend()
plt.grid(True)
plt.show()
