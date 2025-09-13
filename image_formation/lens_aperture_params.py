
import numpy as np
import matplotlib.pyplot as plt


# Question 1 (Thin lens plot)
FOCAL_LENGTHS_MM = [3, 9, 50, 200]  # f in mm (ultra-wide, telephoto phone, 50mm, 200mm)
Z0_MIN_FACTOR = 1.1                  # lower bound for z0 is 1.1 * f
Z0_MAX_MM = 1e4                      # upper bound for z0 in mm
POINTS_PER_MM = 4                    # sampling density for z0
Y_LIM_ZI_MM = (0, 3000)              # y-limits for zi plot

# Question 2 (Aperture plots)
F_NUMBERS = [1.4, 1.8, 2.8, 4.0]     # popular f-numbers to plot
F_RANGE_MIN_MM = 10                  # min focal length for D vs f plot
F_RANGE_MAX_MM = 600                 # max focal length for D vs f plot
F_RANGE_POINTS = 350                 # for plotting; generate 350 evenlty spaced points in [F_RANGE_MIN_MM, F_RANGE_MAX_MM]

REAL_LENSES = [
    ("24mm f/1.4", [(24, 1.4)]),
    ("50mm f/1.8", [(50, 1.8)]),
    ("70–200mm f/2.8", [(70, 2.8), (200, 2.8)]),
    ("400mm f/2.8", [(400, 2.8)]),
    ("600mm f/4.0", [(600, 4.0)]),
]


def thin_lens_zi(f, z0):
    """
    Thin lens law:
        1/f = 1/z0 + 1/zi  ->  zi = 1 / (1/f - 1/z0)
    """
    return 1.0 / (1.0 / f - 1.0 / z0)


def make_z0_grid(f):
    """
    Build x-axis values for thin lens plot.
    Build the z0 grid in [1.1*f, Z0_MAX_MM] with ~POINTS_PER_MM points per mm.
    """
    z0_min = Z0_MIN_FACTOR * f # To add space on the left of the plot.
    # z0_min = Z0_MIN_FACTOR
    n_pts = int(max(10, (Z0_MAX_MM - z0_min) * POINTS_PER_MM))
    return np.linspace(z0_min, Z0_MAX_MM, n_pts)


def aperture_diameter(f, f_number):
    """
    f-number N = f / D  ->  D = f / N
    Returns D in millimeters.
    """
    return f / f_number


# Plotting question 1
def plot_zi_vs_z0():
    """
    Plot image distance zi as a function of object distance z0
    for each focal length in FOCAL_LENGTHS_MM, using log-log axes.
    loglog makes both axes logarithmic so we can see wide ranges more clearly.
    """
    plt.figure(figsize=(7.2, 5.2))

    for f in FOCAL_LENGTHS_MM:
        z0 = make_z0_grid(f)
        zi = thin_lens_zi(f, z0)

        # Curve (log-log). Capture line to reuse its color.
        line, = plt.loglog(z0, zi, label=f"f = {f:g} mm") # :g removes trailing zeros

        # Vertical dashed line at z0 = f, same color as curve
        plt.axvline(f, linestyle="--", color=line.get_color())

    # Aesthetics / requirements
    plt.ylim(*Y_LIM_ZI_MM)
    plt.xlabel("Object distance $z_0$ (mm)")
    plt.ylabel("Image distance $z_i$ (mm)")
    plt.title("Thin Lens: $z_i$ vs $z_0$ for multiple focal lengths")
    plt.grid(True, which="both", linestyle=":", axis="both")
    plt.legend()
    plt.tight_layout()


def compute_D_values(f_vals, f_numbers):
    """
    Given an array of focal lengths and a list of f-numbers,
    return a dict mapping f-number -> array of D values.
    """
    results = {}
    for N in f_numbers:
        results[N] = f_vals / N
    return results

def plot_D_vs_f():
    """
    Plot aperture diameter D as a function of focal length f for each f-number in F_NUMBERS.
    """
    f_vals = np.linspace(F_RANGE_MIN_MM, F_RANGE_MAX_MM, F_RANGE_POINTS)
    plt.figure(figsize=(7.2, 5.2))

    for N in F_NUMBERS:
        D_vals = f_vals / N
        plt.plot(f_vals, D_vals, label=f"f/{N:g}")

    plt.xlabel("Focal length f (mm)")
    plt.ylabel("Aperture diameter D (mm)")
    plt.title("Aperture Diameter vs Focal Length for Popular f-numbers")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()

def print_real_lens_apertures():
    print("\n Maximum aperture diameters (D = f / N)")
    for name, specs in REAL_LENSES:
        if len(specs) == 1: # single focal length
            f_mm, N = specs[0]
            D = aperture_diameter(f_mm, N)
            print(f"  {name}: D ≈ {D:.2f} mm")
        else:
            parts = []
            for f_mm, N in specs:   # multiple focal lengths
                D = aperture_diameter(f_mm, N)
                parts.append(f"{f_mm}mm → {D:.2f} mm")
            print(f"  {name}: " + "; ".join(parts))

def main():
    # Part 1: Thin lens plot (zi vs z0)
    plot_zi_vs_z0()

    # Part 2: Aperture diameter plots
    plot_D_vs_f()
    print_real_lens_apertures()

    # Show figures
    plt.show()


if __name__ == "__main__":
    main()
