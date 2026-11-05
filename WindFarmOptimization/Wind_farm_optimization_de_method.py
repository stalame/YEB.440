"""
YEB.440 Harjoitustyö osa 4: tuulipuiston layoutin optimointi
Amelia Stalter 151193060
Optimizing wind farm wind turbine placement using genetic algorithm
"""
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Wind rose data
# --------------------------------------------------

wind_data = {
    "N":  {"freq": 0.20, "v": 5.1, "dir": 0},
    "NE": {"freq": 0.05, "v": 7.7, "dir": 45},
    "E":  {"freq": 0.06, "v": 5.4, "dir": 90},
    "SE": {"freq": 0.17, "v": 6.3, "dir": 135},
    "S":  {"freq": 0.10, "v": 9.8, "dir": 180},
    "SW": {"freq": 0.11, "v": 7.7, "dir": 225},
    "W":  {"freq": 0.10, "v": 9.4, "dir": 270},
    "NW": {"freq": 0.21, "v": 6.7, "dir": 315},
}

# --------------------------------------------------
# Turbine locations from Figure 4
# --------------------------------------------------

turbines = np.array([
    [1700 , 1900],  # T1
    [700, 1600],    # T2
    [1200, 1000],    # T3
    [400, 600],    # T4
    [1600, 400]     # T5
])

# --------------------------------------------------
# Turbine parameters
# --------------------------------------------------

rho = 1.225          # kg/m³
Cp = 0.45
D = 120.0            # rotor diameter [m]
R = D / 2
hub_height = 100     # m

# Wake parameters
Ct = 0.8             # thrust coefficient
k = 0.075            # wake decay constant

hours_year = 8760

# --------------------------------------------------
# Power calculation
# --------------------------------------------------

A = np.pi * R**2

def turbine_power(v):
    """
    Available power limited only by Cp.
    """
    return 0.5 * rho * A * Cp * v**3


# --------------------------------------------------
# Coordinate transformation
# --------------------------------------------------

def rotate_to_wind(coords, wind_dir_deg):
    """
    Transform coordinates so x-axis points downstream.
    """
    theta = np.radians(wind_dir_deg)

    rot = np.array([
        [np.sin(theta),  np.cos(theta)],
        [-np.cos(theta), np.sin(theta)]
    ])

    return coords @ rot.T


# --------------------------------------------------
# Jensen wake model
# --------------------------------------------------

def effective_speeds(coords, free_speed):

    rotated = rotate_to_wind(coords, wind_dir_deg)

    n = len(coords)
    deficits = np.zeros(n)

    order = np.argsort(rotated[:, 0])

    for jj in range(1, n):

        j = order[jj]

        total_deficit_sq = 0

        for ii in range(jj):

            i = order[ii]

            dx = rotated[j, 0] - rotated[i, 0]

            if dx <= 0:
                continue

            dy = abs(rotated[j, 1] - rotated[i, 1])

            wake_radius = R + k * dx

            if dy <= wake_radius:

                deficit = (
                    (1 - np.sqrt(1 - Ct))
                    * (R / wake_radius) ** 2
                )

                total_deficit_sq += deficit**2

        deficits[j] = np.sqrt(total_deficit_sq)

    return free_speed * (1 - deficits)


# --------------------------------------------------
# Energy without wake
# --------------------------------------------------

energy_no_wake = 0

for d in wind_data.values():

    p = turbine_power(d["v"])

    energy_no_wake += (
        len(turbines)
        * p
        * d["freq"]
        * hours_year
    )

# --------------------------------------------------
# Energy with wake
# --------------------------------------------------

energy_wake = 0

for name, d in wind_data.items():

    wind_dir_deg = d["dir"]

    velocities = effective_speeds(
        turbines,
        d["v"]
    )

    powers = turbine_power(velocities)

    energy_wake += (
        np.sum(powers)
        * d["freq"]
        * hours_year
    )

# --------------------------------------------------
# Results
# --------------------------------------------------

loss_percent = (
    (energy_no_wake - energy_wake)
    / energy_no_wake
    * 100
)

print("\nRESULTS")
print("-" * 40)

print(
    f"Annual production without wake:"
    f" {energy_no_wake/1e6:.2f} MWh"
)

print(
    f"Annual production with wake:"
    f" {energy_wake/1e6:.2f} MWh"
)

print(
    f"Wake losses:"
    f" {loss_percent:.2f} %"
)

# --------------------------------------------------
# Production per turbine
# --------------------------------------------------

print("\nProduction per turbine:")

for idx in range(len(turbines)):

    annual = 0

    for name, d in wind_data.items():

        wind_dir_deg = d["dir"]

        v_eff = effective_speeds(
            turbines,
            d["v"]
        )[idx]

        annual += (
            turbine_power(v_eff)
            * d["freq"]
            * hours_year
        )

    print(
        f"Turbine {idx+1}: "
        f"{annual/1e6:.2f} MWh/year"
    )

# --------------------------------------------------
# Plot the new coordinates
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

labels = ["T1", "T2", "T3", "T4", "T5"]

for (x, y), label in zip(turbines, labels):
    ax.scatter(x, y, s=120, marker='o')

    ax.text(
        x + 40,
        y + 40,
        f"{label}\n({x},{y})",
        fontsize=10
    )

# Plot boundary of wind farm
ax.plot(
    [0, 2000, 2000, 0, 0],
    [0, 0, 2000, 2000, 0],
    'k-'
)

ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

ax.set_title("Wind Farm Layout")

ax.grid(True)
ax.set_aspect("equal")

plt.tight_layout()
plt.show()

