"""
YEB.440 Harjoitustyö osa 4: tuulipuiston layoutin optimointi
Amelia Stalter 151193060
Optimizing wind farm wind turbine placement using differential evolution
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
# Other parameters
# --------------------------------------------------

# turbine parameters
Cp = 0.45
D = 120.0            # rotor diameter [m]
R = D / 2
hub_height = 100     # m

# Wake parameters
Ct = 0.8             # thrust coefficient
k = 0.075            # wake decay constant

rho = 1.225          # kg/m³
hours_year = 8760

bounds = [(R, 2000-R)]*10
min_spacing = 4*D


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

def effective_speeds(coords, free_speed, wind_dir_deg):

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

def energy_no_wake(turbines):
    energy_no_wake = 0

    for d in wind_data.values():

        p = turbine_power(d["v"])

        energy_no_wake += (
            len(turbines)
            * p
            * d["freq"]
            * hours_year
        )
    
    return energy_no_wake

# --------------------------------------------------
# Energy with wake
# --------------------------------------------------

def energy_wake(turbines):

    energy_wake = 0

    for name, d in wind_data.items():

        velocities = effective_speeds(
            turbines,
            d["v"],
            d["dir"]
        )

        powers = turbine_power(velocities)

        energy_wake += (
            np.sum(powers)
            * d["freq"]
            * hours_year
        )

    return energy_wake

def spacing_penalty(turbines):

    penalty = 0

    for i in range(len(turbines)):
        for j in range(i+1,len(turbines)):

            dist = np.linalg.norm(
                turbines[i]-turbines[j]
            )

            if dist < 480:
                penalty += 1e12

    return penalty

def objective(vars):

    turbines_test = np.array([
        [vars[0], vars[1]],
        [vars[2], vars[3]],
        [vars[4], vars[5]],
        [vars[6], vars[7]],
        [vars[8], vars[9]]
    ])

    penalty = spacing_penalty(turbines_test)

    return -energy_wake(turbines_test) + penalty

# --------------------------------------------------
# Run optimization
# --------------------------------------------------

result = differential_evolution(
    objective,
    bounds,
    maxiter=500,
    popsize=20,
    seed=42
)

# --------------------------------------------------
# Results
# --------------------------------------------------

coords = result.x.reshape((5,2))

print(coords)

optimized_energy = energy_wake(coords)
base_energy = energy_no_wake(coords)

loss_percent = (
    (base_energy - optimized_energy)
    / base_energy
    * 100
)

print("\nRESULTS")
print("-" * 40)

print(
    f"Annual production without wake:"
    f" {base_energy/1e6:.2f} MWh"
)

print(
    f"Annual production with wake:"
    f" {optimized_energy/1e6:.2f} MWh"
)

print(
    f"Wake losses:"
    f" {loss_percent:.2f} %"
)

# --------------------------------------------------
# Production per turbine
# --------------------------------------------------

print("\nProduction per turbine:")

for idx in range(len(coords)):

    annual = 0

    for name, d in wind_data.items():


        v_eff = effective_speeds(
            coords,
            d["v"],
            d["dir"]
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
# Plot the new [optimized] coordinates
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

labels = ["T1", "T2", "T3", "T4", "T5"]

for (x, y), label in zip(coords, labels):
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

