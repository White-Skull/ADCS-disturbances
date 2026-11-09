
"""
Mercury polar-orbit spacecraft attitude simulation
===================================================

Simulates the spacecraft's attitude deviation from nominal LVLH attitude
over one complete orbit.

Alternatively, with LIMIT_CHECK_MODE enabled, the simulation runs until the
attitude deviation AND the angular velocity both exceed their allowed
limits, continues for a short overrun time, and then stops.

Assumptions
-----------
1. Circular polar orbit around Mercury.
2. Attitude motion is restricted to ONE rotational axis / plane.
3. Small-angle approximation is used.
4. Spacecraft is a rigid body with a specified moment of inertia.
5. Disturbances are represented as torques.
6. Disturbance torques can be constant, sinusoidal, or arbitrary functions.
7. The nominal attitude is LVLH; theta = 0 means perfect LVLH pointing.
8. Gravity-gradient torque is modeled approximately as a restoring torque.
9. No active attitude-control system is included in this basic model.

Outputs
-------
- Attitude deviation from LVLH [deg]
- Angular acceleration [deg/s^2]
- Individual disturbance torques [N m]
- Angular velocity [deg/s]

Dependencies
------------
numpy
scipy
matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.integrate import solve_ivp


# =============================================================================
# 1. MERCURY / ORBIT PARAMETERS
# =============================================================================

# Mercury gravitational parameter
MU_MERCURY = 2.2032e13          # m^3/s^2

# Mercury mean radius
R_MERCURY = 2.4397e6            # m

# Orbit altitude
ORBIT_ALTITUDE = 750e3          # m

# Orbit radius
ORBIT_RADIUS = R_MERCURY + ORBIT_ALTITUDE

# Spacecraft orbital period
ORBIT_PERIOD = 2 * np.pi * np.sqrt(
    ORBIT_RADIUS**3 / MU_MERCURY
)

# Fraction of the orbit spent in eclipse
ECLIPSE_FRACTION = 0.277

print(f"Orbit radius  : {ORBIT_RADIUS / 1e3:.1f} km")
print(f"Orbit period  : {ORBIT_PERIOD / 60:.2f} min")
print(f"Orbit period  : {ORBIT_PERIOD:.2f} s")


# =============================================================================
# 2. SPACECRAFT PARAMETERS
# =============================================================================

# Spacecraft mass
SPACECRAFT_MASS = 936.6         # kg

# Moment of inertia about the rotational axis being modeled
I_SPACECRAFT = 560.1            # kg m^2


# =============================================================================
# 3. INITIAL ATTITUDE CONDITIONS
# =============================================================================

# Initial attitude error relative to LVLH
INITIAL_ANGLE_DEG = 0.0         # degrees

# Initial angular velocity relative to LVLH
INITIAL_RATE_DEG_S = 0.0        # degrees/s


# =============================================================================
# 4. DISTURBANCE TORQUES
# =============================================================================
#
# Each disturbance can be:
#
#   constant:
#       torque = CONSTANT
#
#   sinusoidal:
#       torque = AMPLITUDE * sin(2*pi*t/PERIOD + PHASE)
#
# The sign convention is:
#
#   positive torque -> positive attitude rotation
#
# All disturbances act about the same rotational axis.
# =============================================================================


# -----------------------------------------------------------------------------
# Solar radiation pressure torque
# -----------------------------------------------------------------------------

SRP_ENABLED = True

SRP_TORQUE = 2.06e-5             # N m

# Make the solar radiation torque vary sinusoidally over the orbit.
SRP_CYCLICAL = False

# Number of cycles during one orbit
SRP_CYCLES_PER_ORBIT = 1.0

# Phase [rad]
SRP_PHASE = 0.0


# -----------------------------------------------------------------------------
# Mercury thermal radiation torque
# -----------------------------------------------------------------------------

THERMAL_ENABLED = True

THERMAL_TORQUE = 1.0e-5         # N m

THERMAL_CYCLICAL = False

THERMAL_CYCLES_PER_ORBIT = 2.0

THERMAL_PHASE = np.pi / 4


# -----------------------------------------------------------------------------
# Gravity-gradient torque
# -----------------------------------------------------------------------------
#
# For a simplified single-axis model:
#
#     T_gg = (3*mu/r^3) * (Iz - Ix) * sin(theta)*cos(theta)
#
# For small theta:
#
#     T_gg ≈ K_gg * theta
#
# The exact sign depends on the selected rotational axis and inertia
# convention. The implementation below uses a restoring sign.
# -----------------------------------------------------------------------------

GRAVITY_GRADIENT_ENABLED = True

# Principal moments of inertia.
# Change these to represent the spacecraft geometry.
I_X = 1090.5                    # kg m^2
I_Y = 560.1                     # kg m^2
I_Z = 899.8                     # kg m^2


# -----------------------------------------------------------------------------
# Magnetic torque
# -----------------------------------------------------------------------------

MAGNETIC_ENABLED = False

MAGNETIC_TORQUE = 5.0e-6          # N m

# Number of magnetic torque cycles per orbit
MAGNETIC_CYCLES_PER_ORBIT = 1.0

# Phase [rad]
MAGNETIC_PHASE = 0.0

# -----------------------------------------------------------------------------
# Optional constant bias torque
# -----------------------------------------------------------------------------

BIAS_ENABLED = False

BIAS_TORQUE = 0.0               # N m


# =============================================================================
# 5. SIMULATION SETTINGS
# =============================================================================

# Number of points used to generate the output
N_POINTS = 5000

# Simulate exactly one orbital period
SIMULATION_TIME = ORBIT_PERIOD


# -----------------------------------------------------------------------------
# Limit-check mode
# -----------------------------------------------------------------------------
#
# False: simulate exactly one orbit (SIMULATION_TIME).
#
# True:  simulate until |attitude deviation| > MAX_ANGLE_ARCSEC AND
#        |angular velocity| > MAX_RATE_ARCSEC_S at the same time, continue
#        for LIMIT_OVERRUN_TIME, then stop.
# -----------------------------------------------------------------------------

LIMIT_CHECK_MODE = False

# Maximum allowed attitude deviation
MAX_ANGLE_ARCSEC = 360.0        # arcsec

# Maximum allowed angular velocity
MAX_RATE_ARCSEC_S = 0.3         # arcsec/s

# How long to keep simulating after both limits are exceeded
LIMIT_OVERRUN_TIME = 60.0       # s

# Give up searching for an exceedance after this long
LIMIT_SEARCH_TIME = 10 * ORBIT_PERIOD   # s


# =============================================================================
# 6. DISTURBANCE FUNCTIONS
# =============================================================================

def is_eclipsed(t):
    """
    Returns True when the spacecraft is in eclipse.

    Eclipse is assumed to be centered at half an orbit.
    """

    orbit_fraction = (t / ORBIT_PERIOD) % 1.0

    eclipse_start = 0.5 - ECLIPSE_FRACTION / 2
    eclipse_end = 0.5 + ECLIPSE_FRACTION / 2

    return eclipse_start <= orbit_fraction <= eclipse_end

def solar_radiation_torque(t):
    """Solar radiation pressure torque."""

    if not SRP_ENABLED:
        return 0.0

    # No solar radiation during eclipse
    if is_eclipsed(t):
        return 0.0

    if SRP_CYCLICAL:
        omega = 2 * np.pi * SRP_CYCLES_PER_ORBIT / ORBIT_PERIOD
        return SRP_TORQUE * np.sin(omega * t + SRP_PHASE)

    return SRP_TORQUE

def thermal_radiation_torque(t):
    """Mercury thermal radiation torque."""

    if not THERMAL_ENABLED:
        return 0.0

    if THERMAL_CYCLICAL:
        omega = 2 * np.pi * THERMAL_CYCLES_PER_ORBIT / ORBIT_PERIOD
        return THERMAL_TORQUE * np.sin(omega * t + THERMAL_PHASE)

    return THERMAL_TORQUE

def magnetic_torque(t):
    """Simplified cyclical magnetic disturbance torque."""

    if not MAGNETIC_ENABLED:
        return 0.0

    omega = (
        2 * np.pi
        * MAGNETIC_CYCLES_PER_ORBIT
        / ORBIT_PERIOD
    )

    return MAGNETIC_TORQUE * np.sin(
        omega * t + MAGNETIC_PHASE
    )

def gravity_gradient_torque(theta):
    """
    Gravity-gradient torque.

    theta is the attitude deviation from nominal LVLH, in radians.

    A nonlinear expression is used here:
        T_gg = -(3*mu/r^3)*(I2-I1)*sin(theta)*cos(theta)

    The sign can be changed depending on the physical axis convention.
    """

    if not GRAVITY_GRADIENT_ENABLED:
        return 0.0

    # Select the relevant inertia difference for the modeled axis.
    inertia_difference = I_Z - I_X

    coefficient = 3.0 * MU_MERCURY / ORBIT_RADIUS**3 * inertia_difference

    return coefficient * np.sin(theta) * np.cos(theta)


def bias_torque():
    """Constant disturbance torque."""

    if not BIAS_ENABLED:
        return 0.0

    return BIAS_TORQUE


# =============================================================================
# 7. TOTAL DISTURBANCE TORQUE
# =============================================================================

def total_torque(t, theta):
    """Return total disturbance torque."""

    return (
        solar_radiation_torque(t)
        + thermal_radiation_torque(t)
        + magnetic_torque(t)
        + gravity_gradient_torque(theta)
        + bias_torque()
    )


# =============================================================================
# 8. SPACECRAFT ROTATIONAL DYNAMICS
# =============================================================================
#
# State vector:
#
#     y[0] = theta
#     y[1] = theta_dot
#
# Equation:
#
#     I * theta_ddot = T_total
#
# therefore:
#
#     theta_ddot = T_total / I
#


def attitude_dynamics(t, state):

    theta = state[0]
    theta_dot = state[1]

    torque = total_torque(t, theta)

    theta_ddot = torque / I_SPACECRAFT

    return [theta_dot, theta_ddot]


# =============================================================================
# 9. INITIAL CONDITIONS
# =============================================================================

theta_0 = np.deg2rad(INITIAL_ANGLE_DEG)
theta_dot_0 = np.deg2rad(INITIAL_RATE_DEG_S)

initial_state = [
    theta_0,
    theta_dot_0
]


# =============================================================================
# 10. RUN SIMULATION
# =============================================================================

max_angle_rad = np.deg2rad(MAX_ANGLE_ARCSEC / 3600.0)
max_rate_rad_s = np.deg2rad(MAX_RATE_ARCSEC_S / 3600.0)

# Time at which both limits are first exceeded (None if they never are)
limit_exceeded_time = None


def limits_exceeded_event(t, state):
    """
    Positive only while BOTH limits are exceeded, negative otherwise.

    The zero crossing (upwards) marks the moment both are exceeded.
    """

    angle_margin = abs(state[0]) - max_angle_rad
    rate_margin = abs(state[1]) - max_rate_rad_s

    return min(angle_margin, rate_margin)


limits_exceeded_event.terminal = True
limits_exceeded_event.direction = 1


if LIMIT_CHECK_MODE:

    if limits_exceeded_event(0.0, initial_state) > 0:
        limit_exceeded_time = 0.0

    else:
        search = solve_ivp(
            attitude_dynamics,
            [0.0, LIMIT_SEARCH_TIME],
            initial_state,
            events=limits_exceeded_event,
            rtol=1e-9,
            atol=1e-12
        )

        if search.t_events[0].size > 0:
            limit_exceeded_time = search.t_events[0][0]

    if limit_exceeded_time is not None:
        SIMULATION_TIME = limit_exceeded_time + LIMIT_OVERRUN_TIME
    else:
        print(
            f"\nLimits not exceeded within "
            f"{LIMIT_SEARCH_TIME / 60:.1f} min; showing the full search time."
        )
        SIMULATION_TIME = LIMIT_SEARCH_TIME


time = np.linspace(
    0.0,
    SIMULATION_TIME,
    N_POINTS
)

solution = solve_ivp(
    attitude_dynamics,
    [0.0, SIMULATION_TIME],
    initial_state,
    t_eval=time,
    rtol=1e-9,
    atol=1e-12
)


# =============================================================================
# 11. EXTRACT RESULTS
# =============================================================================

theta = solution.y[0]
theta_dot = solution.y[1]

# Calculate angular acceleration at every output point
angular_acceleration = np.zeros_like(time)

for i, t in enumerate(time):

    torque = total_torque(
        t,
        theta[i]
    )

    angular_acceleration[i] = torque / I_SPACECRAFT


# Convert to useful engineering units

theta_deg = np.rad2deg(theta)

theta_dot_deg_s = np.rad2deg(theta_dot)

theta_dot_arcsec_s = theta_dot_deg_s * 3600.0

angular_acceleration_deg_s2 = np.rad2deg(
    angular_acceleration
)


# =============================================================================
# 12. INDIVIDUAL DISTURBANCE TORQUES
# =============================================================================

srp_torque_history = np.array([
    solar_radiation_torque(t)
    for t in time
])

thermal_torque_history = np.array([
    thermal_radiation_torque(t)
    for t in time
])

magnetic_torque_history = np.array([
    magnetic_torque(t)
    for t in time
])

gravity_gradient_history = np.array([
    gravity_gradient_torque(theta[i])
    for i in range(len(time))
])

bias_torque_history = np.array([
    bias_torque()
    for t in time
])


# =============================================================================
# 13. PRINT SUMMARY
# =============================================================================

print("\nSimulation results")
print("------------------")

print(
    f"Maximum attitude deviation: "
    f"{np.max(np.abs(theta_deg)):.6f} deg"
)

print(
    f"Maximum angular velocity: "
    f"{np.max(np.abs(theta_dot_deg_s)):.6e} deg/s"
)

print(
    f"Maximum angular acceleration: "
    f"{np.max(np.abs(angular_acceleration_deg_s2)):.6e} deg/s^2"
)

if limit_exceeded_time is not None:
    print(
        f"Limits exceeded at: "
        f"{limit_exceeded_time:.2f} s ({limit_exceeded_time / 60:.2f} min)"
    )


# =============================================================================
# 14. PLOT RESULTS
# =============================================================================

time_minutes = time / 60.0

# Eclipse intervals [min] of every orbit, clipped to the simulated time
eclipse_intervals = []

for orbit in range(int(np.ceil(SIMULATION_TIME / ORBIT_PERIOD))):

    eclipse_start = (
        orbit + 0.5 - ECLIPSE_FRACTION / 2
    ) * ORBIT_PERIOD

    eclipse_end = (
        orbit + 0.5 + ECLIPSE_FRACTION / 2
    ) * ORBIT_PERIOD

    if eclipse_start < SIMULATION_TIME:
        eclipse_intervals.append((
            eclipse_start / 60,
            min(eclipse_end, SIMULATION_TIME) / 60
        ))

# Plot panels, in display order, and whether each is shown at start-up.
# Each panel can be switched on/off with the checkboxes at the top of the
# window while the figure is open.
PLOT_NAMES = [
    "angle",
    "angular velocity",
    "angular acceleration",
    "disturbance torques",
]

PLOT_DEFAULTS = {
    "angle": True,
    "angular velocity": True,
    "angular acceleration": False,
    "disturbance torques": True,
}

plot_visibility = dict(PLOT_DEFAULTS)

fig = plt.figure(figsize=(10, 12))

# Panels are created once; update_layout() positions the visible ones.
axes = [fig.add_subplot(4, 1, i + 1) for i in range(len(PLOT_NAMES))]

for ax in axes[1:]:
    ax.sharex(axes[0])

for ax in axes:
    for eclipse_start, eclipse_end in eclipse_intervals:
        ax.axvspan(
            eclipse_start,
            eclipse_end,
            alpha=0.2,
            color="gray"
        )


def first_exceedance_minutes(values, limit):
    """Time [min] at which |values| first exceeds limit, or None."""

    exceeded = np.abs(values) > limit

    if not np.any(exceeded):
        return None

    return time_minutes[np.argmax(exceeded)]


def mark_exceedance(ax, values, limit):
    """Draw a vertical line where |values| first exceeds limit."""

    exceedance_time = first_exceedance_minutes(values, limit)

    if exceedance_time is not None:
        ax.axvline(
            exceedance_time,
            color="red",
            linestyle=":",
            linewidth=1.5
        )


# -----------------------------------------------------------------------------
# Attitude deviation
# -----------------------------------------------------------------------------

axes[0].plot(
    time_minutes,
    theta_deg,
    linewidth=2
)

axes[0].axhline(
    0,
    linestyle="--",
    linewidth=1
)

if LIMIT_CHECK_MODE:
    for sign in (1, -1):
        axes[0].axhline(
            sign * MAX_ANGLE_ARCSEC / 3600.0,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Limit (±{MAX_ANGLE_ARCSEC:g} arcsec)" if sign > 0 else None
        )

    mark_exceedance(axes[0], theta_deg, MAX_ANGLE_ARCSEC / 3600.0)

    axes[0].legend(loc="upper left")

axes[0].set_ylabel(
    "Attitude deviation [deg]"
)

axes[0].set_title(
    "Attitude deviation from LVLH"
)

axes[0].grid(True)


# -----------------------------------------------------------------------------
# Angular velocity
# -----------------------------------------------------------------------------

axes[1].plot(
    time_minutes,
    theta_dot_arcsec_s,
    linewidth=2
)

axes[1].axhline(
    0,
    linestyle="--",
    linewidth=1
)

if LIMIT_CHECK_MODE:
    for sign in (1, -1):
        axes[1].axhline(
            sign * MAX_RATE_ARCSEC_S,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Limit (±{MAX_RATE_ARCSEC_S:g} arcsec/s)" if sign > 0 else None
        )

    mark_exceedance(axes[1], theta_dot_arcsec_s, MAX_RATE_ARCSEC_S)

    axes[1].legend(loc="upper left")

axes[1].set_ylabel(
    "Angular velocity [arcsec/s]"
)

axes[1].set_title(
    "Angular velocity relative to LVLH"
)

axes[1].grid(True)


# -----------------------------------------------------------------------------
# Angular acceleration
# -----------------------------------------------------------------------------

axes[2].plot(
    time_minutes,
    angular_acceleration_deg_s2,
    linewidth=2
)

axes[2].axhline(
    0,
    linestyle="--",
    linewidth=1
)

axes[2].set_ylabel(
    "Angular acceleration [deg/s²]"
)

axes[2].set_title(
    "Angular acceleration"
)

axes[2].grid(True)


# -----------------------------------------------------------------------------
# Disturbance torques
# -----------------------------------------------------------------------------

axes[3].plot(
    time_minutes,
    srp_torque_history,
    label="Solar radiation pressure"
)

axes[3].plot(
    time_minutes,
    thermal_torque_history,
    label="Thermal (re-emission and Mercury radiation)"
)

if MAGNETIC_ENABLED:
    axes[3].plot(
        time_minutes,
        magnetic_torque_history,
        label="Magnetic"
    )

axes[3].plot(
    time_minutes,
    gravity_gradient_history,
    label="Gravity gradient"
)

if BIAS_ENABLED:
    axes[3].plot(
        time_minutes,
        bias_torque_history,
        label="Bias torque"
    )

axes[3].set_ylabel(
    "Torque [N m]"
)

axes[3].set_title(
    "Disturbance torques"
)

axes[3].legend()

axes[3].grid(True)


# -----------------------------------------------------------------------------
# Panel layout and plot-selection checkboxes
# -----------------------------------------------------------------------------

# Figure-fraction limits of the panel area (the strip above it holds the
# checkboxes) and the vertical gap left between panels for their titles.
PANELS_LEFT = 0.10
PANELS_RIGHT = 0.95
PANELS_TOP = 0.89
PANELS_BOTTOM = 0.06
PANELS_GAP = 0.06


def update_layout():
    """Stack the selected panels to fill the window; hide the rest."""

    visible_axes = [
        ax for ax, name in zip(axes, PLOT_NAMES)
        if plot_visibility[name]
    ]

    for ax in axes:
        ax.set_visible(False)

    n_visible = len(visible_axes)

    if n_visible > 0:

        panel_height = (
            PANELS_TOP - PANELS_BOTTOM - PANELS_GAP * (n_visible - 1)
        ) / n_visible

        for k, ax in enumerate(visible_axes):

            bottom = PANELS_TOP - (k + 1) * panel_height - k * PANELS_GAP

            ax.set_position([
                PANELS_LEFT,
                bottom,
                PANELS_RIGHT - PANELS_LEFT,
                panel_height
            ])

            ax.set_visible(True)

            # Time axis labels only on the lowest visible panel
            is_lowest = (k == n_visible - 1)
            ax.tick_params(labelbottom=is_lowest)
            ax.set_xlabel("Time [min]" if is_lowest else "")

    fig.canvas.draw_idle()


def on_checkbox_clicked(label):
    plot_visibility[label] = not plot_visibility[label]
    update_layout()


# One single-item checkbox per panel, placed side by side in a row at the
# top of the window. References are kept in a list so the widgets stay alive.
CHECKBOX_LEFT = 0.08
CHECKBOX_WIDTH = 0.22
CHECKBOX_BOTTOM = 0.945
CHECKBOX_HEIGHT = 0.035

checkboxes = []

for i, name in enumerate(PLOT_NAMES):

    checkbox_ax = fig.add_axes([
        CHECKBOX_LEFT + i * CHECKBOX_WIDTH,
        CHECKBOX_BOTTOM,
        CHECKBOX_WIDTH,
        CHECKBOX_HEIGHT
    ])

    for spine in checkbox_ax.spines.values():
        spine.set_visible(False)

    checkbox = CheckButtons(
        checkbox_ax,
        [name],
        [plot_visibility[name]]
    )

    checkbox.on_clicked(on_checkbox_clicked)

    checkboxes.append(checkbox)


update_layout()

plt.show()
