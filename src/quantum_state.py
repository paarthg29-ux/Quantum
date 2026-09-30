import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


# ============================================================
# HARYANA GNN → QUANTUM WARM-START
# ============================================================
#
# Five qubits represent five selected Haryana generation buses:
#
# q0 -> Panipat
# q1 -> Faridabad
# q2 -> Hisar
# q3 -> Yamunanagar
# q4 -> Jhajjar
#
# The GNN produces probabilities p_i.
#
# These are converted to Ry rotation angles:
#
#       theta_i = 2 * arcsin(sqrt(p_i))
#
# and encoded into the quantum state using:
#
#       Ry(theta_i)
#
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_QUBITS = 5

GENERATOR_NAMES = [
    "Panipat",
    "Faridabad",
    "Hisar",
    "Yamunanagar",
    "Jhajjar"
]

# ------------------------------------------------------------
# GNN probabilities from the current Haryana test scenario
# ------------------------------------------------------------
#
# These values came from the current GNN run for:
#
#       Test demand = 4000 MW
#
# IMPORTANT:
# These are the current prototype values.
# Later, once the GNN is trained using proper commitment labels,
# these numbers will be generated automatically by the GNN.
#

GNN_PROBABILITIES = np.array(
    [
        0.58,   # Panipat
        0.65,   # Faridabad
        0.65,   # Hisar
        0.89,   # Yamunanagar
        0.70    # Jhajjar
    ],
    dtype=np.float64
)


# ============================================================
# 2. CONVERT GNN PROBABILITIES TO QUANTUM ANGLES
# ============================================================

# Numerical safety:
# probabilities must remain strictly inside [0, 1]
GNN_PROBABILITIES = np.clip(
    GNN_PROBABILITIES,
    1e-6,
    1.0 - 1e-6
)


# Formula:
#
#       theta = 2 * arcsin(sqrt(p))
#
theta_angles = (
    2.0
    * np.arcsin(
        np.sqrt(GNN_PROBABILITIES)
    )
)


# ============================================================
# 3. DISPLAY GNN → QUANTUM MAPPING
# ============================================================

print()
print("=" * 70)
print("        HARYANA GNN → QUANTUM WARM-START")
print("=" * 70)

print()
print("Qubit / Generator Mapping:")
print("-" * 70)

for i, name in enumerate(GENERATOR_NAMES):

    print(
        f"q{i} → {name:12s} | "
        f"GNN Probability = {GNN_PROBABILITIES[i]:.4f} | "
        f"Ry Angle = {theta_angles[i]:.4f} rad"
    )


# ============================================================
# 4. INITIALIZE 5-QUBIT QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=NUM_QUBITS
)


# ============================================================
# 5. QUANTUM STATE GENERATION
# ============================================================

@qml.qnode(dev)
def generate_quantum_state_vector(angles):

    # Start from:
    #
    #       |00000>
    #
    # Apply one Ry rotation to each generator qubit.

    for i in range(NUM_QUBITS):

        qml.RY(
            angles[i],
            wires=i
        )

    # Return the complete state vector:
    #
    #       |psi> = sum_k c_k |k>
    #
    return qml.state()


# ============================================================
# 6. EXECUTE QUANTUM CIRCUIT
# ============================================================

state_vector = generate_quantum_state_vector(
    theta_angles
)


# ============================================================
# 7. EXTRACT AMPLITUDES AND PROBABILITIES
# ============================================================

real_amplitudes = np.real(
    state_vector
)

imag_amplitudes = np.imag(
    state_vector
)

probabilities = np.abs(
    state_vector
) ** 2


# ============================================================
# 8. COMPUTATIONAL BASIS STATES
# ============================================================

NUM_STATES = 2 ** NUM_QUBITS

# Use ASCII > instead of the Unicode ket symbol in plots.
# This avoids the Windows/Arial missing-glyph warning.

basis_states = [
    f"|{i:05b}>"
    for i in range(NUM_STATES)
]


# ============================================================
# 9. TERMINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("              5-QUBIT QUANTUM STATE VECTOR")
print("=" * 70)

print(
    f"\nNumber of Qubits: {NUM_QUBITS}"
)

print(
    f"Hilbert Space Dimension: 2^{NUM_QUBITS} = "
    f"{NUM_STATES}"
)

print(
    f"State Vector Shape: {state_vector.shape}"
)

print()
print(
    "Generator / Qubit Mapping:"
)

for i, name in enumerate(GENERATOR_NAMES):

    print(
        f"q{i} = {name}"
    )


print()
print("-" * 70)
print("STATE VECTOR")
print("-" * 70)


for i in range(NUM_STATES):

    print(
        f"State {i:2d} {basis_states[i]:7s}: "
        f"Amplitude = "
        f"{real_amplitudes[i]:+.4f} "
        f"{imag_amplitudes[i]:+.4f}i"
        f"   |   "
        f"Probability = "
        f"{probabilities[i]:.4f}"
    )


# ============================================================
# 10. NORMALIZATION CHECK
# ============================================================

normalization = np.sum(
    probabilities
)

print()
print("-" * 70)
print("NORMALIZATION CHECK")
print("-" * 70)

print(
    f"Sum of |c_k|^2 = {normalization:.6f}"
)

if np.isclose(
    normalization,
    1.0
):

    print(
        "✓ Quantum state is properly normalized."
    )

else:

    print(
        "WARNING: State normalization is incorrect."
    )


# ============================================================
# 11. FIND MOST PROBABLE STATES
# ============================================================

top_k = 5

top_indices = np.argsort(
    probabilities
)[-top_k:][::-1]


print()
print("=" * 70)
print(f"TOP {top_k} MOST PROBABLE QUANTUM STATES")
print("=" * 70)


for rank, idx in enumerate(
    top_indices,
    start=1
):

    bitstring = format(
        idx,
        "05b"
    )

    print(
        f"{rank}. |{bitstring}> "
        f"→ Probability = "
        f"{probabilities[idx]:.4f} "
        f"({probabilities[idx] * 100:.2f}%)"
    )


# ============================================================
# 12. INTERPRETATION OF A BITSTRING
# ============================================================

def decode_generator_state(bitstring):
    """
    Converts a 5-bit quantum state into generator ON/OFF status.

    Example:

        11111

    means all five generators are ON.

    Bit order:

        q0 -> Panipat
        q1 -> Faridabad
        q2 -> Hisar
        q3 -> Yamunanagar
        q4 -> Jhajjar
    """

    result = []

    for qubit, bit in enumerate(bitstring):

        status = (
            "ON"
            if bit == "1"
            else "OFF"
        )

        result.append(
            f"{GENERATOR_NAMES[qubit]}={status}"
        )

    return ", ".join(result)


print()
print("=" * 70)
print("TOP STATES — GENERATOR INTERPRETATION")
print("=" * 70)

for rank, idx in enumerate(
    top_indices,
    start=1
):

    bitstring = format(
        idx,
        "05b"
    )

    interpretation = decode_generator_state(
        bitstring
    )

    print()
    print(
        f"{rank}. |{bitstring}>"
    )

    print(
        f"   Probability = "
        f"{probabilities[idx] * 100:.2f}%"
    )

    print(
        f"   {interpretation}"
    )


# ============================================================
# 13. VISUALIZATION
# ============================================================

# Use a clean Matplotlib style.
# No seaborn dependency is required.

plt.style.use(
    "default"
)


# ============================================================
# FIGURE 1:
# AMPLITUDES + MEASUREMENT PROBABILITIES
# ============================================================

x_indices = np.arange(
    NUM_STATES
)

fig, (
    ax1,
    ax2
) = plt.subplots(
    2,
    1,
    figsize=(16, 10),
    sharex=True
)


# ------------------------------------------------------------
# TOP: STATE VECTOR AMPLITUDES
# ------------------------------------------------------------

markerline, stemlines, baseline = ax1.stem(
    x_indices,
    real_amplitudes
)

plt.setp(
    markerline,
    markersize=7
)

plt.setp(
    stemlines,
    linewidth=2
)

plt.setp(
    baseline,
    linewidth=1
)


ax1.set_title(
    "GNN-Warm-Started 5-Qubit State Vector",
    fontsize=18,
    fontweight="bold"
)

ax1.set_ylabel(
    "Amplitude $c_k$",
    fontsize=13
)

ax1.set_ylim(
    0,
    max(real_amplitudes) * 1.25
)

ax1.grid(
    True,
    alpha=0.25
)


# ------------------------------------------------------------
# Highlight top 3 amplitudes
# ------------------------------------------------------------

top_amplitude_indices = np.argsort(
    real_amplitudes
)[-3:][::-1]


for idx in top_amplitude_indices:

    ax1.annotate(
        f"{basis_states[idx]}\n"
        f"$c_{{{idx}}}$ = "
        f"{real_amplitudes[idx]:.4f}",

        xy=(
            idx,
            real_amplitudes[idx]
        ),

        xytext=(
            idx - 5,
            real_amplitudes[idx] + 0.045
        ),

        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2
        ),

        fontsize=10
    )


# ============================================================
# BOTTOM: PROBABILITIES
# ============================================================

bars = ax2.bar(
    x_indices,
    probabilities,
    alpha=0.85
)


ax2.set_title(
    "Computational-Basis Measurement Probabilities",
    fontsize=18,
    fontweight="bold"
)

ax2.set_xlabel(
    "Generator Commitment State",
    fontsize=13
)

ax2.set_ylabel(
    "Probability $|c_k|^2$",
    fontsize=13
)

ax2.set_xticks(
    x_indices
)

ax2.set_xticklabels(
    basis_states,
    rotation=90,
    fontsize=9,
    fontfamily="monospace"
)

ax2.grid(
    axis="y",
    alpha=0.25
)


# ------------------------------------------------------------
# Highlight top 3 probabilities
# ------------------------------------------------------------

top_probability_indices = np.argsort(
    probabilities
)[-3:][::-1]


for idx in top_probability_indices:

    ax2.text(
        idx,
        probabilities[idx] + 0.005,

        f"{probabilities[idx] * 100:.1f}%",

        ha="center",
        fontsize=10,
        fontweight="bold"
    )


# ------------------------------------------------------------
# Generator mapping annotation
# ------------------------------------------------------------

mapping_text = (
    "Qubit Mapping:\n"
    "q0 = Panipat\n"
    "q1 = Faridabad\n"
    "q2 = Hisar\n"
    "q3 = Yamunanagar\n"
    "q4 = Jhajjar"
)

ax2.text(
    1.01,
    0.95,
    mapping_text,
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        alpha=0.08
    )
)


plt.tight_layout()


# Save high-resolution version for PPT/report

plt.savefig(
    "quantum_state_analysis.png",
    dpi=300,
    bbox_inches="tight"
)


print()
print(
    "Saved: quantum_state_analysis.png"
)


plt.show()


# ============================================================
# FIGURE 2:
# TOP QUANTUM STATES ONLY
# ============================================================

top_states = [
    basis_states[i]
    for i in top_indices
]

top_probabilities = [
    probabilities[i]
    for i in top_indices
]


plt.figure(
    figsize=(11, 6)
)

bars = plt.bar(
    top_states,
    top_probabilities,
    alpha=0.9
)


plt.title(
    "Most Probable 5-Qubit Generator States",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Generator Commitment State",
    fontsize=13
)

plt.ylabel(
    "Measurement Probability",
    fontsize=13
)

plt.grid(
    axis="y",
    alpha=0.25
)


# Add percentage labels

for bar, probability in zip(
    bars,
    top_probabilities
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        probability + 0.004,

        f"{probability * 100:.2f}%",

        ha="center",
        fontweight="bold"
    )


plt.tight_layout()


plt.savefig(
    "top_quantum_states.png",
    dpi=300,
    bbox_inches="tight"
)


print(
    "Saved: top_quantum_states.png"
)


plt.show()


# ============================================================
# FIGURE 3:
# GNN PROBABILITY → Ry ANGLE
# ============================================================

x = np.arange(
    NUM_QUBITS
)

width = 0.35


fig, ax1 = plt.subplots(
    figsize=(11, 6)
)


# GNN probabilities

bars1 = ax1.bar(
    x - width / 2,
    GNN_PROBABILITIES,
    width,
    label="GNN Probability"
)


ax1.set_ylabel(
    "GNN Probability",
    fontsize=12
)

ax1.set_ylim(
    0,
    1
)


# Quantum angles

ax2 = ax1.twinx()


bars2 = ax2.bar(
    x + width / 2,
    theta_angles,
    width,
    label="Ry Angle"
)


ax2.set_ylabel(
    "Ry Rotation Angle (radians)",
    fontsize=12
)


ax1.set_xticks(
    x
)

ax1.set_xticklabels(
    GENERATOR_NAMES
)


plt.title(
    "GNN Probability → Quantum $R_y$ Rotation",
    fontsize=18,
    fontweight="bold"
)


ax1.grid(
    axis="y",
    alpha=0.25
)


# Probability labels

for bar, value in zip(
    bars1,
    GNN_PROBABILITIES
):

    ax1.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.025,

        f"{value:.2f}",

        ha="center",
        fontsize=9
    )


# Angle labels

for bar, value in zip(
    bars2,
    theta_angles
):

    ax2.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 0.05,

        f"{value:.3f}",

        ha="center",
        fontsize=9
    )


fig.tight_layout()


plt.savefig(
    "gnn_to_quantum_angles.png",
    dpi=300,
    bbox_inches="tight"
)


print(
    "Saved: gnn_to_quantum_angles.png"
)


plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("                    EXECUTION COMPLETE")
print("=" * 70)

print()
print("Quantum Pipeline:")
print()
print("Haryana Grid")
print("     ↓")
print("GNN Generator Probabilities")
print("     ↓")
print("p_i")
print("     ↓")
print("theta_i = 2 asin(sqrt(p_i))")
print("     ↓")
print("Ry(theta_i)")
print("     ↓")
print("5-Qubit Quantum State")
print("     ↓")
print("32 Computational-Basis States")
print("     ↓")
print("Measurement Probability Distribution")

print()
print("Generated files:")
print("  1. quantum_state_analysis.png")
print("  2. top_quantum_states.png")
print("  3. gnn_to_quantum_angles.png")

print()
print("Normalization:")
print(f"  Sum |c_k|^2 = {normalization:.6f}")

print()
print("=" * 70)