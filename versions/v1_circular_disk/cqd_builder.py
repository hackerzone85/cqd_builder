#!/usr/bin/env python3
"""
================================================================================
 cqd_builder.py -- Carbon Quantum Dot (CQD) Structure Builder  (v2)
================================================================================

Generates an all-atom, edge-functionalized Carbon Quantum Dot (CQD) model as
a valid PDB coordinate file, suitable as a starting geometry for force-field
relaxation (GROMACS/OpenMM/AMBER) or as a template for subsequent DFT
(ORCA/Gaussian) cluster/QM-region calculations.

ALGORITHMIC / MATHEMATICAL BASIS
---------------------------------
Version 2 rebuilds the lattice engine to precisely follow the algorithm
published in the Supporting Information of:

    Paloncyova, M.; Langer, M.; Otyepka, M. "Structural Dynamics of Carbon
    Dots in Water and N,N-Dimethylformamide Probed by All-Atom Molecular
    Dynamics Simulations." J. Chem. Theory Comput. 2018, 14, 2076-2083.
    (VMD "Carbon Dot Builder" plug-in; http://cd-builder.upol.cz)

WHAT IS REPRODUCED EXACTLY:
  1. The two-atom hexagonal basis and honeycomb lattice geometry (aromatic
     C-C bond length a, default 1.42 A).
  2. The "edge length in benzene rings" sizing parameter itself (the
     reference GUI's "Edge length of middle layer (benzene rings)" field),
     and its effect on shape: `--edge_rings` N produces the same
     circumscribed-PAH-family flake size as benzene (N=1), coronene (N=2),
     circumcoronene (N=3), etc. -- see IMPLEMENTATION NOTE below for how
     this is constructed and verified.
  3. The per-layer size rule while stacking: the layer immediately above
     (or below) the middle layer keeps the SAME edge-ring count as the
     middle layer; every layer after that has its edge-ring count reduced
     by 1, stacked at the graphitic (002) spacing (3.4 A default). This
     reproduces Table 2 of the paper exactly (edge_rings=6 -> 3 layers
     above middle -> 7 total layers).
  4. Randomized edge passivation at a user-set coverage fraction, with an
     option controlling whether functionalized edge sites are allowed to
     sit next to each other (disabling this caps achievable coverage at
     roughly 50%, exactly as noted in the SI).

IMPLEMENTATION NOTE on the hexagonal flake shape:
  The SI's Figure S1 describes the reference tool's internal lattice loop
  (a row/cell tiling with a corner-trimming rule) only in prose, with the
  full numeric detail given in a raster figure. During development, a
  literal transcription of that prose description was implemented and
  rigorously tested (atom-count checks, bond-degree checks, 6-fold
  rotational symmetry checks) -- and *failed* those tests: the resulting
  shape was a sheared, asymmetric hexagon, not a proper one. Rather than
  ship that (or guess at further correction constants that couldn't be
  independently verified), this tool instead builds the flake as
  `edge_rings` concentric shells of complete hexagonal rings around a
  central ring -- the same construction family as the real reference
  molecules benzene (1 ring), coronene (7 rings / 2 shells), and
  circumcoronene (19 rings / 3 shells). This is verified correct three
  independent ways: (a) atom count is exactly 6*N^2, matching those real
  molecules' known formulas (C6, C24, C54, ...); (b) the resulting
  carbon-degree distribution exactly matches their known bonding pattern
  (e.g. coronene's real C24H12 = 12 interior degree-3 carbons + 12 rim
  degree-2 carbons, exactly reproduced here); (c) every atom has degree
  >= 2 by construction, so dangling/under-bonded atoms are geometrically
  impossible. This trades literal line-by-line transcription of an
  unverifiable prose description for a construction that is independently
  checkable against real, well-characterized chemistry -- and produces
  the same physical target (a hexagonal, edge-length-in-benzene-rings-
  sized graphitic flake) either way.

WHAT IS AN EQUIVALENT-RESULT APPROXIMATION (documented, not hidden):
  - Idealized Bernal (AB) interlayer registry and concentric alignment of
    differently-sized layers are achieved here by (a) centering each
    layer's own centroid at (x=0, y=0) and (b) offsetting alternating
    layers by the honeycomb sublattice vector (r2 - r1). This produces the
    same physically correct AB-stacked, concentric result as the
    reference tool, but via an independently-verifiable geometric
    construction rather than by transcribing the reference's internal
    per-layer offset constants, which are only fully specified through a
    raster figure (Figure S2) and were not reliably reproducible from
    prose alone (see IMPLEMENTATION NOTE above for the analogous issue
    encountered with the flake shape itself). The chemistry/registry
    result (AB stacking, concentric layers) is identical either way.
  - Outward-pointing directions for edge substituents are computed from
    each edge carbon's actual local bonding topology (reversed average of
    its remaining neighbor-bond vectors -- the standard sp2 "vacant
    valence" direction), rather than the six precomputed edge-vectors of
    Figure S2. This generalizes correctly to every edge site (zigzag,
    armchair, and hexagon corners) without needing the six hardcoded
    vector formulas, which the SI text describes only qualitatively.
  - Functional-group internal geometry (bond lengths/angles) uses standard
    literature values (see FUNCTIONAL_GROUPS below), in the same spirit as
    the reference tool's own approach ("angles ... does not ideally
    reflect the true relaxed geometry ... but allows assignment of proper
    bonds with the expectation of following structure relaxation" -- SI,
    Figure S3 caption). As in the reference tool, output structures should
    be energy-minimized before production MD.
  - Optional partial charges (``--assign_charges``) reproduce the
    HF/6-31G*-derived, AMBER99SB-compatible, "armchair" values from Table 1
    of the paper for the four groups it explicitly parameterized (edge
    C/H, hydroxyl, carbonyl, carboxyl, carboxyl-charged). The paper
    reports distinct armchair/zigzag values per atom; this tool applies
    the armchair value uniformly as a documented simplification (proper
    zigzag/armchair site classification requires ring perception not
    implemented here). Groups outside the paper's scope (NH2, SH, F) have
    no literature charge set and are left unassigned.

DESIGN NOTE ON DEPENDENCIES
----------------------------
Lattice generation and neighbor/bond perception use NumPy + SciPy
(cKDTree) rather than a full cheminformatics stack (ASE/RDKit), keeping
the tool dependency-light, deterministic, and fully inspectable. Every
generated structure passes through explicit geometric validation
(`validate_structure`) before being written. Output PDBs load cleanly into
ASE, RDKit, MDTraj, VMD, or PyMOL for further processing.

Author: Generated for computational chemistry / nanomaterial workflow use.
License: MIT (adapt freely for lab / academic use).
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    sys.exit(
        "ERROR: this script requires SciPy for neighbor-list construction.\n"
        "Install it with:  mamba install -c conda-forge scipy"
    )


# ==============================================================================
# 1. PHYSICAL / CHEMICAL CONSTANTS
# ==============================================================================

#: Default aromatic C-C bond length in a graphitic sheet (Angstrom).
DEFAULT_BOND_LENGTH = 1.42

#: Default graphitic (002) interlayer spacing (Angstrom). The reference SI
#: text uses 0.335 nm in one place and the main paper / most graphite
#: literature cites 0.34 nm; 3.4 A is kept as the default here (also used
#: for the Radial Distribution Function discussion in the paper's Fig. S11).
DEFAULT_INTERLAYER_SPACING = 3.4

#: Neighbor-search cutoff multiplier applied to the bond length when
#: perceiving covalent C-C bonds within a single graphitic layer.
BOND_CUTOFF_FACTOR = 1.13

#: C-H bond length used for simple valence-capping hydrogens on
#: non-functionalized edge carbons (Angstrom).
EDGE_CAP_CH_BOND_LENGTH = 1.09

#: Van der Waals radii (Angstrom), converted from the standard pm table.
#: Used for element-pair-aware steric clash detection during mixed-group
#: edge functionalization: two non-bonded atoms are flagged as clashing if
#: their separation falls below `CLASH_VDW_FRACTION * (r_vdw[a] + r_vdw[b])`.
VDW_RADII_ANGSTROM: Dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Na": 2.27, "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}

#: Fraction of the summed vdW radii used as the default non-bonded clash
#: threshold (see `pairwise_clash_threshold`). Calibrated so that O-O pairs
#: land at ~1.8 A (deep vdW-sphere overlap, the point at which MM/QM
#: geometry optimizers typically produce unphysically large forces), while
#: larger atom pairs (e.g. S-S) scale up proportionally rather than sharing
#: a single flat cutoff.
DEFAULT_CLASH_VDW_FRACTION = 0.6

#: Maximum number of placement attempts (each with a fresh random dihedral
#: rotation) before falling back to a hydrogen cap at a given site under
#: mixed-group, sterically-constrained functionalization.
DEFAULT_MAX_PLACEMENT_ATTEMPTS = 8


def pairwise_clash_threshold(
    element_a: str,
    element_b: str,
    fraction: float = DEFAULT_CLASH_VDW_FRACTION,
) -> float:
    """Compute the non-bonded steric clash distance threshold for a pair
    of elements, as a fraction of their summed van der Waals radii.

    Args:
        element_a: Element symbol of the first atom.
        element_b: Element symbol of the second atom.
        fraction: Fraction of the summed vdW radii to use as the clash
            threshold (default `DEFAULT_CLASH_VDW_FRACTION` = 0.6, which
            gives ~1.8 A for O-O and scales appropriately for other pairs).

    Returns:
        The clash-distance threshold in Angstrom. Falls back to a generic
        1.5 A heavy-atom estimate for any element not in
        `VDW_RADII_ANGSTROM` (e.g. if a group using an unlisted element is
        added later), so the check never silently no-ops.
    """
    r_a = VDW_RADII_ANGSTROM.get(element_a, 1.5)
    r_b = VDW_RADII_ANGSTROM.get(element_b, 1.5)
    return fraction * (r_a + r_b)


def apportion_counts(total: int, group_names: Sequence[str], ratios: Sequence[float]) -> Dict[str, int]:
    """Split `total` items among `group_names` according to `ratios`, using
    the largest-remainder method so the resulting integer counts always sum
    to exactly `total` (simple per-group rounding can under- or over-shoot
    the total by a few sites).

    Ratios need not be normalized or sum to any particular value --
    `[1, 2, 1]` and `[25, 50, 25]` produce identical splits, since each
    ratio is first divided by the sum of all ratios.

    Args:
        total: Total number of sites to distribute.
        group_names: Functional group keys, in the same order as `ratios`.
        ratios: Relative weights, one per group (need not sum to 1 or 100).

    Returns:
        Dict mapping each group name to its exact integer allocation;
        values sum to exactly `total`.

    Raises:
        ValueError: If `group_names` and `ratios` have different lengths,
            or any ratio is negative, or all ratios are zero.
    """
    if len(group_names) != len(ratios):
        raise ValueError(
            f"--functional_groups has {len(group_names)} entries but "
            f"--group_ratios has {len(ratios)}; they must match 1:1."
        )
    if any(r < 0 for r in ratios):
        raise ValueError("--group_ratios values must be non-negative.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("--group_ratios must sum to a positive value.")

    normalized = [r / ratio_sum for r in ratios]
    raw = [total * f for f in normalized]
    floors = [int(math.floor(x)) for x in raw]
    remainder = total - sum(floors)

    # Hand out the leftover slots to whichever groups have the largest
    # fractional remainder (ties broken by input order, deterministically).
    fractional_parts = sorted(
        range(len(group_names)), key=lambda i: (raw[i] - floors[i]), reverse=True
    )
    counts = list(floors)
    for i in fractional_parts[:remainder]:
        counts[i] += 1

    return dict(zip(group_names, counts))

#: Library of supported edge/surface functional groups. Each entry is a
#: "recipe": a list of atoms placed sequentially via a local z-matrix-like
#: scheme. The first atom of every recipe (parent index -1) is bonded
#: directly to the edge carbon and placed along the outward direction
#: computed from that carbon's local bonding topology (see
#: `compute_outward_directions`). Subsequent atoms (parent = index of a
#: previously placed atom in this recipe) are placed using
#: (bond_length, theta, phi):
#'    theta = interior bond angle (parent's-parent -- parent -- new atom)
#'    phi   = azimuthal angle about the parent's incoming-bond axis
#:
#: OH, COOH, CO (carbonyl) and COO (charged carboxylate) match the four
#: groups explicitly parameterized in Paloncyova et al. Table 1. NH2, SH,
#: and F extend the set to the broader surface-chemistry categories
#: reported across the CQD literature (see accompanying theory notes).
FUNCTIONAL_GROUPS: Dict[str, dict] = {
    "OH": {
        "label": "Hydroxyl (-OH)",
        "resname": "HYD",
        "recipe": [
            # element, atom name, parent, bond length (A), theta (deg), phi (deg)
            ("O", "O1", -1, 1.43, None, None),
            ("H", "H1", 0, 0.96, 105.0, 0.0),
        ],
    },
    "COOH": {
        "label": "Carboxylic acid, neutral/protonated (-COOH)",
        "resname": "COA",
        "recipe": [
            ("C", "C1", -1, 1.52, None, None),   # carboxyl carbon
            ("O", "O1", 0, 1.21, 120.0, 0.0),    # C=O (carbonyl oxygen)
            ("O", "O2", 0, 1.36, 120.0, 180.0),  # C-OH (hydroxyl oxygen)
            ("H", "H1", 2, 0.96, 109.5, 0.0),    # hydroxyl hydrogen
        ],
    },
    "COO": {
        "label": "Carboxylate, deprotonated/charged (-COO-)",
        "resname": "COM",
        "recipe": [
            ("C", "C1", -1, 1.52, None, None),   # carboxylate carbon
            ("O", "O1", 0, 1.26, 120.0, 0.0),    # resonance-delocalized O
            ("O", "O2", 0, 1.26, 120.0, 180.0),  # resonance-delocalized O
        ],
    },
    "NH2": {
        "label": "Amine (-NH2)",
        "resname": "AMN",
        "recipe": [
            ("N", "N1", -1, 1.47, None, None),
            ("H", "H1", 0, 1.01, 109.5, 60.0),
            ("H", "H2", 0, 1.01, 109.5, -60.0),
        ],
    },
    "CO": {
        "label": "Carbonyl / ketone edge state (=O)",
        "resname": "KET",
        "recipe": [
            ("O", "O1", -1, 1.21, None, None),
        ],
    },
    "SH": {
        "label": "Thiol (-SH)",
        "resname": "THL",
        "recipe": [
            ("S", "S1", -1, 1.81, None, None),
            ("H", "H1", 0, 1.34, 96.0, 0.0),
        ],
    },
    "F": {
        "label": "Fluorine (-F)",
        "resname": "FLR",
        "recipe": [
            ("F", "F1", -1, 1.35, None, None),
        ],
    },
}

#: Partial charges (e) reproduced from Table 1 of Paloncyova et al. (2018),
#: HF/6-31G* values compatible with AMBER99SB, "armchair" column. Keyed by
#: (resname, atom_name) for group atoms, and by a special "EDGE" key for
#: the edge carbon each group is bonded to, and "CAP" for H-capped edge
#: carbons/hydrogens. Only the four groups explicitly studied in the paper
#: have literature values; NH2/SH/F are intentionally absent.
PARTIAL_CHARGES: Dict[Tuple[str, str], float] = {
    # Pure/H-capped edge (no functional group)
    ("COR", "CAP_C"): -0.180,
    ("COR", "CAP_H"): 0.120,
    # Hydroxyl
    ("HYD", "EDGE_C"): 0.374,
    ("HYD", "O1"): -0.605,
    ("HYD", "H1"): 0.400,
    # Carbonyl
    ("KET", "EDGE_C"): 0.705,
    ("KET", "O1"): -0.580,
    # Carboxylic acid (neutral)
    ("COA", "EDGE_C"): -0.106,
    ("COA", "C1"): 0.766,
    ("COA", "O1"): -0.610,
    ("COA", "O2"): -0.610,
    ("COA", "H1"): 0.427,
    # Carboxylate (charged)
    ("COM", "EDGE_C"): -0.150,
    ("COM", "C1"): 1.017,
    ("COM", "O1"): -0.901,
    ("COM", "O2"): -0.901,
}


# ==============================================================================
# 2. DATA STRUCTURES
# ==============================================================================

@dataclass
class Atom:
    """A single output atom record destined for the PDB file.

    Attributes:
        element: Element symbol (e.g. 'C', 'O', 'H').
        name: PDB atom name (e.g. 'C1', 'O2').
        x, y, z: Cartesian coordinates in Angstrom.
        resname: 3-character PDB residue name (e.g. 'COR' for core carbon,
            'HYD' for a hydroxyl group instance).
        resseq: PDB residue sequence number.
        chain: Single-character PDB chain identifier.
        layer_index: Internal bookkeeping index of the graphitic layer this
            atom belongs to (or was spawned from, for edge substituents).
        charge_key: Optional (resname, atom_name) lookup key into
            PARTIAL_CHARGES for this atom, used only when
            `--assign_charges` is requested.
    """

    element: str
    name: str
    x: float
    y: float
    z: float
    resname: str
    resseq: int
    chain: str = "A"
    layer_index: int = -1
    charge_key: Optional[Tuple[str, str]] = None


@dataclass
class Structure:
    """Container for the full generated CQD atomic model.

    Attributes:
        atoms: Ordered list of Atom records. List position (0-based) plus 1
            is the final PDB serial number.
        bonds: List of (i, j) 0-based index pairs into `atoms` describing
            covalent connectivity, used to emit PDB CONECT records.
    """

    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Tuple[int, int]] = field(default_factory=list)

    def add_atom(self, atom: Atom) -> int:
        """Append an atom and return its 0-based index in the structure."""
        self.atoms.append(atom)
        return len(self.atoms) - 1

    def add_bond(self, i: int, j: int) -> None:
        """Register a covalent bond between two atom indices."""
        self.bonds.append((i, j))


# ==============================================================================
# 3. LATTICE GEOMETRY  (exact reference-algorithm reproduction)
# ==============================================================================

def _hexagon_ring_centers(n_shells: int, bond_length: float) -> List[np.ndarray]:
    """Enumerate hexagonal-ring center points within `n_shells` graph-distance
    steps of a central ring, via breadth-first shell expansion over the
    triangular lattice of ring centers (lattice constant = bond_length*sqrt(3),
    the correct ring-center-to-ring-center spacing in a honeycomb lattice).

    Using BFS (rather than an algebraic axial-distance formula) avoids the
    off-by-convention errors that a closed-form hex-distance expression is
    easy to get subtly wrong on; each shell is simply "every not-yet-visited
    neighbor of the previous shell," which is correct by construction.

    Args:
        n_shells: Number of shells to expand outward from the central ring
            (0 returns just the central ring).
        bond_length: Aromatic C-C bond length in Angstrom.

    Returns:
        List of (2,) ring-center coordinate arrays. The count is exactly
        the centered hexagonal number 1 + 3*n_shells*(n_shells + 1).
    """
    a = bond_length
    t1 = np.array([a * math.sqrt(3.0), 0.0])
    t2 = np.array([a * math.sqrt(3.0) / 2.0, 1.5 * a])
    directions = [t1, -t1, t2, -t2, t1 - t2, t2 - t1]

    def key(p: np.ndarray) -> Tuple[float, float]:
        return (round(float(p[0]), 3), round(float(p[1]), 3))

    origin = np.array([0.0, 0.0])
    visited = {key(origin): origin}
    frontier = [origin]
    for _ in range(n_shells):
        new_frontier = []
        for p in frontier:
            for d in directions:
                q = p + d
                k = key(q)
                if k not in visited:
                    visited[k] = q
                    new_frontier.append(q)
        frontier = new_frontier
    return list(visited.values())


def generate_hexagonal_flake(edge_rings: int, bond_length: float = DEFAULT_BOND_LENGTH) -> np.ndarray:
    """Generate one hexagonal graphene-like sheet: a "circumscribed" PAH-
    family flake built from `edge_rings` concentric shells of complete
    hexagonal rings around a central ring (the same construction family as
    benzene [1 ring], coronene [7 rings / 2 shells], and circumcoronene
    [19 rings / 3 shells]).

    Each of the `edge_rings` shells' hexagons contributes its 6 corner
    (carbon) atoms, with atoms shared between adjacent rings deduplicated.
    Because every included atom is a corner of at least one fully-included
    ring, this construction is chemically valid by design -- it cannot
    produce a dangling (under-bonded) atom, unlike a naive geometric
    boundary cut on individual lattice points (which can slice through a
    partially-included ring and leave stub atoms; this was tested and
    rejected during development -- see module docstring "IMPLEMENTATION
    NOTE" for details).

    The resulting atom count is exactly 6 * edge_rings**2, matching the
    well-established atom counts of the real reference molecules in this
    series (6, 24, 54, 96, 150, ... for edge_rings = 1, 2, 3, 4, 5, ...),
    and every atom has degree >= 2 (never a dangling degree-1 or isolated
    degree-0 atom).

    Args:
        edge_rings: Number of concentric hexagonal-ring shells, i.e. the
            "edge length" in benzene rings (the reference GUI's "Edge
            length of middle layer (benzene rings)" field). Must be >= 1.
        bond_length: Aromatic C-C bond length in Angstrom.

    Returns:
        An (N, 2) NumPy array of atomic (x, y) coordinates, not yet
        centered (the caller is expected to center the result at its own
        centroid before use, which `build_graphitic_core` does).

    Raises:
        ValueError: If `edge_rings` < 1.
    """
    if edge_rings < 1:
        raise ValueError(f"edge_rings must be >= 1, got {edge_rings}")

    a = bond_length
    n_shells = edge_rings - 1
    centers = _hexagon_ring_centers(n_shells, a)

    seen: Dict[Tuple[float, float], np.ndarray] = {}
    coords: List[np.ndarray] = []
    for center in centers:
        for k in range(6):
            ang = math.radians(30.0 + 60.0 * k)
            atom = center + a * np.array([math.cos(ang), math.sin(ang)])
            key = (round(float(atom[0]), 3), round(float(atom[1]), 3))
            if key not in seen:
                seen[key] = atom
                coords.append(atom)

    return np.vstack(coords)


def layer_ring_count(edge_rings: int, distance_from_middle: int) -> int:
    """Determine the edge-ring count of a layer at a given distance from
    the middle layer, following the SI's stacking rule: "The number of
    benzene rings on each edge stays the same for the second layer but
    then is reduced by one until the number of layers ... is reached."

    Args:
        edge_rings: Edge-ring count of the middle (m=0) layer.
        distance_from_middle: 0 for the middle layer, 1 for the layer
            immediately above/below it, 2 for the next one out, etc.

    Returns:
        The edge-ring count for that layer (same as `edge_rings` for
        m in {0, 1}, then decremented by 1 per additional step).
    """
    m = distance_from_middle
    if m <= 1:
        return edge_rings
    return edge_rings - (m - 1)


def estimate_edge_rings_for_diameter(diameter: float, bond_length: float = DEFAULT_BOND_LENGTH) -> int:
    """Back-compatibility helper: pick the smallest `edge_rings` whose
    generated middle-layer flake has a max pairwise extent (diameter) at
    least as large as the requested value, by direct construction and
    measurement (rather than an approximate closed-form formula).

    Args:
        diameter: Desired approximate middle-layer diameter, in Angstrom.
        bond_length: Aromatic C-C bond length in Angstrom.

    Returns:
        The estimated `edge_rings` integer (>= 1).
    """
    edge_rings = 1
    while edge_rings < 60:
        xy = generate_hexagonal_flake(edge_rings, bond_length)
        span = xy.max(axis=0) - xy.min(axis=0)
        measured_diameter = float(np.max(span))
        if measured_diameter >= diameter:
            return edge_rings
        edge_rings += 1
    return edge_rings


def find_bonded_pairs(
    xy: np.ndarray,
    bond_length: float = DEFAULT_BOND_LENGTH,
) -> List[Tuple[int, int]]:
    """Perceive covalent C-C bonds within a single planar layer via a
    distance cutoff, using a KD-tree for efficient neighbor search.

    Args:
        xy: (N, 2) array of in-plane atomic coordinates for one layer.
        bond_length: Nominal aromatic C-C bond length in Angstrom; the
            search cutoff is `bond_length * BOND_CUTOFF_FACTOR`, which is
            wide enough to catch nearest-neighbor (~1.42 A) bonds while
            safely excluding next-nearest-neighbor distances (~2.46 A).

    Returns:
        A list of (i, j) 0-based index pairs (i < j) local to `xy`.
    """
    if len(xy) == 0:
        return []
    cutoff = bond_length * BOND_CUTOFF_FACTOR
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=cutoff)
    return sorted(pairs)


# ==============================================================================
# 4. FUNCTIONAL GROUP / EDGE-CAP GEOMETRY
# ==============================================================================

def _orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a right-handed orthonormal frame (u, v, w) with u parallel to
    the given axis.

    `v` and `w` are arbitrary but deterministic perpendicular directions,
    chosen via a robust reference-vector cross product so the frame never
    degenerates (switches reference vector when axis is nearly vertical).

    Args:
        axis: A 3-vector defining the desired u-direction (need not be
            normalized).

    Returns:
        Tuple of three orthonormal 3-vectors (u, v, w).
    """
    u = axis / np.linalg.norm(axis)
    world_up = np.array([0.0, 0.0, 1.0])
    reference = np.array([1.0, 0.0, 0.0]) if abs(np.dot(u, world_up)) > 0.95 else world_up
    v = np.cross(reference, u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    return u, v, w


def _place_substituent(
    parent_pos: np.ndarray,
    parent_back_bond: np.ndarray,
    bond_length: float,
    theta_deg: float,
    phi_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place a new atom bonded to `parent_pos` using local internal
    (z-matrix-style) coordinates.

    Args:
        parent_pos: Cartesian position of the parent atom.
        parent_back_bond: Unit vector pointing FROM the parent atom TOWARD
            *its* own parent (i.e. the existing bond direction, reversed
            relative to how it was created). Defines the reference axis for
            `theta_deg`.
        bond_length: New bond length in Angstrom.
        theta_deg: Interior bond angle (grandparent-parent-new_atom) in
            degrees.
        phi_deg: Azimuthal rotation (degrees) of the new bond about the
            `parent_back_bond` axis.

    Returns:
        Tuple of (new_position, outgoing_back_bond) where `outgoing_back_bond`
        is the unit vector from the new atom back toward `parent_pos`,
        ready to be used as `parent_back_bond` for a further substituent.
    """
    u, v, w = _orthonormal_frame(parent_back_bond)
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    direction = math.cos(theta) * u + math.sin(theta) * (math.cos(phi) * v + math.sin(phi) * w)
    direction = direction / np.linalg.norm(direction)
    new_pos = parent_pos + bond_length * direction
    outgoing_back_bond = -direction
    return new_pos, outgoing_back_bond


def build_functional_group(
    group_key: str,
    edge_carbon_pos: np.ndarray,
    outward_dir: np.ndarray,
    dihedral_offset_deg: float = 0.0,
) -> Tuple[List[Tuple[str, str, np.ndarray]], List[Tuple[int, int]], str]:
    """Construct the atoms and internal bonds of one surface functional
    group instance, attached to a given edge carbon.

    The first atom of the group is placed directly along `outward_dir`
    from the edge carbon (i.e. pointing away from the sheet interior,
    matching the reference builder's convention that "the first atom
    outside of the benzene rings lies on the edge vector"). Any further
    atoms in the group's recipe are placed using idealized bond angles
    about the preceding atom's incoming-bond axis.

    Args:
        group_key: Key into FUNCTIONAL_GROUPS (e.g. 'OH', 'COOH').
        edge_carbon_pos: (3,) Cartesian position of the edge carbon this
            group is attached to.
        outward_dir: (3,) unit vector pointing away from the sheet at this
            edge carbon (see `compute_outward_directions`).
        dihedral_offset_deg: Extra rotation (degrees) added to every
            recipe atom's `phi`, i.e. a single rigid twist of the whole
            branching substructure around each local bond axis. Passing a
            fresh random value per placement attempt gives each group
            instance an independent rotamer, avoiding perfectly identical,
            unphysically parallel orientations across many instances of
            the same group -- and, combined with a fresh value on each
            retry, gives steric-clash re-rolls (see `functionalize_edges`)
            a real chance of finding a non-clashing conformation rather
            than reproducing the same clash deterministically.

    Returns:
        Tuple of:
          - list of (element, atom_name, position) for the new atoms,
          - list of local bonds as (parent_local_index, child_local_index)
            pairs, where parent_local_index == -1 denotes a bond back to
            the edge carbon itself (to be resolved by the caller),
          - the 3-character PDB residue name for this group instance.

    Raises:
        KeyError: If `group_key` is not a recognized functional group.
    """
    spec = FUNCTIONAL_GROUPS[group_key]
    recipe = spec["recipe"]

    positions: List[Optional[np.ndarray]] = [None] * len(recipe)
    back_bonds: List[Optional[np.ndarray]] = [None] * len(recipe)
    local_bonds: List[Tuple[int, int]] = []
    atoms_out: List[Tuple[str, str, np.ndarray]] = []

    outward_dir = outward_dir / np.linalg.norm(outward_dir)

    for local_idx, (element, name, parent, length, theta, phi) in enumerate(recipe):
        if parent == -1:
            pos = edge_carbon_pos + length * outward_dir
            outgoing = -outward_dir
            local_bonds.append((-1, local_idx))
        else:
            parent_pos = positions[parent]
            parent_back = back_bonds[parent]
            effective_phi = phi + dihedral_offset_deg
            pos, outgoing = _place_substituent(parent_pos, parent_back, length, theta, effective_phi)
            local_bonds.append((parent, local_idx))

        positions[local_idx] = pos
        back_bonds[local_idx] = outgoing
        atoms_out.append((element, name, pos))

    return atoms_out, local_bonds, spec["resname"]


def build_edge_cap(
    edge_carbon_pos: np.ndarray,
    outward_dir: np.ndarray,
    tilt_deg: float = 0.0,
    tilt_azimuth_deg: float = 0.0,
) -> Tuple[str, str, np.ndarray]:
    """Construct a single valence-capping hydrogen for an edge carbon that
    was not selected for functionalization (keeps the edge chemically
    sane, analogous to peripheral C-H bonds in a PAH fragment; this is
    also the reference builder's default behavior for all edge sites).

    Args:
        edge_carbon_pos: (3,) Cartesian position of the edge carbon.
        outward_dir: (3,) unit vector pointing away from the sheet at this
            edge carbon.
        tilt_deg: Small deviation (degrees) from the pure outward
            direction. 0.0 (default) reproduces the original, purely
            radial C-H placement. A small nonzero tilt gives this
            placement a free parameter to retry with if the default
            position turns out to sterically clash with an already-placed
            neighbor (see `functionalize_edges`) -- a real C-H bond is not
            perfectly rigidly radial anyway, so a small tilt is chemically
            unremarkable.
        tilt_azimuth_deg: Azimuthal angle (degrees) about the outward-
            direction axis for the tilt direction, when `tilt_deg` != 0.

    Returns:
        Tuple of (element, atom_name, position) for the capping hydrogen.
    """
    outward_dir = outward_dir / np.linalg.norm(outward_dir)
    if tilt_deg == 0.0:
        direction = outward_dir
    else:
        u, v, w = _orthonormal_frame(outward_dir)
        theta = math.radians(tilt_deg)
        phi = math.radians(tilt_azimuth_deg)
        direction = math.cos(theta) * u + math.sin(theta) * (math.cos(phi) * v + math.sin(phi) * w)
        direction = direction / np.linalg.norm(direction)
    pos = edge_carbon_pos + EDGE_CAP_CH_BOND_LENGTH * direction
    return "H", "H1", pos


# ==============================================================================
# 5. FULL STRUCTURE ASSEMBLY
# ==============================================================================

def build_graphitic_core(
    edge_rings: int,
    layers_above_middle: int,
    spacing: float,
    bond_length: float = DEFAULT_BOND_LENGTH,
    ab_stacking: bool = True,
) -> Tuple[Structure, List[Tuple[int, int]], List[int]]:
    """Build the unfunctionalized, multi-layer graphitic core of the CQD,
    following the reference builder's exact per-layer hexagon construction
    and edge-ring decrement rule.

    Args:
        edge_rings: Edge-ring count of the middle layer (>= 1).
        layers_above_middle: Number of additional layers to build above
            the middle layer (an equal number is built below it,
            mirrored), before the edge-ring count would drop below 1 --
            whichever limit is reached first.
        spacing: Interlayer (002) spacing in Angstrom.
        bond_length: Aromatic C-C bond length in Angstrom.
        ab_stacking: If True, every other layer (by stacking order,
            starting from the middle) is registry-shifted by the basis
            vector (r2 - r1), approximating idealized Bernal (AB) graphite
            stacking (see module docstring for how this differs from a
            literal transcription of the reference tool's internal
            offset formula while producing the same physical result).

    Returns:
        Tuple of:
          - Structure populated with core carbon Atom records (resname
            'COR', one residue per layer) and their intra-layer C-C bonds,
          - a list of (start_index, end_index) atom-index ranges into
            structure.atoms for each layer (end exclusive),
          - a list of the edge-ring count actually used for each layer
            (aligned with the ranges list), for reporting purposes.

    Raises:
        ValueError: If `edge_rings` < 1.
    """
    if edge_rings < 1:
        raise ValueError(f"edge_rings must be >= 1, got {edge_rings}")

    a = bond_length
    sin60 = math.sin(math.radians(60.0))
    cos60 = math.cos(math.radians(60.0))
    ab_shift_vector = np.array([a * sin60, -a * cos60]) - np.array([0.0, 0.0])  # r2 - r1

    # Build the ordered list of (z, edge_ring_count, stack_index) layers,
    # stopping early (per side) if the ring count would fall below 1.
    layer_specs: List[Tuple[float, int, int]] = [(0.0, edge_rings, 0)]
    for m in range(1, layers_above_middle + 1):
        rings = layer_ring_count(edge_rings, m)
        if rings < 1:
            break
        layer_specs.append((m * spacing, rings, m))
        layer_specs.append((-m * spacing, rings, -m))
    layer_specs.sort(key=lambda t: t[0])

    structure = Structure()
    layer_ranges: List[Tuple[int, int]] = []
    ring_counts_used: List[int] = []

    for layer_number, (z, rings, stack_index) in enumerate(layer_specs):
        xy = generate_hexagonal_flake(rings, bond_length)
        xy = xy - xy.mean(axis=0)  # center this layer's own centroid at (0, 0)
        if ab_stacking and (stack_index % 2 != 0):
            xy = xy + ab_shift_vector

        start = len(structure.atoms)
        resseq = layer_number + 1
        for x, y in xy:
            structure.add_atom(
                Atom(
                    element="C",
                    name="C1",
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    resname="COR",
                    resseq=resseq,
                    layer_index=layer_number,
                )
            )
        end = len(structure.atoms)
        layer_ranges.append((start, end))
        ring_counts_used.append(rings)

        for i, j in find_bonded_pairs(xy, bond_length):
            structure.add_bond(start + i, start + j)

    return structure, layer_ranges, ring_counts_used


def compute_outward_directions(
    structure: Structure,
    layer_ranges: Sequence[Tuple[int, int]],
) -> Dict[int, np.ndarray]:
    """Compute an outward-pointing unit vector for every core carbon atom,
    based on its local bonding topology rather than a global radial
    reference.

    For an atom with remaining covalent neighbors, the outward direction
    is the reversed average of the unit vectors to those neighbors -- the
    standard sp2 "vacant valence" direction (e.g. an atom with two
    neighbors at ~120 deg has its missing third bond pointing exactly away
    from the average of the other two). This generalizes correctly across
    zigzag edges, armchair edges, and hexagon corners without needing a
    precomputed per-edge vector table.

    Args:
        structure: Structure whose `bonds` already encode intra-layer C-C
            connectivity.
        layer_ranges: (start, end) atom-index ranges per layer.

    Returns:
        Dict mapping atom index -> (3,) outward-pointing unit vector.
        Only computed for atoms within `layer_ranges` (core carbons).
    """
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for i, j in structure.bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    directions: Dict[int, np.ndarray] = {}
    for start, end in layer_ranges:
        for idx in range(start, end):
            atom = structure.atoms[idx]
            pos = np.array([atom.x, atom.y, atom.z])
            neighbor_unit_vectors = []
            for n_idx in adjacency.get(idx, []):
                n_atom = structure.atoms[n_idx]
                v = np.array([n_atom.x, n_atom.y, n_atom.z]) - pos
                norm = np.linalg.norm(v)
                if norm > 1e-6:
                    neighbor_unit_vectors.append(v / norm)

            if not neighbor_unit_vectors:
                directions[idx] = np.array([1.0, 0.0, 0.0])
                continue

            outward = -np.mean(neighbor_unit_vectors, axis=0)
            norm = np.linalg.norm(outward)
            if norm < 1e-6:
                # Neighbor bonds nearly cancel (rare, symmetric case):
                # fall back to a direction perpendicular to the first bond,
                # in the layer plane.
                v0 = neighbor_unit_vectors[0]
                perp = np.array([-v0[1], v0[0], 0.0])
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.array([1.0, 0.0, 0.0])
                outward = perp / np.linalg.norm(perp)
            else:
                outward = outward / norm
            directions[idx] = outward

    return directions


def compute_edge_mask(structure: Structure, layer_ranges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Flag every core carbon atom that is under-coordinated (fewer than 3
    intra-layer covalent neighbors), i.e. sits on the edge of its layer.

    Args:
        structure: Structure whose `bonds` already encode intra-layer C-C
            connectivity (as produced by `build_graphitic_core`).
        layer_ranges: (start, end) atom-index ranges per layer, used only
            to guarantee every core atom is covered even if isolated
            (degree 0, which also correctly flags as "edge").

    Returns:
        A boolean NumPy array of length len(structure.atoms) that is True
        for core carbons with fewer than 3 neighbors.
    """
    degree = defaultdict(int)
    for i, j in structure.bonds:
        degree[i] += 1
        degree[j] += 1

    n_atoms = len(structure.atoms)
    mask = np.zeros(n_atoms, dtype=bool)
    for start, end in layer_ranges:
        for idx in range(start, end):
            if degree[idx] < 3:
                mask[idx] = True
    return mask


def _candidate_has_clash(
    candidate_atoms: Sequence[Tuple[str, str, np.ndarray]],
    parent_idx: int,
    core_tree: cKDTree,
    core_positions: np.ndarray,
    core_elements: Sequence[str],
    placed_group_positions: List[np.ndarray],
    placed_group_elements: List[str],
    fraction: float,
    search_radius: float = 4.0,
) -> bool:
    """Check whether any atom of a proposed functional-group placement is
    in steric clash with the existing structure: either the static
    graphitic core (excluding the group's own attachment carbon, which is
    intentionally bonded at normal bond distance), or any previously
    accepted functional-group instance elsewhere on the sheet.

    Args:
        candidate_atoms: The proposed group's atoms, as returned by
            `build_functional_group` -- (element, atom_name, position)
            tuples.
        parent_idx: Index of the edge carbon this group is attached to
            (excluded from the core-clash check, since it is the intended
            bonded neighbor).
        core_tree: A cKDTree built once over all core-carbon positions,
            reused across every candidate check for efficiency.
        core_positions: (N, 3) array aligned with `core_tree`.
        core_elements: Element symbols aligned with `core_positions`
            (always "C" for the graphitic core, but kept generic).
        placed_group_positions: Growing list of every atom position from
            previously *accepted* group placements (mutated by the
            caller, not this function).
        placed_group_elements: Element symbols aligned with
            `placed_group_positions`.
        fraction: Fraction of summed vdW radii to use as the clash
            threshold (see `pairwise_clash_threshold`).
        search_radius: Radius (Angstrom) for the core-atom neighbor query;
            wide enough to catch any plausible clash given typical group
            sizes, tight enough to keep each query fast.

    Returns:
        True if any candidate atom is closer than its element-pair-
        specific vdW-based threshold to any existing atom.
    """
    if placed_group_positions:
        placed_arr = np.asarray(placed_group_positions)
        placed_vdw = np.array(
            [VDW_RADII_ANGSTROM.get(e, 1.5) for e in placed_group_elements]
        )
    else:
        placed_arr = None
        placed_vdw = None

    for element, _name, pos in candidate_atoms:
        cand_vdw = VDW_RADII_ANGSTROM.get(element, 1.5)

        nearby_core = core_tree.query_ball_point(pos, r=search_radius)
        for ci in nearby_core:
            if ci == parent_idx:
                continue
            d = float(np.linalg.norm(pos - core_positions[ci]))
            if d < fraction * (cand_vdw + VDW_RADII_ANGSTROM.get(core_elements[ci], 1.5)):
                return True

        if placed_arr is not None:
            dists = np.linalg.norm(placed_arr - pos, axis=1)
            thresholds = fraction * (placed_vdw + cand_vdw)
            if np.any(dists < thresholds):
                return True

    return False


def _place_edge_cap_with_retry(
    edge_carbon_pos: np.ndarray,
    outward_dir: np.ndarray,
    parent_idx: int,
    core_tree: cKDTree,
    core_positions: np.ndarray,
    core_elements: Sequence[str],
    existing_positions: List[np.ndarray],
    existing_elements: List[str],
    clash_vdw_fraction: float,
    max_attempts: int,
    rng: random.Random,
) -> Tuple[str, str, np.ndarray]:
    """Place a valence-capping hydrogen, trying the standard purely-radial
    position first and falling back to a small random tilt (with a fresh
    clash check each try) if that position clashes with an
    already-placed neighbor. Unlike functional-group placement, capping
    hydrogens have no group-identity fallback -- if every attempt still
    clashes, the last attempted position is accepted regardless, since an
    edge carbon must always end up with a complete valence.

    Args:
        edge_carbon_pos: Position of the edge carbon being capped.
        outward_dir: Its outward-pointing unit vector.
        parent_idx: Index of the edge carbon (excluded from the core
            clash check, since it is the intended bonded neighbor).
        core_tree, core_positions, core_elements: Static core-lattice
            clash-check inputs (see `_candidate_has_clash`).
        existing_positions, existing_elements: Growing lists of every
            previously-placed edge substituent atom (mutated by the
            caller after this returns).
        clash_vdw_fraction: Steric clash threshold fraction.
        max_attempts: Retries before accepting the clash as unavoidable.
        rng: Seeded random.Random for reproducible tilt sampling.

    Returns:
        Tuple of (element, atom_name, position) for the capping hydrogen.
    """
    element, name, pos = build_edge_cap(edge_carbon_pos, outward_dir)
    if not _candidate_has_clash(
        [(element, name, pos)], parent_idx, core_tree, core_positions, core_elements,
        existing_positions, existing_elements, clash_vdw_fraction,
    ):
        return element, name, pos

    for _attempt in range(max_attempts):
        tilt = rng.uniform(5.0, 20.0)
        azimuth = rng.uniform(0.0, 360.0)
        element, name, pos = build_edge_cap(edge_carbon_pos, outward_dir, tilt, azimuth)
        if not _candidate_has_clash(
            [(element, name, pos)], parent_idx, core_tree, core_positions, core_elements,
            existing_positions, existing_elements, clash_vdw_fraction,
        ):
            return element, name, pos

    return element, name, pos  # last attempt, accepted regardless (see docstring)


def functionalize_edges(
    structure: Structure,
    layer_ranges: Sequence[Tuple[int, int]],
    edge_mask: np.ndarray,
    outward_directions: Dict[int, np.ndarray],
    functional_groups: Sequence[str],
    group_ratios: Sequence[float],
    edge_coverage: float,
    allow_adjacent: bool,
    rng: random.Random,
    clash_vdw_fraction: float = DEFAULT_CLASH_VDW_FRACTION,
    max_placement_attempts: int = DEFAULT_MAX_PLACEMENT_ATTEMPTS,
) -> Dict[str, object]:
    """Passivate every edge carbon in-place, supporting a heterogeneous mix
    of functional groups (e.g. -NH2 and -COOH on the same particle, as
    real IR spectra often show).

    Runs in two passes:

    Pass A -- site selection (which edge carbons get *some* functional
    group vs. a plain hydrogen cap): identical mechanism to the
    single-group case. A random `edge_coverage` percent of edge sites are
    selected; when `allow_adjacent` is False, a site already within 2
    bonds of a previously-selected site is skipped (mechanically capping
    achievable coverage near 50%, as in the reference SI). This pass is
    entirely group-identity-agnostic.

    Pass B -- group assignment and steric placement: the selected sites
    are split among `functional_groups` according to `group_ratios` (via
    `apportion_counts`, so the realized counts always sum exactly to the
    selected-site total), then shuffled and assigned one group per site.
    Each assignment is attempted up to `max_placement_attempts` times,
    each attempt using a fresh random dihedral rotation (see
    `build_functional_group`); an attempt is accepted only if none of its
    atoms clash (element-pair vdW-based threshold, see
    `_candidate_has_clash`) with the graphitic core or any previously
    accepted group instance. If every attempt clashes, the site falls
    back to a hydrogen cap and the shortfall is recorded per-group for
    reporting -- targets are never silently satisfied by substituting a
    different, smaller group (see module discussion on mixed-group
    design decisions).

    Mutates `structure` by appending new Atom records and bonds directly.

    Args:
        structure: The (already built) core Structure; modified in place.
        layer_ranges: (start, end) atom-index ranges per layer.
        edge_mask: Boolean mask (see `compute_edge_mask`) flagging which
            core atoms are edge sites.
        outward_directions: Dict from `compute_outward_directions` mapping
            atom index to its outward-pointing unit vector.
        functional_groups: Ordered list of FUNCTIONAL_GROUPS keys to mix
            onto the selected edge sites.
        group_ratios: Relative weights, one per entry in
            `functional_groups` (need not be normalized -- see
            `apportion_counts`).
        edge_coverage: Target percentage (0-100) of edge sites to
            functionalize (with any group); the remainder are H-capped.
        allow_adjacent: Whether functionalized edge sites may be within
            2 bonds of one another, regardless of which groups occupy
            them (the constraint is identity-agnostic).
        rng: A seeded random.Random instance controlling site selection,
            group assignment, and dihedral sampling, for reproducibility.
        clash_vdw_fraction: Fraction of summed vdW radii used as the
            steric clash threshold (see `pairwise_clash_threshold`).
        max_placement_attempts: Retries per site before falling back to
            an H-cap.

    Returns:
        Dict with overall counts {'edge_sites', 'functionalized',
        'hydrogen_capped'} and a `'per_group'` dict mapping each
        functional group to {'requested', 'achieved', 'steric_fallback'}.

    Raises:
        KeyError: If any entry of `functional_groups` is not recognized.
        ValueError: If `edge_coverage` is outside [0, 100], or
            `functional_groups`/`group_ratios` lengths mismatch (raised by
            `apportion_counts`).
    """
    for g in functional_groups:
        if g not in FUNCTIONAL_GROUPS:
            raise KeyError(
                f"Unknown functional group '{g}'. Supported: {sorted(FUNCTIONAL_GROUPS)}"
            )
    if not (0.0 <= edge_coverage <= 100.0):
        raise ValueError(f"edge_coverage must be in [0, 100], got {edge_coverage}")

    core_adjacency: Dict[int, List[int]] = defaultdict(list)
    for i, j in list(structure.bonds):
        core_adjacency[i].append(j)
        core_adjacency[j].append(i)

    # Static snapshot of the core lattice for clash-checking. Safe to take
    # here because functionalize_edges is always called before any group
    # or cap atoms exist in `structure`.
    core_positions = np.array([[a.x, a.y, a.z] for a in structure.atoms])
    core_elements = [a.element for a in structure.atoms]
    core_tree = cKDTree(core_positions)

    n_layers = len(layer_ranges)
    n_edge_total = 0
    n_capped = 0
    blocked: set = set()

    # ---- Pass A: site selection (group-identity-agnostic) ----
    # Every atom added anywhere in this function (Pass A's H-caps, Pass B's
    # accepted groups, Pass B's own fallback H-caps) is tracked here so
    # later placements -- possibly at a neighboring site processed later,
    # in a different layer, or in a different pass entirely -- always
    # clash-check against everything already committed, not just against
    # Pass B's own group placements.
    existing_positions: List[np.ndarray] = []
    existing_elements: List[str] = []

    pending_sites: List[Tuple[int, int]] = []  # (atom_idx, layer_idx)
    for layer_idx, (start, end) in enumerate(layer_ranges):
        edge_indices = [idx for idx in range(start, end) if edge_mask[idx]]
        n_edge_total += len(edge_indices)
        order = list(edge_indices)
        rng.shuffle(order)

        for idx in order:
            if not allow_adjacent and idx in blocked:
                select = False
            else:
                select = rng.uniform(0.0, 100.0) < edge_coverage

            if select:
                pending_sites.append((idx, layer_idx))
                if not allow_adjacent:
                    # 2-bond exclusion radius -- see functionalize_edges
                    # docstring and module docs for why 1 bond is too
                    # narrow on a hexagonal rim.
                    blocked.add(idx)
                    one_hop = core_adjacency.get(idx, [])
                    for nbr in one_hop:
                        blocked.add(nbr)
                    for nbr in one_hop:
                        for nbr2 in core_adjacency.get(nbr, []):
                            blocked.add(nbr2)
            else:
                atom = structure.atoms[idx]
                pos = np.array([atom.x, atom.y, atom.z])
                outward = outward_directions[idx]
                element, name, apos = _place_edge_cap_with_retry(
                    pos, outward, idx, core_tree, core_positions, core_elements,
                    existing_positions, existing_elements, clash_vdw_fraction,
                    max_placement_attempts, rng,
                )
                new_idx = structure.add_atom(
                    Atom(
                        element=element, name=name,
                        x=float(apos[0]), y=float(apos[1]), z=float(apos[2]),
                        resname="COR", resseq=atom.resseq, layer_index=layer_idx,
                        charge_key=("COR", "CAP_H"),
                    )
                )
                structure.atoms[idx].charge_key = ("COR", "CAP_C")
                structure.add_bond(idx, new_idx)
                n_capped += 1
                existing_positions.append(apos)
                existing_elements.append(element)

    # ---- Pass B: group assignment + steric placement ----
    targets = apportion_counts(len(pending_sites), functional_groups, group_ratios)
    bag: List[str] = []
    for g in functional_groups:
        bag.extend([g] * targets[g])
    rng.shuffle(bag)

    achieved: Dict[str, int] = {g: 0 for g in functional_groups}
    steric_fallback: Dict[str, int] = {g: 0 for g in functional_groups}
    group_counter = 0
    n_functionalized = 0

    for (idx, layer_idx), group_key in zip(pending_sites, bag):
        atom = structure.atoms[idx]
        pos = np.array([atom.x, atom.y, atom.z])
        outward = outward_directions[idx]

        accepted = False
        for _attempt in range(max_placement_attempts):
            dihedral = rng.uniform(0.0, 360.0)
            new_atoms, local_bonds, resname = build_functional_group(
                group_key, pos, outward, dihedral_offset_deg=dihedral
            )
            if _candidate_has_clash(
                new_atoms, idx, core_tree, core_positions, core_elements,
                existing_positions, existing_elements, clash_vdw_fraction,
            ):
                continue

            resseq = n_layers + 1 + group_counter
            group_counter += 1
            offset = len(structure.atoms)
            for element, name, apos in new_atoms:
                structure.add_atom(
                    Atom(
                        element=element, name=name,
                        x=float(apos[0]), y=float(apos[1]), z=float(apos[2]),
                        resname=resname, resseq=resseq, layer_index=layer_idx,
                        charge_key=(resname, name),
                    )
                )
                existing_positions.append(apos)
                existing_elements.append(element)
            for parent_local, child_local in local_bonds:
                parent_global = idx if parent_local == -1 else offset + parent_local
                structure.add_bond(parent_global, offset + child_local)
            structure.atoms[idx].charge_key = (resname, "EDGE_C")

            achieved[group_key] += 1
            n_functionalized += 1
            accepted = True
            break

        if not accepted:
            element, name, apos = _place_edge_cap_with_retry(
                pos, outward, idx, core_tree, core_positions, core_elements,
                existing_positions, existing_elements, clash_vdw_fraction,
                max_placement_attempts, rng,
            )
            new_idx = structure.add_atom(
                Atom(
                    element=element, name=name,
                    x=float(apos[0]), y=float(apos[1]), z=float(apos[2]),
                    resname="COR", resseq=atom.resseq, layer_index=layer_idx,
                    charge_key=("COR", "CAP_H"),
                )
            )
            structure.atoms[idx].charge_key = ("COR", "CAP_C")
            structure.add_bond(idx, new_idx)
            steric_fallback[group_key] += 1
            n_capped += 1
            existing_positions.append(apos)
            existing_elements.append(element)

    per_group = {
        g: {
            "requested": targets[g],
            "achieved": achieved[g],
            "steric_fallback": steric_fallback[g],
        }
        for g in functional_groups
    }

    return {
        "edge_sites": n_edge_total,
        "functionalized": n_functionalized,
        "hydrogen_capped": n_capped,
        "per_group": per_group,
    }


# ==============================================================================
# 6. STRUCTURAL VALIDATION
# ==============================================================================

def validate_structure(structure: Structure, bond_length: float = DEFAULT_BOND_LENGTH) -> List[str]:
    """Run basic sanity checks on the assembled structure and return a list
    of human-readable warning strings (empty if everything looks sane).

    Checks performed:
      1. No two atoms are unphysically close (< 0.3 A), which would
         indicate a geometry-construction bug or an unresolved steric
         clash between independently-placed edge substituents.
      2. Every registered bond has a length within a broad chemically
         plausible window (0.6-2.2 A), catching mis-wired bond indices.

    Args:
        structure: The fully assembled Structure to validate.
        bond_length: Reference C-C bond length, used only for context in
            warning messages (not as a hard cutoff).

    Returns:
        List of warning message strings. An empty list means all checks
        passed.
    """
    warnings: List[str] = []
    coords = np.array([[a.x, a.y, a.z] for a in structure.atoms])

    if len(coords) > 1:
        tree = cKDTree(coords)
        close_pairs = tree.query_pairs(r=0.3)
        if close_pairs:
            warnings.append(
                f"{len(close_pairs)} atom pair(s) closer than 0.3 A detected "
                f"(likely independently-placed edge substituents crowding "
                f"each other on a small/densely-covered edge -- see README "
                f"'Known Limitations'; an energy minimization resolves this)."
            )

    for i, j in structure.bonds:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        if not (0.6 <= d <= 2.2):
            warnings.append(
                f"Bond between atom {i + 1} ({structure.atoms[i].element}) and "
                f"atom {j + 1} ({structure.atoms[j].element}) has an unusual "
                f"length of {d:.3f} A (expected roughly 0.9-1.8 A; reference "
                f"C-C = {bond_length:.2f} A)."
            )

    return warnings


# ==============================================================================
# 7. PDB / CHARGE OUTPUT
# ==============================================================================

def _format_atom_name(name: str, element: str) -> str:
    """Right-pad/space an atom name into the standard 4-character PDB atom
    name field, following the convention that single-letter element
    symbols leave column 13 blank (name starts at column 14).

    Args:
        name: Raw atom name (e.g. 'O1', 'H1').
        element: Element symbol used to decide field alignment.

    Returns:
        Exactly 4 characters, ready to be inserted into a PDB ATOM/HETATM
        line at columns 13-16.
    """
    if len(element) >= 2:
        return f"{name:<4}"[:4]
    if len(name) >= 4:
        return name[:4]
    return f" {name:<3}"[:4]


def write_pdb(
    filename: str,
    structure: Structure,
    remarks: Optional[List[str]] = None,
) -> None:
    """Write the assembled structure to a standards-compliant PDB file,
    including HETATM records (this is a nanomaterial cluster, not a
    biopolymer) and CONECT records encoding full covalent connectivity.

    Column layout follows the official PDB format specification:
        1-6   Record name ("HETATM")
        7-11  Atom serial number
        13-16 Atom name
        18-20 Residue name
        22    Chain identifier
        23-26 Residue sequence number
        31-38 / 39-46 / 47-54  x / y / z coordinates (8.3f)
        55-60 / 61-66          Occupancy / temperature factor (6.2f)
        77-78 Element symbol

    Args:
        filename: Output file path.
        structure: The fully assembled, edge-functionalized Structure.
        remarks: Optional list of strings to emit as leading REMARK lines
            (e.g. a summary of build parameters).

    Raises:
        ValueError: If the structure has more than 99999 atoms (exceeds
            the fixed-width PDB serial number field).
    """
    if len(structure.atoms) > 99999:
        raise ValueError(
            f"Structure has {len(structure.atoms)} atoms, exceeding the "
            f"99999-atom limit of the fixed-width PDB serial number field. "
            f"Reduce --edge_rings / --layers_above_middle or split output."
        )

    with open(filename, "w") as fh:
        if remarks:
            for line in remarks:
                fh.write(f"REMARK   {line}\n")

        for i, atom in enumerate(structure.atoms):
            serial = i + 1
            name_field = _format_atom_name(atom.name, atom.element)
            line = (
                f"HETATM"
                f"{serial:>5} "
                f"{name_field}"
                f" "
                f"{atom.resname:>3} "
                f"{atom.chain:1}"
                f"{atom.resseq:>4}"
                f"    "
                f"{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
                f"{1.00:>6.2f}{0.00:>6.2f}"
                f"          "
                f"{atom.element:>2}"
            )
            fh.write(line + "\n")

        adjacency: Dict[int, List[int]] = defaultdict(list)
        for i, j in structure.bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)

        for atom_idx in sorted(adjacency.keys()):
            partners = adjacency[atom_idx]
            for chunk_start in range(0, len(partners), 4):
                chunk = partners[chunk_start : chunk_start + 4]
                parts = "".join(f"{p + 1:>5}" for p in chunk)
                fh.write(f"CONECT{atom_idx + 1:>5}{parts}\n")

        fh.write("END\n")


def write_charge_csv(filename: str, structure: Structure) -> Dict[str, int]:
    """Write a companion CSV of partial charges for every atom whose
    `charge_key` resolves in PARTIAL_CHARGES (see module docstring for
    scope/caveats: only OH, COOH, COO, and CO groups plus plain
    edge C/H have literature values here).

    Args:
        filename: Output CSV path.
        structure: The fully assembled Structure (with `charge_key` set on
            relevant atoms by `functionalize_edges`).

    Returns:
        Dict with counts {'charged_atoms', 'uncharged_atoms'} for
        reporting.
    """
    charged = 0
    uncharged = 0
    with open(filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["serial", "resname", "atom_name", "element", "partial_charge_e", "note"])
        for i, atom in enumerate(structure.atoms):
            charge = PARTIAL_CHARGES.get(atom.charge_key) if atom.charge_key else None
            if charge is None:
                uncharged += 1
                note = "no literature value available for this group"
                charge_str = ""
            else:
                charged += 1
                note = "HF/6-31G* AMBER99SB-compatible, armchair (Paloncyova et al. 2018, Table 1)"
                charge_str = f"{charge:.3f}"
            writer.writerow([i + 1, atom.resname, atom.name, atom.element, charge_str, note])
    return {"charged_atoms": charged, "uncharged_atoms": uncharged}


# ==============================================================================
# 8. REPORTING
# ==============================================================================

def summarize_composition(structure: Structure) -> str:
    """Produce a human-readable elemental composition string, e.g.
    'C512 O48 H96 N12'.

    Args:
        structure: The assembled Structure.

    Returns:
        A space-separated "Element+Count" summary string, elements ordered
        Hill-style-ish (Carbon first, then alphabetical).
    """
    counts: Dict[str, int] = defaultdict(int)
    for atom in structure.atoms:
        counts[atom.element] += 1

    ordered_elements = ["C"] + sorted(e for e in counts if e != "C")
    return " ".join(f"{el}{counts[el]}" for el in ordered_elements)


# ==============================================================================
# 9. COMMAND-LINE INTERFACE
# ==============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Define and parse the CQD Builder command-line interface.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace with all CLI options.
    """
    group_help = ", ".join(f"{k} ({v['label']})" for k, v in FUNCTIONAL_GROUPS.items())

    parser = argparse.ArgumentParser(
        prog="cqd_builder.py",
        description=(
            "Build an edge-functionalized, multi-layer graphitic Carbon "
            "Quantum Dot (CQD) model -- sized and stacked following the "
            "VMD Carbon Dot Builder algorithm (Paloncyova et al. 2018) -- "
            "and write it to a PDB file. See the module docstring / README "
            "for exactly which parts are literal reproductions vs. "
            "verified-equivalent constructions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--edge_rings",
        type=int,
        default=6,
        help="Edge length of the middle layer, in number of benzene rings "
        "(matches the reference GUI's 'Edge length of middle layer' "
        "field). 6 reproduces the paper's flagship ~2.1 nm CD.",
    )
    parser.add_argument(
        "--layers_above_middle",
        type=int,
        default=3,
        help="Number of graphitic layers to build above the middle layer "
        "(an equal number is built below it). Reduced automatically if "
        "the edge-ring count would drop below 1 before this is reached.",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="OPTIONAL legacy convenience flag: if given, overrides "
        "--edge_rings by picking the smallest edge-ring count whose "
        "middle layer measures at least this diameter (Angstrom).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_INTERLAYER_SPACING,
        help="Interlayer spacing between stacked graphitic sheets, in "
        "Angstrom (3.4 A matches the graphitic (002) plane spacing).",
    )
    parser.add_argument(
        "--functional_groups",
        type=str,
        nargs="+",
        choices=sorted(FUNCTIONAL_GROUPS.keys()),
        default=None,
        metavar="GROUP",
        help=f"One or more surface functional groups to mix onto edge "
        f"sites (e.g. --functional_groups NH2 COOH). Supported groups: "
        f"{group_help}. If omitted, falls back to --functional_group.",
    )
    parser.add_argument(
        "--group_ratios",
        type=float,
        nargs="+",
        default=None,
        metavar="RATIO",
        help="Relative weights for --functional_groups, same order, same "
        "count (e.g. --group_ratios 1 2 1, or equivalently 25 50 25 -- "
        "ratios are normalized internally so both notations give an "
        "identical split). Omit for an equal split across the groups.",
    )
    parser.add_argument(
        "--functional_group",
        type=str,
        choices=sorted(FUNCTIONAL_GROUPS.keys()),
        default="OH",
        help="LEGACY single-group flag, kept for backward compatibility. "
        "Ignored if --functional_groups is given.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cqd_output.pdb",
        help="Output PDB filename.",
    )
    parser.add_argument(
        "--bond_length",
        type=float,
        default=DEFAULT_BOND_LENGTH,
        help="Aromatic C-C bond length used to build the hexagonal lattice, "
        "in Angstrom.",
    )
    parser.add_argument(
        "--edge_coverage",
        type=float,
        default=100.0,
        help="Percentage (0-100) of edge sites that receive ANY functional "
        "group; this pool is then split among --functional_groups per "
        "--group_ratios. Remaining edge sites are capped with H. "
        "(Equivalent to the reference GUI's 0-1 'Coverage' fraction "
        "multiplied by 100.)",
    )
    parser.add_argument(
        "--allow_adjacent",
        type=str,
        choices=["yes", "no"],
        default="yes",
        help="Whether functionalized edge sites may sit next to each "
        "other, regardless of which group(s) occupy them (matches the "
        "reference GUI's 'Edge groups can be next to each other' Y/N "
        "toggle). 'no' mechanically caps achievable coverage at roughly "
        "50%%, as noted in the reference SI.",
    )
    parser.add_argument(
        "--clash_vdw_fraction",
        type=float,
        default=DEFAULT_CLASH_VDW_FRACTION,
        help="Fraction of two atoms' summed van der Waals radii used as "
        "the non-bonded steric clash threshold during placement (default "
        "0.6, which gives ~1.8 A for O-O and scales up for bulkier pairs "
        "like S-S). Lower = looser/denser packing allowed, higher = "
        "stricter.",
    )
    parser.add_argument(
        "--max_placement_attempts",
        type=int,
        default=DEFAULT_MAX_PLACEMENT_ATTEMPTS,
        help="Retries per edge site (each with a fresh random rotamer) "
        "before falling back to a hydrogen cap when every attempt clashes "
        "sterically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling site selection, group assignment, "
        "and dihedral sampling (for reproducibility).",
    )
    parser.add_argument(
        "--no_ab_stacking",
        action="store_true",
        help="Disable idealized Bernal (AB) interlayer registry shifting; "
        "all layers are stacked in identical (AA) registry instead.",
    )
    parser.add_argument(
        "--assign_charges",
        action="store_true",
        help="Also write a companion '<output>_charges.csv' with "
        "literature partial charges (Paloncyova et al. 2018, Table 1; "
        "AMBER99SB-compatible) for OH/COOH/COO/CO groups and plain edge "
        "C/H, for whichever groups are present in the mix. NH2/SH/F atoms "
        "are left unassigned (no literature source).",
    )
    return parser.parse_args(argv)


# ==============================================================================
# 10. MAIN ORCHESTRATION
# ==============================================================================

def build_cqd(
    edge_rings: int,
    layers_above_middle: int,
    spacing: float,
    functional_groups: Sequence[str],
    group_ratios: Optional[Sequence[float]] = None,
    bond_length: float = DEFAULT_BOND_LENGTH,
    edge_coverage: float = 100.0,
    allow_adjacent: bool = True,
    seed: int = 42,
    ab_stacking: bool = True,
    clash_vdw_fraction: float = DEFAULT_CLASH_VDW_FRACTION,
    max_placement_attempts: int = DEFAULT_MAX_PLACEMENT_ATTEMPTS,
) -> Tuple[Structure, dict]:
    """High-level orchestration: build the graphitic core, detect edges,
    compute outward directions, functionalize the edges with a (possibly
    mixed) set of functional groups, and validate the result.

    Args:
        edge_rings: Edge-ring count of the middle layer.
        layers_above_middle: Number of layers to build above (and,
            mirrored, below) the middle layer.
        spacing: Interlayer spacing in Angstrom.
        functional_groups: One or more FUNCTIONAL_GROUPS keys to mix onto
            the selected edge sites.
        group_ratios: Relative weights, one per entry in
            `functional_groups`. Defaults to an equal split if omitted.
        bond_length: Aromatic C-C bond length in Angstrom.
        edge_coverage: Percentage (0-100) of edge sites functionalized
            (with any group).
        allow_adjacent: Whether functionalized sites may be adjacent,
            regardless of which group(s) occupy them.
        seed: Random seed for reproducible site selection, group
            assignment, and dihedral sampling.
        ab_stacking: Whether to apply idealized Bernal AB stacking
            registry between alternate layers.
        clash_vdw_fraction: Fraction of summed vdW radii used as the
            steric clash threshold during placement.
        max_placement_attempts: Retries per site (fresh random rotamer
            each time) before falling back to an H-cap.

    Returns:
        Tuple of (Structure, report) where `report` is a dict containing
        layer count, edge/functionalization statistics (including a
        `'per_group'` breakdown), and any structural validation warnings.
    """
    if group_ratios is None:
        group_ratios = [1.0] * len(functional_groups)

    rng = random.Random(seed)

    structure, layer_ranges, ring_counts_used = build_graphitic_core(
        edge_rings, layers_above_middle, spacing, bond_length, ab_stacking
    )
    n_core_atoms = len(structure.atoms)

    edge_mask = compute_edge_mask(structure, layer_ranges)
    outward_directions = compute_outward_directions(structure, layer_ranges)

    edge_stats = functionalize_edges(
        structure,
        layer_ranges,
        edge_mask,
        outward_directions,
        functional_groups,
        group_ratios,
        edge_coverage,
        allow_adjacent,
        rng,
        clash_vdw_fraction,
        max_placement_attempts,
    )

    warnings = validate_structure(structure, bond_length)

    report = {
        "n_layers": len(layer_ranges),
        "ring_counts_used": ring_counts_used,
        "n_core_carbons": n_core_atoms,
        "n_total_atoms": len(structure.atoms),
        "composition": summarize_composition(structure),
        "validation_warnings": warnings,
        **edge_stats,
    }
    return structure, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)

    edge_rings = args.edge_rings
    if args.diameter is not None:
        edge_rings = estimate_edge_rings_for_diameter(args.diameter, args.bond_length)
        print(
            f"[--diameter {args.diameter:.2f} A given: using the closest "
            f"matching --edge_rings {edge_rings}]"
        )

    functional_groups = args.functional_groups if args.functional_groups else [args.functional_group]
    group_ratios = args.group_ratios  # may be None -> equal split in build_cqd

    if args.group_ratios is not None and len(args.group_ratios) != len(functional_groups):
        print(
            f"ERROR: --functional_groups has {len(functional_groups)} entries "
            f"but --group_ratios has {len(args.group_ratios)}; they must match 1:1.",
            file=sys.stderr,
        )
        return 1

    try:
        structure, report = build_cqd(
            edge_rings=edge_rings,
            layers_above_middle=args.layers_above_middle,
            spacing=args.spacing,
            functional_groups=functional_groups,
            group_ratios=group_ratios,
            bond_length=args.bond_length,
            edge_coverage=args.edge_coverage,
            allow_adjacent=(args.allow_adjacent == "yes"),
            seed=args.seed,
            ab_stacking=not args.no_ab_stacking,
            clash_vdw_fraction=args.clash_vdw_fraction,
            max_placement_attempts=args.max_placement_attempts,
        )
    except (ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    group_labels = ", ".join(
        f"{g} ({FUNCTIONAL_GROUPS[g]['label']})" for g in functional_groups
    )
    remarks = [
        "Carbon Quantum Dot model generated by cqd_builder.py (v3)",
        "Lattice/stacking algorithm follows Paloncyova, Langer & Otyepka,",
        "J. Chem. Theory Comput. 2018, 14, 2076-2083 (VMD Carbon Dot Builder).",
        f"Edge rings (middle layer): {edge_rings}",
        f"Layers above middle (requested): {args.layers_above_middle}  "
        f"(actual layers built: {report['n_layers']})",
        f"Interlayer spacing: {args.spacing:.2f} A "
        f"({'AB' if not args.no_ab_stacking else 'AA'} stacking registry)",
        f"C-C bond length: {args.bond_length:.3f} A",
        f"Edge functional group(s): {group_labels}",
        f"Edge coverage: {args.edge_coverage:.1f}% target, "
        f"adjacent groups allowed: {args.allow_adjacent}, seed={args.seed}",
        f"Steric clash threshold: {args.clash_vdw_fraction:.2f} x summed vdW radii, "
        f"max {args.max_placement_attempts} placement attempts per site",
        f"Core carbons: {report['n_core_carbons']}  "
        f"Total atoms: {report['n_total_atoms']}",
        f"Composition: {report['composition']}",
    ]

    write_pdb(args.output, structure, remarks)

    charge_report = None
    if args.assign_charges:
        base, _ext = os.path.splitext(args.output)
        charge_path = f"{base}_charges.csv"
        charge_report = write_charge_csv(charge_path, structure)

    print("=" * 70)
    print(" Carbon Quantum Dot Builder -- build summary")
    print("=" * 70)
    print(f" Edge rings (middle layer)  : {edge_rings}")
    print(f" Ring counts per layer      : {report['ring_counts_used']}")
    print(f" Layers above middle        : requested {args.layers_above_middle}, "
          f"total layers built {report['n_layers']}")
    print(f" Interlayer spacing         : {args.spacing:.2f} A")
    print(f" Stacking registry          : {'AB (Bernal)' if not args.no_ab_stacking else 'AA'}")
    print(f" C-C bond length            : {args.bond_length:.3f} A")
    print(f" Core carbon atoms          : {report['n_core_carbons']}")
    print(f" Edge sites detected        : {report['edge_sites']}")
    print(f" Total functionalized       : {report['functionalized']}")
    print(" Per-group breakdown:")
    for g, stats in report["per_group"].items():
        req, ach, fallback = stats["requested"], stats["achieved"], stats["steric_fallback"]
        print(f"   - {g:<6s} requested={req:<5d} achieved={ach:<5d} "
              f"steric_fallback_to_H={fallback}")
        if req > 0 and ach < req:
            pct = 100.0 * ach / req
            print(
                f"     [Warning] Steric hindrance prevented full placement. "
                f"Achieved {pct:.0f}% of requested {g} target; "
                f"remaining sites capped with H."
            )
    print(f" Hydrogen-capped edges      : {report['hydrogen_capped']} "
          f"(unselected-for-coverage + steric-fallback combined)")
    print(f" Adjacent groups allowed    : {args.allow_adjacent}")
    print(f" Total atoms written        : {report['n_total_atoms']}")
    print(f" Elemental composition      : {report['composition']}")
    if report["validation_warnings"]:
        print(" Validation warnings:")
        for w in report["validation_warnings"]:
            print(f"   - {w}")
    else:
        print(" Structural validation      : PASSED (no anomalies detected)")
    print(f" Output written to          : {args.output}")
    if charge_report is not None:
        print(f" Partial charges written to : {os.path.splitext(args.output)[0]}_charges.csv "
              f"({charge_report['charged_atoms']} charged, "
              f"{charge_report['uncharged_atoms']} unassigned)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
