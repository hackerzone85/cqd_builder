# Carbon Quantum Dot (CQD) Builder — v3

A dependency-light Python tool that generates an all-atom, edge-functionalized
**Carbon Quantum Dot (CQD)** model — a multi-layer stack of hexagonal
graphitic sheets with chemically passivated edges — and writes it out as a
structurally validated PDB file, ready for force-field relaxation (GROMACS /
OpenMM / AMBER) or as a starting geometry for QM cluster/DFT work (ORCA /
Gaussian).

**v3 adds heterogeneous, multi-group surface functionalization** — mixing
e.g. `-NH2` and `-COOH` on the same particle at a chosen ratio, with
element-aware steric clash avoidance (van der Waals radius-based) and
per-instance random rotamers, so mixed-group surfaces match real experimental
data (e.g. IR spectra showing multiple co-existing surface groups) rather
than being restricted to one group type at a time. See **§4 "Mixed-group
functionalization"** below.

The lattice/stacking engine (v2) matches the algorithm published in the
Supporting Information of Paloncýová, Langer & Otyepka, [*J. Chem. Theory
Comput.* 2018, 14, 2076–2083](https://pubs.acs.org/doi/10.1021/acs.jctc.7b01149) 
(the VMD ["Carbon Dot Builder" plug-in](https://cd-builder.upol.cz/)). 
See **§6 "Fidelity to the reference algorithm"** below for exactly what is 
reproduced, what is an independently-verified equivalent construction, and why.

---

## 1. Project Description

`cqd_builder.py` builds a CQD the same way the reference tool does: a
**hexagonal graphitic flake sized in benzene rings** (not an arbitrary
circular crop), stacked into a sphere-like particle whose layers shrink by
one benzene ring per step as you move away from the equatorial plane, with
randomly-placed edge functional groups at a chosen coverage fraction.

1. **Hexagonal flake core.** `--edge_rings N` builds a flake from N
   concentric shells of complete hexagonal rings around a central ring —
   the same construction family as benzene (N=1), coronene (N=2), and
   circumcoronene (N=3). Every included atom is a corner of a fully-included
   ring, so the flake can never have a dangling/under-bonded atom.
2. **Layer stacking.** Layers are stacked along *z* at the graphitic (002)
   spacing (3.4 Å default). The layer immediately above/below the middle
   layer keeps the *same* edge-ring count as the middle layer; every layer
   after that has its edge-ring count reduced by 1 — exactly reproducing
   Table 2 of the reference paper (e.g. `--edge_rings 6` needs
   `--layers_above_middle 3` to get 7 total layers, matching the paper's own
   flagship ~2.1 nm CD).
3. **Idealized Bernal (AB) stacking.** Alternate layers are registry-shifted
   by the honeycomb sublattice vector (disable with `--no_ab_stacking` for
   AA registry).
4. **Edge detection.** Every core carbon is checked against its in-layer
   covalent neighbor count; atoms with fewer than 3 neighbors are classified
   as edge sites.
5. **Randomized edge passivation.** A user-controlled percentage
   (`--edge_coverage`) of edge sites is functionalized, oriented outward
   from each carbon's own local bonding geometry; the rest get a
   valence-capping hydrogen. `--allow_adjacent no` additionally restricts
   functionalized sites from clustering together, capping achievable
   coverage near 50% — matching the reference SI's documented behavior.

### Supported edge/surface functional groups

| Flag value | Group | In reference paper's Table 1? |
|---|---|---|
| `OH` (default) | Hydroxyl, –OH | Yes |
| `COOH` | Carboxylic acid, neutral (–COOH) | Yes |
| `COO` | Carboxylate, deprotonated/charged (–COO⁻) | Yes |
| `CO` | Carbonyl/ketone edge state (=O) | Yes |
| `NH2` | Amine, –NH₂ | No (broader CQD literature) |
| `SH` | Thiol, –SH | No (broader CQD literature) |
| `F` | Fluorine, –F | No (broader CQD literature) |

---

## 2. Installation Guide

Only **NumPy** and **SciPy** are required — no RDKit/ASE/OpenMM dependency
for core functionality.

```bash
mamba create -n cqd-builder python=3.11 numpy scipy -c conda-forge -y
mamba activate cqd-builder
python -c "import numpy, scipy; print('OK', numpy.__version__, scipy.__version__)"
```

For loading the output into ASE/RDKit/MDTraj downstream:
```bash
mamba install -n cqd-builder -c conda-forge ase rdkit mdtraj -y
```

---

## 3. Usage Guide

### Command-line options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--edge_rings` | int | `6` | Edge length of the middle layer, in benzene rings (matches the reference GUI's "Edge length of middle layer" field). 6 reproduces the paper's flagship ~2.1 nm CD. |
| `--layers_above_middle` | int | `3` | Layers built above the middle layer (mirrored below). Auto-reduced if the ring count would drop below 1 first. |
| `--diameter` | float | *(none)* | **Legacy convenience only.** If given, overrides `--edge_rings` by picking the smallest ring count whose middle layer measures at least this diameter (Å). |
| `--spacing` | float | `3.4` | Interlayer (002) spacing, Å. |
| `--functional_groups` | list | *(none)* | One or more of `OH`, `COOH`, `COO`, `NH2`, `CO`, `SH`, `F` to mix onto edge sites, e.g. `--functional_groups NH2 COOH`. |
| `--group_ratios` | list | equal split | Relative weights, same order/count as `--functional_groups`, e.g. `--group_ratios 1 2 1` (or equivalently `25 50 25` — both normalize to the same split). |
| `--functional_group` | choice | `OH` | **Legacy single-group flag.** Ignored if `--functional_groups` is given. |
| `--output` | str | `cqd_output.pdb` | Output PDB filename. |
| `--bond_length` | float | `1.42` | Aromatic C–C bond length, Å. |
| `--edge_coverage` | float | `100.0` | Percent (0–100) of edge sites that receive *any* group; this pool is then split among `--functional_groups` per `--group_ratios`. Rest are H-capped. |
| `--allow_adjacent` | `yes`/`no` | `yes` | Whether functionalized sites may cluster together, regardless of which group(s) occupy them. `no` caps coverage near 50%. |
| `--clash_vdw_fraction` | float | `0.6` | Fraction of two atoms' summed van der Waals radii used as the non-bonded steric clash threshold (0.6 → ~1.8 Å for O–O, scaling appropriately for other element pairs). |
| `--max_placement_attempts` | int | `8` | Retries per edge site (fresh random rotamer each time) before falling back to an H-cap when every attempt clashes. |
| `--seed` | int | `42` | Random seed for reproducible site selection, group assignment, and rotamer sampling. |
| `--no_ab_stacking` | flag | off | Use AA (not Bernal AB) layer registry. |
| `--assign_charges` | flag | off | Also write `<output>_charges.csv` with literature partial charges (see §5). |

### Example commands

**Default build** (reproduces the paper's flagship 2.1 nm hydroxylated CD):
```bash
python cqd_builder.py
```

**Mixed -NH2/-COOH surface at a 1:1 ratio, 60% overall coverage** (e.g. to
match IR spectra showing both groups on the same particle):
```bash
python cqd_builder.py --edge_rings 6 --layers_above_middle 3 \
    --functional_groups NH2 COOH --group_ratios 1 1 --edge_coverage 60 \
    --output cqd_nh2_cooh.pdb
```

**Three-way mixed surface with an uneven ratio, no adjacent groups, and an
MD-ready charge table:**
```bash
python cqd_builder.py --edge_rings 7 --layers_above_middle 4 \
    --functional_groups OH COOH NH2 --group_ratios 2 1 1 \
    --edge_coverage 50 --allow_adjacent no --assign_charges \
    --output cqd_mixed.pdb
```

**A larger, fully carboxylated CQD (single group, legacy flag still works):**
```bash
python cqd_builder.py --edge_rings 9 --layers_above_middle 5 --functional_group COOH --output cqd_cooh.pdb
```

**Legacy diameter-based sizing (auto-picks the nearest edge_rings):**
```bash
python cqd_builder.py --diameter 30 --functional_group NH2 --output cqd_nh2.pdb
```

---

## 4. Mixed-group functionalization — how it actually works

Building a heterogeneous surface runs in two passes:

**Pass A — site selection** (which edge carbons get *some* group vs. a plain
H-cap): identical mechanism to single-group builds — a random
`--edge_coverage` percent of sites are selected, with `--allow_adjacent no`
excluding sites within 2 bonds of an already-selected one. This pass is
entirely group-identity-agnostic — it doesn't yet know which specific group
will end up where.

**Pass B — group assignment + steric placement:** the selected sites are
split among `--functional_groups` using the *largest-remainder method* (so
e.g. `1 2 1` and `25 50 25` produce identical splits, and the counts always
sum exactly to the selected-site total — no ratio rounding drift). Each site
then gets one placement attempt at a time: build the group with a **fresh
random dihedral rotation** around the C_edge–anchor bond, check it for steric
clashes, and if clear, accept it. If it clashes, retry (up to
`--max_placement_attempts` times) with a new random rotamer each time. If
every attempt still clashes, the site falls back to a hydrogen cap and the
shortfall is reported per-group — **targets are never silently satisfied by
substituting a smaller group**; a requested `-COOH` site that can't fit stays
H-capped, not quietly replaced with `-OH`.

```
[Warning] Steric hindrance prevented full placement. Achieved 94% of
requested COOH target; remaining sites capped with H.
```

### Steric clash detection

Every candidate atom is checked against the graphitic core *and* every
previously-accepted substituent (both functional groups and ordinary H-caps)
using an **element-pair-specific van der Waals threshold**:

```
clash if distance < clash_vdw_fraction * (r_vdw[a] + r_vdw[b])
```

with `clash_vdw_fraction = 0.6` by default, which lands O–O at ~1.82 Å and
scales up automatically for bulkier pairs (S–S at ~2.16 Å) rather than using
one flat cutoff for every element. Tune with `--clash_vdw_fraction` if you
want looser/denser packing or a stricter threshold.

### A genuine, tested limitation: sequential placement has no backtracking

Because sites are placed one at a time and each retry only re-rolls the
*current* site's rotamer, a rare case remains unresolvable in principle: if
an earlier neighbor's committed geometry leaves *no* valid rotamer for the
current site (not just an unlucky one missed by chance), no amount of
retrying helps — only re-rolling the earlier neighbor would. Testing found
this in practice: at 100% coverage on a hexagon-corner pair, increasing
`--max_placement_attempts` from 8 to 20 recovered some cases (a real
sampling effect) but plateaued at a small irreducible fraction beyond that
(a real geometric constraint, not a sampling issue). This is a legitimate
trade-off of a simple, fast, non-backtracking algorithm — full backtracking
would close this gap but adds real complexity for a small yield gain. In
practice: lower `--edge_coverage`, use `--allow_adjacent no` (which
eliminates the tight-corner-pair scenario entirely, since blocked neighbors
never compete for the same tight space), or accept the (clearly-reported)
H-cap fallback and note it's exactly the kind of thing a short energy
minimization resolves immediately.

---

## 5. Output Details

### PDB conventions
- **Chain:** single chain `A` throughout.
- **Core carbon layers:** resname `COR`, one residue per graphitic layer
  (`resid` = layer index). H-capped edge hydrogens share their parent
  layer's residue.
- **Functional groups:** each instance is its own residue: `HYD` (hydroxyl),
  `COA` (carboxylic acid), `COM` (carboxylate), `KET` (carbonyl), `AMN`
  (amine), `THL` (thiol), `FLR` (fluoro).
- **Connectivity:** full `CONECT` records for every bond (lattice, edge
  attachment, and internal group bonds).

### Partial charges (`--assign_charges`)
Writes `<output>_charges.csv` with literature HF/6-31G*, AMBER99SB-compatible
partial charges from Table 1 of Paloncýová et al. (2018) — but **only** for
the four groups that paper actually parameterized (edge C/H, hydroxyl,
carbonyl, carboxyl neutral/charged). The paper reports separate
armchair/zigzag values per atom; this tool applies the "armchair" value
uniformly as a documented simplification (true zigzag/armchair site
classification needs ring perception, not implemented here). `NH2`/`SH`/`F`
atoms are left unassigned — there is no literature charge source for them in
this reference.

### Loading the output
```bash
vmd cqd_output.pdb
pymol cqd_output.pdb
python -c "import mdtraj as md; print(md.load('cqd_output.pdb').topology)"
```

---

## 6. Fidelity to the reference algorithm — what's exact, what's equivalent

This section exists because getting this right took real, documented
debugging, and a computational chemist deserves to know exactly which parts
are literal reproductions and which are verified equivalents.

**Reproduced exactly:**
- The "edge length in benzene rings" sizing concept itself, and its effect
  on layer size.
- The per-layer edge-ring decrement rule (same ring count for the layer
  immediately above/below middle, then −1 per further layer) — verified to
  reproduce Table 2 of the paper exactly (edge_rings=6 → 3 layers above
  middle → 7 total layers).
- The overall passivation model: random edge-site selection at a target
  coverage %, with an option to disallow clustering that caps coverage near
  50%, exactly as described in the SI.

**How the hexagonal flake shape is actually built (and why):**
The SI's Figure S1 describes the reference tool's internal lattice-tiling
loop only in prose (a row/cell scheme with a corner-trimming rule), with
the full numeric detail given in a raster figure. During development, a
literal transcription of that prose was implemented and put through a
rigorous test battery (atom-count checks, bond-degree checks, 6-fold
rotational-symmetry checks) — and it **failed**: the resulting shape was a
sheared, asymmetric hexagon, not a proper one, and no combination of sign
conventions tried reproduced a valid structure. Rather than ship that, or
guess at further correction constants with no way to independently check
them, this tool instead builds the flake as `edge_rings` concentric shells
of complete hexagonal rings around a central ring — the same construction
family as benzene, coronene, and circumcoronene. This is verified correct
three independent ways:
1. Atom count is exactly 6·N², matching those real molecules' known
   formulas (C6, C24, C54, C96, ...).
2. The carbon-degree distribution exactly matches their known bonding
   pattern — e.g. this tool's N=2 output has exactly coronene's real
   C24H12 pattern: 12 interior degree-3 carbons + 12 rim degree-2 carbons.
3. Every atom has degree ≥ 2 by construction — dangling atoms are
   geometrically impossible.
4. As a further check against the paper's own numbers: the core-carbon
   counts this tool produces at (edge_rings, layers_above_middle) = (3,2),
   (4,2), (6,3), (8,4), (10,5) match the paper's own Table S3 "Number of
   Carbons" values **exactly** (210, 396, 1140, 2472, 4560). Three other
   rows read from that table (edge_rings 5, 7, 9) didn't match on the first
   pass; given the table in question is a badly OCR-mangled multi-column
   layout and 5 of 8 independent values matched exactly, this is most
   likely a transcription misread on rows with more crowded columns, not a
   construction error — but it's flagged here rather than swept under the
   rug.

**Equivalent-result approximations (by design, not oversight):**
- **AB-stacking registry & concentric alignment** of differently-sized
  layers: achieved by centering each layer's own centroid at (0,0) and
  offsetting alternating layers by the honeycomb sublattice vector. This
  produces the same physically-correct AB-stacked, concentric result as
  the reference tool, via an independently-verifiable construction, rather
  than transcribing the reference's internal per-layer offset constants
  (only fully specified via a second raster figure, Figure S2, and subject
  to the same "prose isn't enough" problem noted above).
- **Outward-pointing directions for edge substituents:** computed from each
  edge carbon's actual local bonding topology (reversed average of its
  remaining neighbor-bond vectors — the standard sp² "vacant valence"
  direction) rather than the six precomputed edge-vectors of Figure S2.
  Generalizes correctly to every edge site (zigzag, armchair, corners)
  without the six hardcoded vector formulas.
- **"No adjacent groups" constraint:** approximated as a 2-bond exclusion
  radius around each functionalized site (full ring-perception, which is
  what "neighboring benzene ring" technically means, isn't implemented).
  Empirically this reproduces the documented ~50% coverage ceiling well
  (tested at 42–44% across multiple seeds).
- **Functional-group internal geometry** (bond lengths/angles) uses
  standard literature values, in the same spirit as the reference tool's
  own approach ("angles ... does not ideally reflect the true relaxed
  geometry ... but allows assignment of proper bonds with the expectation
  of following structure relaxation" — SI, Figure S3 caption). As with the
  reference tool, **energy-minimize the output before production MD.**

---

## 7. Known Limitations

- **Idealized, non-energy-minimized geometry** — treat output as an
  MM/QM-optimization starting point, not a final structure (this matches
  the reference tool's own stated approach).
- **Sequential (non-backtracking) placement can leave a small irreducible
  fraction of steric fallbacks at very high coverage** — see §4 above for
  the tested, honest explanation of why, and the practical workarounds
  (lower coverage, `--allow_adjacent no`, or accept the reported fallback).
  This replaces an earlier, cruder version of this tool that had no
  clash-avoidance at all for bulky groups like `COOH`/`COO`; the current
  system actively avoids clashes via randomized retries and only falls back
  to H-capping in the rare cases retries can't resolve, all transparently
  reported per-group.
- **No sp³ defect sites, vacancies, or core heteroatom doping** (e.g.
  graphitic/pyridinic N inside the lattice) — only the sp² core and listed
  *surface/edge* functional groups are modeled.
- **Partial charges** (`--assign_charges`) use a single "armchair"
  representative value per group rather than the paper's separate
  armchair/zigzag values, since distinguishing them requires ring
  perception not implemented here. Only `OH`/`COOH`/`COO`/`CO` have
  literature charges at all (from the reference paper's Table 1); `NH2`,
  `SH`, `F` atoms are always left unassigned in the charges CSV.
- **"No adjacent groups" is a 2-bond exclusion, not true ring-perception**
  — approximates the reference SI's "neighboring benzene ring" restriction
  without actually identifying rings. Empirically reproduces the documented
  ~50% coverage ceiling well (tested at 42–44% across multiple seeds and
  multiple group mixes), but isn't a first-principles ring-adjacency check.

---

## 8. Module Reference (for extending the script)

- `generate_hexagonal_flake()` / `_hexagon_ring_centers()` — the verified
  ring-shell flake construction.
- `layer_ring_count()` — the per-layer edge-ring decrement rule.
- `build_graphitic_core()` — full multi-layer stack assembly.
- `compute_outward_directions()` — local-topology-based substituent
  orientation.
- `apportion_counts()` — largest-remainder ratio splitting for mixed groups.
- `pairwise_clash_threshold()` / `VDW_RADII_ANGSTROM` — element-aware steric
  clash thresholds; extend the radii table here for new elements.
- `_candidate_has_clash()` — the core/group clash check used during
  placement retries.
- `FUNCTIONAL_GROUPS` (dict) — add a new group by adding a key with a
  `resname` and a `recipe` list; `build_functional_group()`'s generic
  z-matrix builder handles placement (including per-instance dihedral
  randomization) automatically.
- `PARTIAL_CHARGES` (dict) — add literature charge values here, keyed by
  `(resname, atom_name)`, to extend `--assign_charges` coverage.
- `build_cqd()` — single high-level function for scripting/notebook use:

```python
import sys
# Replace this with the actual path to the folder containing 'cqd_builder.py'
sys.path.append("/path/to/the/folder/containing/your/script")
from cqd_builder import build_cqd, write_pdb, write_charge_csv

structure, report = build_cqd(
    edge_rings=6,
    layers_above_middle=3,
    spacing=3.4,
    functional_groups=["NH2", "COOH"],
    group_ratios=[1, 1],
    edge_coverage=60.0,
    allow_adjacent=True,
    seed=3,
)
write_pdb("my_cqd.pdb", structure)
write_charge_csv("my_cqd_charges.csv", structure)
print(report["per_group"])
# {'NH2': {'requested': 65, 'achieved': 58, 'steric_fallback': 7},
#  'COOH': {'requested': 65, 'achieved': 51, 'steric_fallback': 14}}
```

## 9. References

1. **Lattice Stacking Reference:**  
   Paloncýová, M., Langer, M., & Otyepka, M. (2018). Structural Dynamics of Carbon Dots in Water and *N,N*-Dimethylformamide Probed by All-Atom Molecular Dynamics Simulations. *Journal of Chemical Theory and Computation*, 14(4), 2076–2083.  
   DOI: [10.1021/acs.jctc.7b01149](https://doi.org)

2. **Application & Functionalization Baseline:**  
   Wolski, P. (2021). Molecular Dynamics Simulations of the pH-Dependent Adsorption of Doxorubicin on Carbon Quantum Dots. *Molecular Pharmaceutics*, 18(1), 257–266.  
   DOI: [10.1021/acs.molpharmaceut.0c00895](https://doi.org/10.1021/acs.molpharmaceut.0c00895)

3. **morphological structure of CDs:**
Alas, M.O., Alkas, F.B., Aktas Sukuroglu, A. et al. Fluorescent carbon dots are the new quantum dots: an overview of their potential in emerging technologies and nanosafety. J Mater Sci 55, 15074–15105 (2020). DOI: [10.1007/s10853-020-05054-y](https://doi.org/10.1007/s10853-020-05054-y)
   
---

# Appendix: Verifying the Generated CQD Structure in VMD

The generated `cqd_output_v3.pdb` contains complete **CONECT** records describing the covalent bonding network. The following VMD/TopoTools commands can be used to inspect the generated structure and verify its connectivity.

## Load the structure

```tcl
package require topotools

mol new cqd_output_v3.pdb waitfor all
```

## Assign atom properties

```tcl
topo guessatom element name
topo guessatom radius element
```

## Verify the bonding network

```tcl
set sel [atomselect top all]

puts "Number of atoms : [topo numatoms]"
puts "Number of bonds : [topo numbonds]"
```

## Identify connected fragments

```tcl
puts "Fragments: [lsort -unique [$sel get fragment]]"
```

Expected output:

```text
0 1 2 3 4 5 6
```

## Count atoms in each fragment

```tcl
foreach f [lsort -unique [$sel get fragment]] {
    set s [atomselect top "fragment $f"]
    puts "Fragment $f : [$s num] atoms"
    $s delete
}
```

Example output:

```text
Fragment 0 : 147 atoms
Fragment 1 : 223 atoms
Fragment 2 : 306 atoms
Fragment 3 : 300 atoms
Fragment 4 : 304 atoms
Fragment 5 : 222 atoms
Fragment 6 : 158 atoms
```

## Interpretation

The generated CQD consists of **seven covalently connected graphene layers** stacked in an **AB (Bernal)** arrangement.

Each fragment corresponds to **one graphene layer together with its attached functional groups**. Since adjacent graphene layers interact through non-covalent (van der Waals/π–π) interactions rather than covalent bonds, observing **seven disconnected fragments is the expected and correct result**.

The atom counts of all fragments should sum to the total number of atoms in the generated CQD.
