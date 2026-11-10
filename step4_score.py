"""Step 4 - score what CDVAE generated against the menu.
Run inside the CDVAE environment:  python step4_score.py <model_dir>/eval_gen.pt <model_dir>/eval_opt.pt
"""
import collections, csv, itertools, os, sys
import numpy as np

MENU = "cdvae/data/perov_5_grouped/menu.csv"
TOP_FRACTION = 0.10      # same definition of "good" as the loop
START_POINTS = 100       # CDVAE optimises 100 trajectories and stacks the snapshots in blocks of 100
ANIONS = {"O", "N", "F", "S"}
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
           "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
SHIFTS = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
# EU Critical Raw Materials Act (Regulation 2024/1252, Annex II) as chemical elements (see the sustainability cell)
EU_LIST = {"Sb", "As", "Al", "Ba", "Be", "Bi", "B", "Co", "Cu", "Ga", "Ge", "Hf", "Li", "Mg", "Mn", "Ni", "Nb", "Sc", "Si",
           "Sr", "Ta", "Ti", "W", "V", "P", "Dy", "Er", "Eu", "Gd", "Ho", "Lu", "Tb", "Tm", "Yb", "Y", "Ce", "La", "Nd",
           "Pr", "Sm", "Ir", "Pd", "Pt", "Rh", "Ru"}


def load_generated(path):
    if path.endswith(".npz"):
        return dict(np.load(path))
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                   # old torch has no weights_only argument
        return torch.load(path, map_location="cpu")


def unpack(d):
    """Cut the stacked arrays into one (frac_coords, atom_numbers, lengths, angles) tuple per crystal."""
    fc, at, ln, an, na = (np.asarray(d[k]) for k in ["frac_coords", "atom_types", "lengths", "angles", "num_atoms"])
    out = []
    for e in range(fc.shape[0]):
        start = 0
        for m, n in enumerate(na[e].astype(int)):
            out.append((fc[e, start:start + n], at[e, start:start + n].astype(int), ln[e, m], an[e, m]))
            start += n
    return out


def lattice_matrix(lengths, angles):
    a, b, c = lengths
    al, be, ga = np.radians(angles)
    bx, by = b * np.cos(ga), b * np.sin(ga)
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, 0, 0], [bx, by, 0], [cx, cy, cz]])


def near(p, q, L):
    """Shortest periodic distance (Angstrom) between two fractional points."""
    d = p - q
    d = d - np.round(d)
    return np.linalg.norm((d + SHIFTS) @ L, axis=1).min()


def classify(frac, Z, lengths, angles, menu, metals, anion_sets):
    syms = [SYMBOLS[z - 1] if 1 <= z <= len(SYMBOLS) else "?" for z in Z]
    formula = "".join(f"{s}{c}" for s, c in sorted(collections.Counter(syms).items()))
    if len(syms) != 5:
        return "wrong number of atoms", None, None, formula
    L = lattice_matrix(lengths, angles)
    dmin = min(near(frac[i], frac[j], L) for i, j in itertools.combinations(range(5), 2))
    an = [i for i, s in enumerate(syms) if s in ANIONS]
    ca = [i for i, s in enumerate(syms) if s not in ANIONS]
    if len(an) != 3 or len(ca) != 2:
        return "not 2 metals + 3 anions", None, dmin, formula
    closest = [min(near(frac[c], frac[x], L) for x in an) for c in ca]
    b = int(np.argmin(closest))                         # B site = the metal closest to the anions
    A, B = syms[ca[1 - b]], syms[ca[b]]
    X = "".join(sorted(syms[x] for x in an))
    if A not in metals or B not in metals:
        return "element outside the menu", None, dmin, formula
    if X not in anion_sets:
        return "anion mix not among the 7", None, dmin, formula
    return "on the menu", (A, B, X), dmin, formula


def report(title, samples, ctx, seconds=None):
    n = len(samples)
    menu, thr = ctx["menu"], ctx["thr"]
    print(f"\n=== {title}: {n} samples" + (f" | sampling took {seconds / 3600:.2f} h" if seconds else "") + " ===")
    for k, v in collections.Counter(s[0] for s in samples).most_common():
        print(f"  {k:<28s}{v:>6d}  ({100 * v / n:.1f}%)")
    on = [s for s in samples if s[0] == "on the menu"]
    if on:
        heat = np.array([menu[s[1]][1] for s in on])
        held = np.array([menu[s[1]][0] != "train" for s in on])
        hit = heat <= thr
        print(f"  --- the {len(on)} on-menu samples ---")
        print(f"  seen in training          : {100 * (~held).mean():5.1f}%  (pure memorising would be 100%)")
        print(f"  never seen (held-out)     : {100 * held.mean():5.1f}%  (picking blindly from the menu gives ~{100 * ctx['held_share']:.0f}%)")
        print(f"  distinct menu entries     : {len({s[1] for s in on})} | distinct held-out: "
              f"{len({s[1] for s in on if menu[s[1]][0] != 'train'})} of {ctx['n_held']}")
        se = np.sqrt(hit.mean() * (1 - hit.mean()) / len(hit))
        print(f"  hit rate (best {TOP_FRACTION:.0%})      : {100 * hit.mean():5.1f}% +/- {100 * se:.1f}  (chance {TOP_FRACTION:.0%} -> x{hit.mean() / TOP_FRACTION:.1f})")
        if held.any():
            print(f"  hit rate, held-out only   : {100 * hit[held].mean():5.1f}%")
        free = np.array([not (s[1][0] in EU_LIST or s[1][1] in EU_LIST) for s in on])
        print(f"  free of EU-list metals    : {100 * free.mean():5.1f}%  (whole menu: {100 * ctx['free_share']:.0f}%)")
        if free.any():
            print(f"  hit rate among those      : {100 * hit[free].mean():5.1f}%  = {int(hit[free].sum())} of {int(free.sum())}  (whole menu: {100 * ctx['free_hit']:.1f}%)")
        print(f"  median heat_ref           : {np.median(heat):.2f}  (whole menu {ctx['median']:.2f})")
        print(f"  atoms closer than 1 A     : {100 * np.mean([s[2] < 1.0 for s in on]):.1f}%")
    off = collections.Counter((s[3], s[0]) for s in samples if s[0] not in ("on the menu", "wrong number of atoms"))
    if off:
        print("  most common off-menu formulas (judge them with your own chemistry):")
        for (f, why), v in off.most_common(10):
            print(f"    {f:<14s} x{v:<5d} {why}")


def snapshot_table(name, blocks, ctx):
    print(f"\n=== {name}: what the optimiser does, snapshot by snapshot (100 samples each) ===")
    print("  snapshot | on the menu | usable* | hit rate of usable | usable hits / 100 | median heat_ref of usable")
    for k, b in enumerate(blocks, 1):
        on = [s for s in b if s[0] == "on the menu"]
        use = [s for s in on if s[2] is not None and s[2] >= 1.0]
        heat = np.array([ctx["menu"][s[1]][1] for s in use])
        hits = int((heat <= ctx["thr"]).sum()) if len(use) else 0
        rate = f"{100 * hits / len(use):5.1f}%" if len(use) else "   n/a"
        med = f"{np.median(heat):.2f}" if len(use) else " n/a"
        print(f"  {k:>8d} | {100 * len(on) / len(b):10.0f}% | {100 * len(use) / len(b):6.0f}% | {rate:>18s} | {hits:>17d} | {med:>25s}")
    print("  * usable = on the menu AND no two atoms closer than 1 Angstrom.  Blind picking from the menu gives ~10 hits / 100.")


def main(paths):
    menu = {}
    with open(MENU, newline="") as f:
        for r in csv.DictReader(f):
            menu[(r["A"], r["B"], r["anions"])] = (r["split_fair"], float(r["heat_ref"]))
    metals = {k[0] for k in menu} | {k[1] for k in menu}
    anion_sets = {k[2] for k in menu}
    heats = np.array([v[1] for v in menu.values()])
    n_held = sum(v[0] != "train" for v in menu.values())
    ctx = dict(menu=menu, thr=float(np.quantile(heats, TOP_FRACTION)), n_held=n_held,
               held_share=n_held / len(menu), median=float(np.median(heats)))
    free_keys = [k for k in menu if k[0] not in EU_LIST and k[1] not in EU_LIST]
    ctx["free_share"] = len(free_keys) / len(menu)
    ctx["free_hit"] = float(np.mean([menu[k][1] <= ctx["thr"] for k in free_keys]))
    print(f"menu: {len(menu)} combinations | good = heat_ref <= {ctx['thr']:.2f} | held-out by the fair split: {n_held}")
    for path in paths:
        d = load_generated(path)
        samples = [classify(*c, menu, metals, anion_sets) for c in unpack(d)]
        name = os.path.basename(path)
        secs = float(d["time"]) if "time" in d else None
        if "opt" in name:                               # snapshots are stacked in blocks of START_POINTS
            blocks = [samples[i:i + START_POINTS] for i in range(0, len(samples), START_POINTS)]
            report(f"{name} BEFORE optimisation (first snapshot)", blocks[0], ctx)
            report(f"{name} AFTER optimisation (last snapshot)", blocks[-1], ctx, secs)
            snapshot_table(name, blocks, ctx)
        else:
            report(name, samples, ctx, secs)
    print("\nthe loop for comparison: hit rate 100% (it reads the table), milliseconds, no GPU.")


if __name__ == "__main__":
    main(sys.argv[1:])
