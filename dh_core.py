"""Definitions for the dh single-locus RNAi notebook (model wrapper, simulation, plot helpers).

Loaded by the notebook with `%run -i dh_lib/dh_core.py`, which executes this file inside the notebook's own
namespace. It therefore uses the settings of the first code cell (SAVE_DIR, PARAM_SETS, TGENS, ...) and defines
GENOS, run_locus, run_loci, final_states, draw_trace, style_trace_axes, save_fig, RC, ... for every later cell.
After changing PARAM_SETS / SHARED, run the settings cell and then the cell that loads this file again.
The settings (the first code cell) hold everything you edit; this file holds only definitions.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def save_fig(fig, name, tight=True):
    """Save in every format of SAVE_FORMATS, transparent background. tight=False keeps the fixed canvas."""
    for ext in SAVE_FORMATS:
        fig.savefig(OUTDIR / ("%s.%s" % (name, ext)), dpi=SAVE_DPI,
                    bbox_inches="tight" if tight else None, transparent=True)


# ---- GillesPy2 wrapper ----
from gillespy2 import Model, Species, Parameter, Reaction, SSACSolver

PARAM_NAMES = "T k1 r1 d1 g kS d2 nu e_rw e_nuc d3 Nn mu".split()

class GP2Locus:
    """Builds the GillesPy2 model from `reactions` and compiles the SSA solver once."""

    def __init__(self, reactions, N, DIV, SHARED, PARAM_SETS, new_seed):
        self.N, self.DIV, self.SHARED, self.PARAM_SETS = N, DIV, SHARED, PARAM_SETS
        self.mu, self.new_seed = np.log(2.0) / DIV, new_seed
        mdl = Model(name="dh_locus")
        mdl.add_species([Species(n, initial_value=0, mode="discrete") for n in ("R", "S", "m")])
        for k in PARAM_NAMES:
            mdl.add_parameter(Parameter(name=k, expression=1))          # set per run
        for name, re, pr, f in reactions:
            mdl.add_reaction(Reaction(name=name, reactants=re, products=pr, propensity_function=f))
        self.mdl, self.solver = mdl, SSACSolver(model=mdl, variable=True)

    def _variables(self, sname, geno, ic):
        ic = (0, 20, int(0.9 * self.N)) if ic is None else ic
        P = {**self.SHARED, **self.PARAM_SETS[sname]}
        _, gs, uw, un, d3, _ = geno
        return dict(T=P["T"], k1=P["K1"], r1=P["R1"], d1=P["D1"], g=P["GAM"] * gs, kS=P["KS"],
                    d2=P["D2"], nu=P["NU"], e_rw=P["E_RW"] * uw, e_nuc=P["E_NUC"] * un,
                    d3=d3, Nn=self.N, mu=self.mu, R=ic[0], S=ic[1], m=ic[2])

    def _run(self, var, tmax, increment, ntraj, seed):
        self.mdl.tspan = None; self.solver.model.tspan = None          # time grid set per run
        out = self.mdl.run(solver=self.solver, t=tmax, increment=increment,
                           number_of_trajectories=ntraj, seed=seed, variables=var)
        return [out] if isinstance(out, dict) else out                 # a single Result -> list

    def run_loci(self, sname, geno, gens, nrec, nloci=1, seed=None, ic=None):
        """Traces of nloci independent loci. Returns R, M (fraction), S summed over the loci, and the seed."""
        seed = self.new_seed() if seed is None else seed
        out = self._run(self._variables(sname, geno, ic), gens * self.DIV, gens * self.DIV / nrec, nloci, seed)
        R = sum(o["R"][:nrec] for o in out); S = sum(o["S"][:nrec] for o in out)
        m = sum(o["m"][:nrec] for o in out)
        return R, m / self.N, S, seed

    def run_locus(self, sname, geno, gens, nrec, seed=None, ic=None):
        return self.run_loci(sname, geno, gens, nrec, 1, seed, ic)

    def final_states(self, sname, geno, gens, ntraj, seed=None, ic=None):
        """Last R, S, m of ntraj independent loci run for `gens` generations."""
        seed = self.new_seed() if seed is None else seed
        tmax = gens * self.DIV
        out = self._run(self._variables(sname, geno, ic), tmax, tmax, ntraj, seed)    # records t = 0 and t = tmax
        R = np.array([o["R"][-1] for o in out]); S = np.array([o["S"][-1] for o in out])
        m = np.array([o["m"][-1] for o in out])
        return R, S, m, seed



D3_OE = 2.5                 # epe1-OE demethylation rate (matches the dh epe1-OE mean R)

# ---- genotypes: (label, GAM scale, E_RW scale, E_NUC scale, D3, colour) ----
GENOS = [
    ("WT",                 1, 1, 1, SHARED["D3"], "#595959"),
    (r"$\it{clr4}\Delta$", 1, 0, 0, SHARED["D3"], "#f06a00"),   # no methylation
    (r"$\it{ago1}\Delta$", 0, 0, 0, SHARED["D3"], "#8e7cc3"),   # GAM = E_RW = E_NUC = 0
    (r"$\it{clr4}$-W31G",  1, 0, 1, SHARED["D3"], "#00aeef"),   # nucleation only
    (r"$\it{epe1}\Delta$", 1, 1, 1, 0.0,         "#7b2fbe"),   # D3 = 0
    (r"$\it{epe1}$-OE",    1, 1, 1, D3_OE,        "#1a64d1"),   # D3 = D3_OE
]
PANEL_COLORS = {"WT": "#595959", "clr4_del": "#f06a00", "ago1_del": "#939598", "clr4_W31G": "#00aeef"}
GENO_KEY = dict(zip([g[0] for g in GENOS], ["WT", "clr4_del", "ago1_del", "clr4_W31G", "epe1_del", "epe1_OE"]))  # short names

# ---- random seeds (new on every run) ----
_rng = np.random.default_rng()
def new_seed():
    return int(_rng.integers(1, 2**31 - 1))


# ---- model: (name, reactants, products, propensity) ----
# M = m/Nn (methylated fraction), S/(kS+S) = siRNA saturation.  Parameters: T k1 r1 d1 g kS d2 nu e_rw e_nuc d3 Nn mu
REACTIONS = [
    ("transcription", {},       {"R": 1},  "T/(1+pow((m/Nn)/k1,r1))"),
    ("R_decay",       {"R": 1}, {},        "d1*R"),
    ("slicing",       {"R": 1}, {"S": 10}, "g*R*(m/Nn)*(S/(kS+S))"),               # R -> NS siRNAs
    ("priRNA",        {"R": 1}, {"S": 1},  "nu*R"),
    ("S_decay",       {"S": 1}, {},        "d2*S"),
    ("methylation",   {},       {"m": 1},  "Nn*R*(S/(kS+S))*(e_rw*(m/Nn)+e_nuc)*(1-(m/Nn))"),
    ("Epe1_demeth",   {"m": 1}, {},        "Nn*d3*(m/Nn)"),
    ("dilution_R",    {"R": 1}, {},        "mu*R"),
    ("dilution_S",    {"S": 1}, {},        "mu*S"),
    ("dilution_m",    {"m": 1}, {},        "mu*m"),
]
_gp2 = GP2Locus(REACTIONS, N, DIV, SHARED, PARAM_SETS, new_seed)

def run_locus(sname, geno, gens, nrec, seed=None, ic=(0, 20, int(0.9 * N))):
    """One run of one locus. Returns R, M (fraction), S and the seed."""
    return _gp2.run_locus(sname, geno, gens, nrec, seed, ic)

final_states = _gp2.final_states        # (set, genotype, gens, ntraj) -> last R, S, m

def run_loci(sname, geno, gens, nrec, nloci, seed=None, ic=(0, 20, int(0.9 * N))):
    """nloci independent loci; returns R, M (fraction), S summed over the loci."""
    return _gp2.run_loci(sname, geno, gens, nrec, nloci, seed, ic)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
def pub_rc(fontsize=9, font="DejaVu Sans"):
    """Publication style used by every figure (plt.rc_context(RC))."""
    return {
        "figure.dpi": 150, "font.family": font,
        "font.size": fontsize, "axes.labelsize": fontsize,
        "axes.titlesize": fontsize + 1, "legend.fontsize": fontsize - 1,
        "xtick.labelsize": fontsize - 1, "ytick.labelsize": fontsize - 1,
        "axes.linewidth": 0.9, "lines.linewidth": 1.6,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.constrained_layout.use": False,
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "text.color": "black",
        "svg.fonttype": "none",                                   # live text
        "figure.facecolor": "none", "axes.facecolor": "none",     # transparent
        "savefig.facecolor": "none", "savefig.edgecolor": "none", "savefig.transparent": True,
    }
RC = pub_rc(9)


# ---- trace format: shared by every time trace ----
def cut_burnin(y, gens_total):
    """Drop the first TRACE_BURNIN generations of a trace simulated for gens_total generations."""
    y = np.asarray(y)
    return y[int(round(TRACE_BURNIN * len(y) / gens_total)):]

def draw_trace(ax, t, y, color, lw=None, alpha=1.0, label=None):
    """Raw counts as steps (the Gillespie state is piecewise constant)."""
    return ax.step(t, y, where="post", color=color, lw=TRACE_LW if lw is None else lw, alpha=alpha,
                   rasterized=TRACE_RASTERIZED, label=label)

def style_trace_axes(ax, xlim=None, ylim=None, xlabel=None, ylabel=None, yticks=None):
    """Limits, labels and ticks of a trace (ticks outside the limits are dropped)."""
    ax.margins(x=0)
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xticks([t for t in ax.get_xticks() if x0 - 1e-9 <= t <= x1 + 1e-9])
    ax.set_yticks([t for t in ax.get_yticks() if y0 - 1e-9 <= t <= y1 + 1e-9])
    if yticks is not None: ax.set_yticks(yticks)             # after the filter above


def make_res():
    """Histogram data: many short runs, last value of each (section 1)."""
    res = {q: {} for q in ("R", "S", "M")}          # res[quantity][(set, genotype label)] -> final value of every run
    for sname in PARAM_SETS:
        nl = NLOCI_SUM[sname]
        print(sname, "(sum of %d locus/loci), %d runs x %d generations" % (nl, NRUNS, RUN_GENS))
        for geno in GENOS:
            R, S, m, _ = final_states(sname, geno, RUN_GENS, NRUNS * nl)
            res["R"][(sname, geno[0])] = R.reshape(NRUNS, nl).sum(axis=1)
            res["S"][(sname, geno[0])] = S.reshape(NRUNS, nl).sum(axis=1)
            res["M"][(sname, geno[0])] = np.round(m).reshape(NRUNS, nl).sum(axis=1)   # methylated nucleosomes (whole numbers)
            print("  %-18s  RNA %6.2f   siRNA %7.1f   methylation fraction %5.2f" % (
                geno[0], res["R"][(sname, geno[0])].mean(), res["S"][(sname, geno[0])].mean(),
                res["M"][(sname, geno[0])].mean() / (N * nl)))
    return res


def load_exp_data():
    """Experimental RNA-FISH spot counts per cell (section 1b); files and folders are set in the settings cell."""
    import pandas as pd
    exp_data = {}                                   # (set, genotype) -> spots per cell
    if SHOW_EXPERIMENT:
        for sname, files in EXP_FILES.items():
            for geno in GENOS:
                if GENO_KEY[geno[0]] not in files: continue
                fname, col = files[GENO_KEY[geno[0]]]
                path = EXP_DIRS[sname] / fname
                if not path.exists():
                    print("experimental file not found, skipped:", path); continue
                v = pd.read_csv(path)[col].dropna().to_numpy()
                exp_data[(sname, geno[0])] = np.round(v).astype(int)
                print("%-11s %-18s %-45s n = %4d cells, mean = %5.2f" % (sname, geno[0], fname, len(v), v.mean()))
    return exp_data


def sim_panel_trace(sname, geno):
    """One new R, S, M trace (burn-in removed; M as a fraction) for a set and genotype: the entry stored in trace_store."""
    nl = NLOCI_SUM[sname]
    R, M, S, seed = run_loci(sname, geno, TGENS + TRACE_BURNIN, (TGENS + TRACE_BURNIN) * 160, nl)
    R, M, S = (cut_burnin(v, TGENS + TRACE_BURNIN) for v in (R, M, S))
    return dict(R=R, S=S, M=M / nl, seed=seed)

def need(name, hint):
    """Stop with a clear message when a settings cell has not been run."""
    if name not in globals(): raise RuntimeError("%s is not defined: %s" % (name, hint))


def plot_trace_panel(keys, basename):
    """R, S, M traces (top to bottom) of the genotypes in `keys`, overlaid."""
    with plt.rc_context(RC):
        fig = plt.figure(figsize=PANEL_FIGSIZE); fig.set_layout_engine(None)
        for i, q in enumerate("RSM"):
            ax = fig.add_axes(PANEL_RECTS[i])
            for k in keys:
                y = trace_store[(PANEL_SET, PANEL_LABEL[k])][q]
                t = np.arange(len(y)) * TGENS / len(y)
                draw_trace(ax, t, y, PANEL_COLORS[k], lw=TRACE_LW_PUB, label=PANEL_LABEL[k])
            style_trace_axes(ax, xlim=PANEL_XLIM, ylim=PANEL_YLIM[q], ylabel=PANEL_YLAB[q],
                             xlabel="time (generations)" if i == 2 else None, yticks=PANEL_YTICK[q])
            if i < 2: ax.tick_params(labelbottom=False)
            ax.yaxis.set_label_coords(-0.115, 0.5)           # aligned y-labels
            if i == 0 and len(keys) > 1: ax.legend(frameon=False, loc="upper right", handlelength=1.5, ncol=len(keys))
    save_fig(fig, basename, tight=False)
    return fig
