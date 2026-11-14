#!/usr/bin/env python
# coding: utf-8

# Batch-run the full EPL-derived framework across the 9 leagues
# 
# Each league runs in a fresh kernel from `full_league_analysis.ipynb`. Executed nbs + summary tables are saved separately, downloaded input files aren't modified. Test EPL first, then expand the league selection. Bookmaker comparisons need league-specific odds sources -> not automated here.

# In[1]:


from pathlib import Path
import pandas as pd
import nbformat
from nbclient import NotebookClient
from IPython.display import display

PROJECT = (Path.cwd().parent if Path.cwd().name == "src" else Path.cwd())  # project folder (run from it or from src)
TEMPLATE = PROJECT / "src" / "full_league_analysis.ipynb"
DATA_ROOT = PROJECT / "data"
RESULTS_ROOT = PROJECT / "results" / "full"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
IMAGES_ROOT = PROJECT / "images"  # every fig from every nb goes here
IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
SELECTED = ["epl", "laliga", "bundesliga", "eredivisie", "ligamx", "seriea", "brasileiro", "ligue1", "ucl"]  # ["epl"] for a quick test
BOOTSTRAPS = 1000  # final setting for the report (300 for a quick test)
assert TEMPLATE.is_file(), f"Template not found: {TEMPLATE}"
print("Selected:", SELECTED)

# In[2]:


runs=[]
for league in SELECTED:
    print("Starting",league,flush=True)
    run=nbformat.read(TEMPLATE,as_version=4)
    parameters=chr(10).join([f"LEAGUE = {league!r}", f"DATA_ROOT = {str(DATA_ROOT)!r}", f"RESULTS_ROOT = {str(RESULTS_ROOT)!r}", f"IMAGES_ROOT = {str(IMAGES_ROOT)!r}", f"B_OVERRIDE = {BOOTSTRAPS}"])
    run.cells.insert(0,nbformat.v4.new_code_cell(parameters))
    try:
        client=NotebookClient(run,timeout=1800,kernel_name="python3",resources={"metadata":{"path":str(PROJECT)}},allow_errors=False)
        client.execute()
        status="success"
        detail=""
    except Exception as exc:
        status="failed"
        detail=f"{type(exc).__name__}: {str(exc)[-1200:]}"
        print(detail[-600:])
    folder=RESULTS_ROOT/league
    folder.mkdir(parents=True,exist_ok=True)
    nbformat.write(run,folder/"executed_full_analysis.ipynb")
    runs.append({"league":league,"status":status,"details":detail})
    pd.DataFrame(runs).to_csv(RESULTS_ROOT/"run_log.csv",index=False)
    print(league,status,"| saved:",folder/"executed_full_analysis.ipynb",flush=True)
display(pd.DataFrame(runs))

# Cross-league comparisons
# 
# Run once results are produced (or after combining folders from partner). Only reads saved summaries, doesn't rerun the analyses.

# In[3]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

MIT = {
    "mit_red": "#750014",
    "red": "#A31F34",
    "light_red": "#D98C97",
    "bright_red": "#FF1423",
    "silver": "#8B959E",
    "dark_gray":"#626A73",
    "light_gray": "#B8C2CC",
    "black": "#212326",
}
MIT_CYCLE = [MIT[k] for k in ["mit_red", "silver", "light_red", "dark_gray", "red", "light_gray", "black"]]
HORIZON_COLORS = {"1d": MIT["mit_red"], "7d": MIT["silver"]}

mpl.rcParams.update({
    "axes.prop_cycle": cycler(color=MIT_CYCLE),
    "axes.edgecolor": MIT["dark_gray"], "axes.labelcolor": MIT["black"], "text.color": MIT["black"],
    "xtick.color": MIT["dark_gray"], "ytick.color": MIT["dark_gray"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": MIT["light_gray"], "grid.alpha": 0.6, "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})

LEAGUE_NAMES = {"epl": "Premier League", "laliga": "La Liga", "bundesliga": "Bundesliga", "seriea": "Serie A",
                "ligue1": "Ligue 1", "eredivisie": "Eredivisie", "ligamx": "Liga MX", "brasileiro": "Brasileirão",
                "ucl": "Champions League"}
FIG_DIR = IMAGES_ROOT

files = sorted(RESULTS_ROOT.glob("*/summary.csv"))
assert files, "No summary files yet. Run the analysis first."
combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
combined = combined.drop_duplicates(["league", "horizon"], keep="last")
combined.to_csv(RESULTS_ROOT / "all_leagues_full_comparison.csv", index=False)
display(combined.sort_values(["horizon", "league"]).round(4))

order = [l for l in LEAGUE_NAMES if l in set(combined["league"])]  # fixed order, full names
for metric, label, unit, mult in [
    ("brier_3way", "Three-outcome Brier score", "Brier score (lower is better)", 1),
    ("median_mean_spread", "Median bid-ask spread", "Percentage points", 100),
    ("draw_gap", "Draw frequency minus price-implied draw probability", "Percentage points", 100),
]:
    w = combined.pivot(index="league", columns="horizon", values=metric).reindex(order) * mult
    w.index = [LEAGUE_NAMES[l] for l in w.index]
    ax = w.plot.bar(figsize=(11, 4.5), rot=35, width=0.75,
                    color=[HORIZON_COLORS.get(h, MIT["dark_gray"]) for h in w.columns])
    ax.axhline(0, color=MIT["black"], lw=0.8)
    ax.set(title=label, xlabel="League", ylabel=unit)
    ax.grid(axis="x", visible=False)
    ax.legend(ax.containers, ["1 day" if h == "1d" else "7 days" for h in w.columns], title="Before expiration")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"compare_{metric}.{ext}")
    plt.show()

# hourly candidates use every pre-match hour -> 1d and 7d rows carry the same count, so 1 bar/league
arbs = combined.drop_duplicates("league").set_index("league")["hourly_model_fee_arbs"].reindex(order)
arbs.index = [LEAGUE_NAMES[l] for l in arbs.index]
ax = arbs.plot.bar(figsize=(11, 4.5), rot=35, width=0.6, color=MIT["mit_red"])
ax.set(title="Hourly fee-model arbitrage candidates (not confirmed executable)", xlabel="League", ylabel="Match-hours")
ax.yaxis.get_major_locator().set_params(integer=True)
ax.grid(axis="x", visible=False)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(FIG_DIR / f"compare_hourly_model_fee_arbs.{ext}")
plt.show()
print("Saved comparison table and figures:", RESULTS_ROOT.resolve(), "|", FIG_DIR.resolve())

