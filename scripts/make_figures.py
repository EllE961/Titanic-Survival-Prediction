"""Cross-validate the Titanic model against baselines and write the README figures.

Run from the repo root:  python scripts/make_figures.py
Reads data/train.csv, prints every number quoted in the README, writes docs/assets/*.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

BG, PANEL = "#0d0c0b", "#131211"
INK, INK2, INK3 = "#ede7de", "#aaa299", "#7a736c"
GRID = "#24221f"
EMBER, TEMPER, PURPLE, GREEN = "#d95926", "#3987e5", "#9085e9", "#199e70"
SERIES = [EMBER, TEMPER, PURPLE, GREEN, "#e8b44c", "#c0509c"]
HEAT = LinearSegmentedColormap.from_list("heat", ["#141110", "#2a1410", "#4b170c", "#7a1f0a", "#ad320c",
                                                  "#d9520f", "#f07a1e", "#f9a63b", "#fccf6b", "#fff0c2"])
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlesize": 12.5, "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "axes.axisbelow": True, "font.family": ["Segoe UI", "DejaVu Sans"], "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK2,
    "figure.dpi": 110, "savefig.dpi": 200,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]  # same features as Titanic-Survival-Prediction.py


def encode(frame):
    X = pd.get_dummies(frame[FEATURES])
    return X.reindex(columns=["Pclass", "SibSp", "Parch", "Sex_female", "Sex_male"], fill_value=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote docs/assets/{name}")


def label_bars(ax, bars, fmt="{:.0%}", dy=0.02):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom",
                color=INK2, fontsize=8.5)


def eda_figure(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [0.8, 1.25, 1.35]})

    ax = axes[0]
    counts = df["Survived"].value_counts().sort_index()
    bars = ax.bar(["died", "survived"], counts.values, color=[INK3, EMBER], width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v}  ({v / len(df):.1%})", ha="center", va="bottom",
                color=INK2, fontsize=8.5)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Outcome (891 passengers)")
    ax.set_ylabel("passengers")

    ax = axes[1]
    rate = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
    x = np.arange(3)
    w = 0.38
    b1 = ax.bar(x - w / 2, rate["female"], w, color=EMBER, label="female")
    b2 = ax.bar(x + w / 2, rate["male"], w, color=TEMPER, label="male")
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.set_xticks(x, ["1st class", "2nd class", "3rd class"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Survival rate by class and sex")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[2]
    fam = (df["SibSp"] + df["Parch"]).clip(upper=6)
    g = df.groupby(fam)["Survived"].agg(["mean", "size"])
    bars = ax.bar(g.index, g["mean"], color=PURPLE, width=0.65)
    for b, (m, n) in zip(bars, g.values):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.0%}", ha="center", va="bottom", color=INK2, fontsize=8.5)
        ax.text(b.get_x() + b.get_width() / 2, 0.025, f"n={int(n)}", ha="center", va="bottom", color=BG, fontsize=7.5)
    ax.set_xticks(g.index, ["0", "1", "2", "3", "4", "5", "6+"])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("relatives aboard (SibSp + Parch)")
    ax.set_title("Survival rate by family size")
    fig.tight_layout(w_pad=2.5)
    save(fig, "eda.png")


def model_figure(names, scores):
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    means = np.array([s.mean() for s in scores])
    stds = np.array([s.std() for s in scores])
    y = np.arange(len(names))[::-1]
    colors = [INK3, INK3, TEMPER, EMBER, PURPLE]
    ax.barh(y, means, xerr=stds, color=colors, height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
    for yi, m, s in zip(y, means, stds):
        ax.text(m + s + 0.012, yi, f"{m:.3f} ± {s:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (mean ± std over 50 folds)")
    ax.set_title("Cross-validated accuracy, same four features for every model")
    fig.text(0.012, -0.015, "Repeated stratified 10-fold CV (5 repeats) on the 891 labelled passengers.",
             color=INK3, fontsize=8.5, ha="left", va="top")
    save(fig, "models.png")


def grouped_permutation_importance(df, y, model, n_shuffles=5, seed=0):
    """Out-of-fold accuracy drop when one raw feature (both Sex dummies together) is shuffled."""
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    drops = {f: [] for f in FEATURES}
    for tr, te in skf.split(df, y):
        m = model.fit(encode(df.iloc[tr]), y.iloc[tr])
        test = df.iloc[te]
        base = (m.predict(encode(test)) == y.iloc[te].values).mean()
        for f in FEATURES:
            for _ in range(n_shuffles):
                shuffled = test.copy()
                shuffled[f] = rng.permutation(shuffled[f].values)
                drops[f].append(base - (m.predict(encode(shuffled)) == y.iloc[te].values).mean())
    return {f: (np.mean(v), np.std(v)) for f, v in drops.items()}


def confusion_importance_figure(cm, imp):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    ax.imshow(cm, cmap=HEAT, vmin=0, vmax=cm.max())
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n{v / total:.1%}", ha="center", va="center", fontsize=11,
                    color=BG if v > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["died", "survived"])
    ax.set_yticks([0, 1], ["died", "survived"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Random forest, out-of-fold predictions")

    ax = axes[1]
    order = sorted(imp, key=lambda f: imp[f][0])
    vals = [imp[f][0] for f in order]
    errs = [imp[f][1] for f in order]
    y = np.arange(len(order))
    ax.barh(y, vals, xerr=errs, color=[EMBER if f == "Sex" else TEMPER for f in order], height=0.6,
            error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 3})
    for yi, v, e in zip(y, vals, errs):
        ax.text(max(v, 0) + e + 0.006, yi, f"{v:+.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, order)
    ax.set_xlim(min(0, min(vals) - 0.01), max(vals) + max(errs) + 0.07)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy drop when the feature is shuffled")
    ax.set_title("Permutation importance (held-out folds)")
    fig.tight_layout(w_pad=3)
    save(fig, "confusion_importance.png")


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    y = df["Survived"]
    X = encode(df)
    print(f"passengers: {len(df)}, survived: {y.sum()} ({y.mean():.3f}), distinct feature rows: "
          f"{len(X.drop_duplicates())}")

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)  # the repo's model
    models = {
        "Majority class (all died)": DummyClassifier(strategy="most_frequent"),
        "Sex only (depth-1 tree)": DecisionTreeClassifier(max_depth=1),
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest (this repo)": rf,
        "Gradient boosting": GradientBoostingClassifier(random_state=0),
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    scores = []
    for name, m in models.items():
        s = cross_val_score(m, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        scores.append(s)
        print(f"{name:28s} {s.mean():.4f} +/- {s.std():.4f}")

    rf_train = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1).fit(X, y)
    print(f"random forest training accuracy: {rf_train.score(X, y):.4f}")

    oof = cross_val_predict(rf, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0))
    cm = confusion_matrix(y, oof)
    tn, fp, fn, tp = cm.ravel()
    print(f"out-of-fold: accuracy {(tn + tp) / cm.sum():.4f}, survivor recall {tp / (tp + fn):.4f}, "
          f"precision {tp / (tp + fp):.4f}, confusion {cm.tolist()}")
    missed = df[(y == 1) & (oof == 0)]
    false_alarm = df[(y == 0) & (oof == 1)]
    print(f"missed survivors who were men: {(missed['Sex'] == 'male').sum()} of {len(missed)}; "
          f"false alarms who were 3rd-class women: "
          f"{((false_alarm['Sex'] == 'female') & (false_alarm['Pclass'] == 3)).sum()} of {len(false_alarm)}")
    print(f"forest agrees with 'women survive, men die' on {(oof == (df['Sex'] == 'female')).mean():.3f} of passengers")

    imp = grouped_permutation_importance(df, y, RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1))
    for f, (m, s) in sorted(imp.items(), key=lambda kv: -kv[1][0]):
        print(f"permutation importance {f:7s} {m:+.4f} +/- {s:.4f}")

    # how much the unused columns would add, same forest settings
    ext = df.copy()
    ext["Title"] = ext["Name"].str.extract(r",\s*([^\.]+)\.")[0]
    ext["Title"] = ext["Title"].where(ext["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Other")
    Xe = pd.get_dummies(ext[FEATURES + ["Age", "Fare", "Embarked", "Title"]])
    ext_model = make_pipeline(SimpleImputer(strategy="median"),
                              RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1))
    s = cross_val_score(ext_model, Xe, y, cv=cv, n_jobs=-1)
    print(f"same forest + Age, Fare, Embarked, Title: {s.mean():.4f} +/- {s.std():.4f}")

    eda_figure(df)
    model_figure(list(models), scores)
    confusion_importance_figure(cm, imp)


if __name__ == "__main__":
    main()
