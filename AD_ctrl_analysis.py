import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums

data_filepath_AD = r"Final_Analysis_Result_AD\Final_Analysis_Result_AD\(1)Raw data\NPX\HB00014732_AD_NPX.csv"
data_AD = pd.read_csv(data_filepath_AD)

mask_samples = data_AD["SampleType"] == "SAMPLE"
mask_assays  = data_AD["AssayType"] == "assay"

samples_long = data_AD.loc[mask_samples & mask_assays].copy()

npx_matrix = samples_long.pivot(
    index="SampleID",
    columns="OlinkID",
    values="NPX"
)

npx_matrix = npx_matrix.sort_index(axis=0)
npx_matrix = npx_matrix.sort_index(axis=1)

print(npx_matrix.shape)

plt.figure(figsize=(24, 6))
ax = sns.heatmap(
    npx_matrix,
    cmap="viridis",
    cbar_kws={"label": "NPX"},
    xticklabels=False
)
ax.set_xlabel("Assays")
ax.set_ylabel("Samples")
ax.set_title("NPX heatmap")
plt.tight_layout()
plt.show()

npx = npx_matrix

fc_threshold = 1.5

dep_results_fc = {}
comparisons_fc_only = {
    "AD-S_1pg_vs_Ctrl-S_1pg": ("AD-S_1pg", "Ctrl-S_1pg"),
    "AD-S_5pg_vs_Ctrl-S_5pg": ("AD-S_5pg", "Ctrl-S_5pg"),
    "AD-S_10pg_vs_Ctrl-S_10pg": ("AD-S_10pg", "Ctrl-S_10pg"),
    "AD_20pg_vs_Ctrl-S_20pg": ("AD_20pg", "Ctrl-S_20pg"),
    "AD-S_40pg_vs_Ctrl-S_40pg": ("AD-S_40pg", "Ctrl-S_40pg")
}

for name, (ad_group, ctrl_group) in comparisons_fc_only.items():
    fold_changes = []
    for assay in npx.columns:
        ad_vals = npx.loc[ad_group, assay]
        ctrl_vals = npx.loc[ctrl_group, assay]
        delta_npx = ad_vals - ctrl_vals
        fc = 2 ** delta_npx
        fold_changes.append(fc)
    fold_changes = np.array(fold_changes)
    deps_mask = (fold_changes >= fc_threshold) | (fold_changes <= 1 / fc_threshold)
    n_deps = deps_mask.sum()
    print(name, "=", n_deps)
    
    deps_df = pd.DataFrame({
        "Assay": np.array(npx.columns),
        "FoldChange": fold_changes,
        "DEP": deps_mask
    })

    deps_only = deps_df[deps_df["DEP"]].sort_values("FoldChange")
    print(deps_only.head())
    dep_results_fc[name] = deps_df

dep_flags = pd.DataFrame(
    {
        name: df.set_index("Assay")["DEP"]
        for name, df in dep_results_fc.items()
    }
)
dep_flags["n_comparisons"] = dep_flags.sum(axis=1)
dep_flags.to_csv("DEPs_overlap_fc_only_flags.csv")



comparisons = {
    "Ctrl_vs_L": (
        ["Ctrl-S_1pg", "Ctrl-S_5pg", "Ctrl-S_10pg", "Ctrl-S_20pg", "Ctrl-S_40pg"],
        ["L-1", "L-2"]
    )
}

ctrl_group, lys_group = comparisons["Ctrl_vs_L"]

fold_changes = []
p_values = []
assays = []

for assay in npx.columns:
    ctrl_vals = npx.loc[ctrl_group, assay].dropna()
    lys_vals  = npx.loc[lys_group, assay].dropna()
    if len(ctrl_vals) == 0 or len(lys_vals) == 0:
        continue
    delta_npx = ctrl_vals.mean() - lys_vals.mean()
    fc = 2 ** delta_npx
    fold_changes.append(fc)
    assays.append(assay)
    stat, p = ranksums(ctrl_vals, lys_vals)
    p_values.append(p)

fold_changes = np.array(fold_changes)
p_values = np.array(p_values)

deps_mask = ((fold_changes >= fc_threshold) | (fold_changes <= 1 / fc_threshold)) & (p_values <= 1)
n_deps = deps_mask.sum()

print("Ctrl-S_vs_L =", n_deps)

deps_df = pd.DataFrame({
    "Assay": np.array(assays),
    "FoldChange": fold_changes,
    "p_value": p_values,
    "DEP": deps_mask
})

deps_only = deps_df[deps_df["DEP"]].sort_values("p_value")
print(deps_only.head())

pooled_comparisons = {
    "Pooled_AD_vs_Pooled_Ctrl": (
        ["AD-S_1pg", "AD-S_5pg", "AD-S_10pg", "AD_20pg", "AD-S_40pg"],
        ["Ctrl-S_1pg", "Ctrl-S_5pg", "Ctrl-S_10pg", "Ctrl-S_20pg", "Ctrl-S_40pg"]
    )
}

ad_group_pooled, ctrl_group_pooled = pooled_comparisons["Pooled_AD_vs_Pooled_Ctrl"]

fold_changes_pooled = []
p_values_pooled = []
assays_pooled = []

for assay in npx.columns:
    ad_vals = npx.loc[ad_group_pooled, assay].dropna()
    ctrl_vals = npx.loc[ctrl_group_pooled, assay].dropna()
    if len(ad_vals) == 0 or len(ctrl_vals) == 0:
        continue
    delta_npx = ad_vals.mean() - ctrl_vals.mean()
    fc = 2 ** delta_npx
    fold_changes_pooled.append(fc)
    assays_pooled.append(assay)
    stat, p = ranksums(ad_vals, ctrl_vals)
    p_values_pooled.append(p)

fold_changes_pooled = np.array(fold_changes_pooled)
p_values_pooled = np.array(p_values_pooled)

deps_mask_pooled = ((fold_changes_pooled >= fc_threshold) | (fold_changes_pooled <= 1 / fc_threshold)) & (p_values_pooled <= 1)
n_deps_pooled = deps_mask_pooled.sum()

print("Pooled_AD_vs_Pooled_Ctrl =", n_deps_pooled)

deps_df_pooled = pd.DataFrame({
    "Assay": np.array(assays_pooled),
    "FoldChange": fold_changes_pooled,
    "p_value": p_values_pooled,
    "DEP": deps_mask_pooled
})

deps_only_pooled = deps_df_pooled[deps_df_pooled["DEP"]].sort_values("p_value")
print(deps_only_pooled.head())
