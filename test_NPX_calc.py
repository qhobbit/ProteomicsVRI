import pandas as pd
import numpy as np

data_filepath_AD = r"Final_Analysis_Result_AD\Final_Analysis_Result_AD\(1)Raw data\NPX\HB00014732_AD_NPX.csv"

def verify_olink_npx(
    csv_path,
    sample_type_sample="SAMPLE",
    sample_type_plate_control="PLATE_CONTROL",
    tol_ext=1e-6,
    tol_pc=1e-6,
    tol_c=1e-6,
):

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # 1. ExtNPX check: ExtNPX ?= log2(Count / ExtCount)
    # ------------------------------------------------------------------
    ext_counts = (
        df[df["AssayType"] == "ext_ctrl"]
        .loc[:, ["PlateID", "SampleID", "WellID", "Count"]]
        .rename(columns={"Count": "ExtCount"})
    )

    df_merged = df.merge(ext_counts, on=["PlateID", "SampleID", "WellID"], how="left")
    df_merged["ExtNPX_expected"] = np.log2(df_merged["Count"] / df_merged["ExtCount"])

    sample_mask = df_merged["SampleType"].eq(sample_type_sample)
    ext_diff = (df_merged.loc[sample_mask, "ExtNPX_expected"]
                - df_merged.loc[sample_mask, "ExtNPX"]).abs()

    print("=== 1) ExtNPX check (SampleType == 'SAMPLE') ===")
    print(ext_diff.describe())
    print(f"Proportion within {tol_ext}: {(ext_diff <= tol_ext).mean():.4f}")

    # ------------------------------------------------------------------
    # 2. Plate-control normalisation:
    #    PCNormalizedNPX ?= ExtNPX - median(ExtNPX of PLATE_CONTROL per assay)
    # ------------------------------------------------------------------
    pc_median = (
        df[df["SampleType"] == sample_type_plate_control]
        .groupby("Assay")["ExtNPX"]
        .median()
        .rename("ExtNPX_PC_median")
    )

    df_merged = df_merged.join(pc_median, on="Assay")
    df_merged["PCNormalized_expected"] = (
        df_merged["ExtNPX"] - df_merged["ExtNPX_PC_median"]
    )
    pc_diff = (df_merged["PCNormalized_expected"]
               - df_merged["PCNormalizedNPX"]).abs()

    print("\n=== 2) Plate-control normalisation check (all rows) ===")
    print(pc_diff.describe())
    print(f"Proportion within {tol_pc}: {(pc_diff <= tol_pc).mean():.4f}")

    # ------------------------------------------------------------------
    # 3. c_i constancy and reconstruction:
    #    c_i,j = NPX - PCNormalizedNPX should be constant over j for each assay i
    # ------------------------------------------------------------------
    df_merged["ci"] = df_merged["NPX"] - df_merged["PCNormalizedNPX"]

    # Per-assay summary of ci
    ci_summary = (
        df_merged.groupby("Assay")["ci"]
        .agg(["min", "max", "mean", "std", "count"])
        .rename(columns={"min": "ci_min", "max": "ci_max", "mean": "ci_mean", "std": "ci_std", "count": "n"})
    )
    ci_summary["range"] = ci_summary["ci_max"] - ci_summary["ci_min"]

    print("\n=== 3) c_i constancy check per assay (NPX - PCNormalizedNPX) ===")
    print(ci_summary[["ci_min", "ci_max", "ci_mean", "ci_std", "range"]].head(15))

    # How many assays have ci range within tolerance?
    n_assays = ci_summary.shape[0]
    n_good = (ci_summary["range"] <= tol_c).sum()
    print(f"\nAssays with max(ci) - min(ci) <= {tol_c}: {n_good} / {n_assays}")

    print("Summary of ci ranges:")
    print(ci_summary["range"].describe())

    # ------------------------------------------------------------------
    # 4. Optional: reconstruction check NPX_recon = PCNormalizedNPX + ci_mean
    # ------------------------------------------------------------------
    # Use per-assay ci_mean as the calibration constant and reconstruct NPX
    df_merged = df_merged.join(
        ci_summary["ci_mean"].rename("c_i_mean"), on="Assay"
    )
    df_merged["NPX_reconstructed"] = (
        df_merged["PCNormalizedNPX"] + df_merged["c_i_mean"]
    )

    recon_diff = (df_merged["NPX_reconstructed"] - df_merged["NPX"]).abs()
    print("\n=== 4) Reconstruction check: NPX ?= PCNormalizedNPX + c_i ===")
    print(recon_diff.describe())
    print(f"Proportion within {tol_c}: {(recon_diff <= tol_c).mean():.4f}")
    # ============================
    # 5. Check relation between c_i and negative controls
    # ============================

    # Identify negative controls
    neg_mask = df_merged["SampleType"] == "NEGATIVE_CONTROL"
    neg_df = df_merged[neg_mask]

    # Per-assay negative control medians
    neg_stats = (
        neg_df.groupby("Assay")["NPX"]
        .agg(neg_median_NPX="median", neg_mean_NPX="mean", neg_min="min", neg_max="max")
    )

    # Combine with c_i summary
    ci_compare = ci_summary.join(neg_stats, how="left")

    # Compute differences
    ci_compare["c_minus_negMedian"] = ci_compare["ci_mean"] - ci_compare["neg_median_NPX"]
    ci_compare["c_minus_negMean"] = ci_compare["ci_mean"] - ci_compare["neg_mean_NPX"]

    print("\n=== 5) Relationship between c_i and Negative Control NPX values ===")
    print(ci_compare[["ci_mean", "neg_median_NPX", "neg_mean_NPX", 
                    "c_minus_negMedian", "c_minus_negMean"]].head(15))

    print("\nSummary of c_i - median(NegCtrl NPX):")
    print(ci_compare["c_minus_negMedian"].describe())

    print("\nSummary of c_i - mean(NegCtrl NPX):")
    print(ci_compare["c_minus_negMean"].describe())

    # OPTIONAL: correlation check
    print("\nCorrelation between c_i and negative control medians:")
    print(ci_compare[["ci_mean", "neg_median_NPX"]].corr())


    return df_merged, ci_summary



# Run
df_checked, ci_summary = verify_olink_npx(data_filepath_AD)
