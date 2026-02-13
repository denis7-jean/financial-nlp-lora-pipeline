import pandas as pd
import os
import datetime

# --- Configuration ---
files = {
    "V1_LoRA": "./outputs/finlora_gpu_v1/evaluation_log.csv",
    "V2_LoRA": "./outputs/finlora_gpu_v2/evaluation_log_v2.csv",
    "Baseline": "./outputs/baseline/baseline_log.csv"
}

# Target Standard (V1/V2 Schema): 0=negative, 1=neutral, 2=positive
target_id2label = {0: "negative", 1: "neutral", 2: "positive"}

dfs = []

for version_tag, path in files.items():
    if not os.path.exists(path):
        print(f"⚠️ Skipping: File not found {path}")
        continue

    print(f"Processing: {version_tag} ...")
    df = pd.read_csv(path)

    # === 1. Special Handling for Baseline (Core Data Cleaning) ===
    if version_tag == "Baseline":
        # Baseline Original Mapping: 0:positive, 1:negative, 2:neutral
        # Target Mapping (to match V1/V2): 0:negative, 1:neutral, 2:positive
        # Mapping Logic:
        #   Old 0 (Pos) -> New 2 (Pos)
        #   Old 1 (Neg) -> New 0 (Neg)
        #   Old 2 (Neu) -> New 1 (Neu)
        baseline_mapping = {0: 2, 1: 0, 2: 1}

        # Correct the numeric IDs
        df['label'] = df['true_label'].map(baseline_mapping)
        df['pred'] = df['pred_label'].map(baseline_mapping)

        # Fill missing metadata (using correct data types)
        df['model_name'] = "ProsusAI/finbert"
        df['use_lora'] = False  # Boolean
        df['fp16'] = False      # Boolean
        df['split'] = "test"
        # Generate current time as timestamp to prevent Power BI errors
        df['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Drop old column names to avoid confusion
        df = df.drop(columns=['true_label', 'pred_label'], errors='ignore')

    # === 2. Handle V1 / V2 ===
    else:
        # V1/V2 already use standard IDs, no mapping needed
        # But ensure True_Label_Name exists for visualization
        pass

    # === 3. General Processing (All Versions) ===
    df['version_tag'] = version_tag

    # Ensure text label columns exist (True_Label_Name)
    # If column missing or has nulls, re-map from label ID to ensure 100% accuracy
    df['True_Label_Name'] = df['label'].map(target_id2label)
    df['Pred_Label_Name'] = df['pred'].map(target_id2label)

    # Ensure confidence is numeric
    if 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')

    dfs.append(df)

# === 4. Merge and Save ===
if dfs:
    master_df = pd.concat(dfs, ignore_index=True, sort=False)

    # Organize column order (for readability)
    cols = ['version_tag', 'text', 'label', 'pred', 'True_Label_Name', 'Pred_Label_Name',
            'confidence', 'margin', 'error_type', 'model_name', 'use_lora', 'timestamp']
    # Place remaining columns at the end
    existing_cols = [c for c in cols if c in master_df.columns]
    other_cols = [c for c in master_df.columns if c not in existing_cols]
    master_df = master_df[existing_cols + other_cols]

    output_path = "./outputs/master_evaluation_log_v3.csv"
    # 'utf-8-sig' ensures compatibility with Excel/Power BI on Windows
    master_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"✅ Cleaning and merging complete!")
    print(f"Label IDs standardized: 0=Negative, 1=Neutral, 2=Positive")
    print(f"Baseline metadata fixed.")
    print(f"Output file: {output_path}")
else:
    print("❌ Failed: No data merged.")