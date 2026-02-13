import argparse
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from src.data_loader import prepare_data
from src.utils import compute_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load config and id2label
    print(f"Loading config from: {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    id2label = config.id2label  # ProsusAI/finbert: {0: 'positive', 1: 'negative', 2: 'neutral'}
    print(f"Model expected mapping: {id2label}")

    # 2. Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.to(device)
    model.eval()

    # 3. Load and Align Data
    print("Preparing dataset splits...")
    tokenized_ds, _ = prepare_data(tokenizer=tokenizer, max_length=128)
    test_dataset = tokenized_ds["test"]

    # --- LABEL ALIGNMENT BLOCK ---
    # Dataset (HF): 0=Neg, 1=Neu, 2=Pos
    # Model (FinBERT): 0=Pos, 1=Neg, 2=Neu
    def align_labels(example):
        mapping = {0: 1, 1: 2, 2: 0}  # Dataset_ID -> Model_ID
        example['labels'] = mapping[int(example['labels'])]
        return example

    print("Aligning labels to match FinBERT's pre-trained expectations...")
    test_dataset = test_dataset.map(align_labels)
    # -----------------------------

    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Running Baseline evaluation...")
    preds_list, labels_list, confidence_list, margin_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)

            preds_list.extend(top2_indices[:, 0].cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            confidence_list.extend(top2_probs[:, 0].cpu().numpy())
            margin_list.extend((top2_probs[:, 0] - top2_probs[:, 1]).cpu().numpy())

    # 5. Metrics
    preds_arr, labels_arr = np.array(preds_list), np.array(labels_list)
    acc = accuracy_score(labels_arr, preds_arr)
    macro_f1 = f1_score(labels_arr, preds_arr, average='macro')

    metrics = {"accuracy": acc, "macro_f1": macro_f1}
    print(f"\nFinal Baseline Metrics: {metrics}")

    # 6. Export Results
    df = pd.DataFrame({
        'text': [str(t)[:240] for t in test_dataset['sentence']],
        'true_label': labels_list,
        'pred_label': preds_list,
        'confidence': confidence_list,
        'margin': margin_list,
        'run_id': 'baseline'
    })

    df['True_Label_Name'] = df['true_label'].map(id2label)
    df['Pred_Label_Name'] = df['pred_label'].map(id2label)
    df['correct'] = (df['true_label'] == df['pred_label']).astype(int)

    def get_error_type(row):
        return 'ok' if row['correct'] == 1 else ('low_confidence' if row['confidence'] < 0.60 else 'misclassification')

    df['error_type'] = df.apply(get_error_type, axis=1)

    output_path = os.path.join(args.output_dir, "baseline_log.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--output_dir", type=str, default="./outputs/baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    evaluate(parser.parse_args())