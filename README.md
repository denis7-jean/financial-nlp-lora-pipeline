# Financial Sentiment Analysis Pipeline with LoRA & Power BI Observability

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 📌 Executive Summary
This project implements an end-to-end **MLOps pipeline** for fine-tuning the **FinBERT** model on financial news sentiment analysis. 
Designed for resource-constrained environments, it leverages **LoRA (Low-Rank Adaptation)** to achieve **state-of-the-art accuracy (98.2%)** on consumer-grade hardware (RTX 2060), while providing enterprise-grade observability through a custom **Power BI Dashboard**.

## 🚀 Key Features

### 1. Efficient Fine-Tuning (LoRA)
- **Parameter Efficiency**: Fine-tuned only **0.6%** of total parameters (Rank=8, Alpha=16), reducing trainable weights from 110M to ~600K.
- **Hardware Optimization**: Enabled mixed-precision (**FP16**) training to fit the entire pipeline on a single 6GB GPU.
- **High Performance**: Achieved convergence in under 3 epochs with a **98.24% Test Accuracy**.

### 2. End-to-End Observability (Power BI)
Integrated a comprehensive analytics dashboard to monitor model health beyond simple metrics:
- **Error Attribution**: Automatically identifies and categorizes failures (e.g., "Misclassification" vs. "Low Confidence").
- **Confidence Calibration**: Visualizes `Prediction Confidence` vs. `Margin` to detect decision boundary issues.
- **Version Control**: Tracks performance metrics across different experiment runs (Baseline vs. V1 vs. V2).

### 3. Production-Ready Engineering
- **Robust Serialization**: Implemented a custom JSON layer to handle **NumPy type compatibility** (`int64`/`float32`), ensuring metadata is ready for downstream API consumption.
- **Automated Logging**: Real-time CSV logging of all test samples for granular error analysis.
- **Modular Design**: Decoupled architecture separating `data_loader`, `model` architecture, and `trainer` logic.

## 🛠️ System Architecture

```mermaid
graph LR
    A["Raw Data (Financial Phrasebank)"] --> B[Tokenization & Formatting]
    B --> C[FinBERT Base Model]
    C --> D{LoRA Adapter Config}
    D --> E["FP16 Training Loop (RTX 2060)"]
    F["Evaluation & Serialization"]
    G["JSON/CSV Logs"]
    H["Power BI Dashboard"]

    E --> F
    F --> G
    G --> H

```

## 📊 Experimental Results & Observability

We conducted a rigorous A/B testing framework comparing the **Zero-shot Baseline**, **LoRA V1 (Rank 8)**, and **LoRA V2 (Rank 16)** on the Financial PhraseBank dataset.

### 1. Model Performance Comparison
![Model Performance Comparison](assets/benchmark_chart.png)
*The chart below illustrates that **V1 (LoRA r=8)** achieves the best balance between accuracy and generalization. Interestingly, increasing the rank to 16 (V2) led to overfitting, confirming that a lightweight adapter is optimal for this task.*

| Experiment | Configuration | Accuracy | Macro F1 | Insight |
| --- | --- | --- | --- | --- |
| **Baseline** | Zero-shot | 97.94% | 0.9721 | Strong foundation, but struggles with edge cases. |
| **V1 (Champion)** | **LoRA r=8** | **98.24%** | **0.9788** | **Best Performance.** Successfully converted critical errors into "Low Confidence" warnings. |
| **V2** | LoRA r=16 | 97.65% | 0.9690 | **Overfitting.** High confidence but lower accuracy (diminishing returns). |

---

### 2. Deep Dive: Champion Model (V1) Analytics

*Visualizations of the best-performing model (V1) using our Power BI dashboard.*

#### A. Model Overview
![Dashboard Overview](assets/dashboard_overview.png)
*Real-time tracking of accuracy, confidence distribution, and class balance.*

#### B. Error Analysis (The "Bad Cases")
![Error Analysis](assets/dashboard_error_analysis.png)
*Drill-down into specific misclassifications. The scatter plot (Top Right) reveals "High Confidence Errors" which require data cleaning.*

## 📂 Project Structure

```bash
.
├── assets/                 # Project screenshots & benchmark charts
├── config/                 # LoRA configuration (lora_config.json)
├── src/
│   ├── data_loader.py      # Dataset preparation & tokenization
│   ├── model.py            # LoRA model initialization
│   ├── trainer.py          # Custom Hugging Face Trainer wrapper
│   └── utils.py            # Metrics (Accuracy, F1) & Logging
├── outputs/                # Experiment artifacts & logs
│   ├── baseline/           # Zero-shot baseline results
│   ├── finlora_gpu_v1/     # Champion model (Rank 8) results
│   ├── finlora_gpu_v2/     # Overfitting experiment (Rank 16) results
│   └── master_evaluation_log_v3.csv # Consolidated log for Power BI
├── train.py                # Main entry point for training
├── evaluate_baseline.py    # Zero-shot evaluation script
├── merge_logs.py           # Script to combine logs for Power BI
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## 💻 Installation & Usage

### 1. Setup Environment

```bash
# Clone the repository
git clone [https://github.com/your-username/financial-nlp-lora-pipeline.git](https://github.com/your-username/financial-nlp-lora-pipeline.git)
cd financial-nlp-lora-pipeline

# Install dependencies
pip install -r requirements.txt

```

### 2. Run Training Pipeline

To start the training process with LoRA enabled:

```bash
python train.py \
    --model_name ProsusAI/finbert \
    --data_path financial_phrasebank \
    --output_dir ./outputs/finlora_gpu_v1 \
    --batch_size 8 \
    --num_epochs 3 \
    --use_lora

```

### 3. View Analytics

Open the `.pbix` file in **Microsoft Power BI Desktop** and refresh the data source to point to your local `outputs/master_evaluation_log_v3.csv`.

## 🔮 Future Improvements

* **Model Serving**: Containerize the inference engine using **Docker** and **FastAPI**.
* **RAG Integration**: Connect the sentiment engine to a retrieval system for analyzing long-form financial reports.