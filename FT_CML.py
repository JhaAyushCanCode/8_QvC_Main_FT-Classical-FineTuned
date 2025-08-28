# fine_tuned_transformer_improved.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed
)
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import os
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 28
BATCH_SIZE = 32
MAX_LENGTH = 256
LEARNING_RATE = 2e-6
NUM_EPOCHS = 2000
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
LOGGING_DIR = "./logs"
OUTPUT_DIR = "./bert_finetuned"
SEED = 42

# GoEmotions label names (actual emotion labels)
GOEMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility (fixed implementation)
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,  # Use dynamic padding with DataCollator
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

class WeightedTrainer(Trainer):
    """Custom Trainer with weighted BCE loss for class imbalance handling."""
    
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight
        if pos_weight is not None:
            self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fct = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Compute metrics with version-safe handling and comprehensive metrics."""
    # Handle different HuggingFace versions
    if hasattr(eval_pred, "predictions"):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        logits, labels = eval_pred
    
    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))
    
    # Binary predictions with 0.5 threshold (will be optimized later)
    preds = (probs >= 0.5).astype(int)
    
    # Compute comprehensive metrics
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    # Per-label metrics for analysis
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mean_precision": np.mean(precision),
        "mean_recall": np.mean(recall),
    }

def optimize_thresholds(trainer, val_dataset, num_labels):
    """Optimize thresholds on validation set to maximize macro F1."""
    logger.info("Optimizing thresholds on validation set...")
    
    val_predictions = trainer.predict(val_dataset)
    val_probs = 1 / (1 + np.exp(-val_predictions.predictions))
    val_labels = val_predictions.label_ids
    
    # Per-label threshold optimization
    best_thresholds = np.zeros(num_labels)
    threshold_candidates = np.linspace(0.05, 0.95, 19)
    
    for label_idx in range(num_labels):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in threshold_candidates:
            pred_binary = (val_probs[:, label_idx] >= threshold).astype(int)
            f1 = f1_score(val_labels[:, label_idx], pred_binary, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        best_thresholds[label_idx] = best_threshold
    
    logger.info(f"Optimized thresholds: min={best_thresholds.min():.3f}, "
                f"max={best_thresholds.max():.3f}, mean={best_thresholds.mean():.3f}")
    
    return best_thresholds

def evaluate_with_thresholds(probs, labels, thresholds):
    """Evaluate predictions using optimized thresholds."""
    preds = np.zeros_like(probs)
    for i in range(len(thresholds)):
        preds[:, i] = (probs[:, i] >= thresholds[i]).astype(int)
    
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    
    # Per-label F1 scores
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label_f1": per_label_f1
    }

def main():
    """Main training and evaluation function."""
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    
    # Load data - assuming df_train, df_val, df_test are available
    # These should be loaded from your data loading pipeline
    logger.info("Loading data...")
    
    
    df_train = pd.read_csv(r"train.csv")
    df_val   = pd.read_csv(r"val.csv")
    df_test  = pd.read_csv(r"test.csv")  

    df_train["labels"] = df_train["labels"].apply(eval)
    df_val["labels"]   = df_val["labels"].apply(eval)
    df_test["labels"]  = df_test["labels"].apply(eval)     
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create datasets
    train_dataset = GoEmotionsDataset(
        df_train["text"].tolist(),
        df_train["labels"].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    val_dataset = GoEmotionsDataset(
        df_val["text"].tolist(),
        df_val["labels"].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    test_dataset = GoEmotionsDataset(
        df_test["text"].tolist(),
        df_test["labels"].tolist(),
        tokenizer,
        MAX_LENGTH
    )
    
    # Load model for fine-tuning (not as fixed feature extractor)
    logger.info("Loading model for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    # Calculate class weights for imbalance handling (corrected implementation)
    logger.info("Calculating class weights for imbalance handling...")
    all_labels = np.array(df_train["labels"].tolist())
    positive_counts = all_labels.sum(axis=0)
    total = len(all_labels)
    negative_counts = total - positive_counts
    
    # Calculate pos_weight for BCEWithLogitsLoss (ratio of negatives to positives)
    # Avoid division by zero
    positive_counts = np.clip(positive_counts, 1, None)
    pos_weight = negative_counts / positive_counts
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float).to(device)
    
    logger.info(f"Class weights - min: {pos_weight.min():.3f}, max: {pos_weight.max():.3f}, "
                f"mean: {pos_weight.mean():.3f}")
    
    # Training arguments
    #-----------------------------Temporary _________________________
    import transformers
    print("DEBUG transformers version:", transformers.__version__)
    print("DEBUG TrainingArguments from:", transformers.TrainingArguments.__module__)
    #-----------------------------------------------------------------



    run_name = f"bert_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=run_name,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,  # Use ratio instead of fixed steps
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        # Use bf16 if supported, fallback to fp16
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        dataloader_pin_memory=True,  # Speed improvement for CUDA
        gradient_accumulation_steps=2,  # Effective batch size = BATCH_SIZE * 2
        max_grad_norm=1.0,  # Gradient clipping for stability
        report_to=None,  # Disable wandb/tensorboard for now
        save_total_limit=3,  # Save space
    )
    
    # Initialize custom trainer with weighted loss
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight_tensor,
    )
    
    # Add early stopping callback
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    
    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "labels": GOEMOTIONS_LABELS,
        "pos_weights": pos_weight.tolist(),
        "training_time": train_result.metrics.get('train_runtime', 0),
        "total_train_samples": len(train_dataset),
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Optimize thresholds on validation set
    best_thresholds = optimize_thresholds(trainer, val_dataset, NUM_LABELS)
    
    # Save optimized thresholds
    np.save(os.path.join(OUTPUT_DIR, "optimized_thresholds.npy"), best_thresholds)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = trainer.predict(test_dataset)
    test_probs = 1 / (1 + np.exp(-test_predictions.predictions))
    test_labels = test_predictions.label_ids
    
    # Evaluate with default 0.5 thresholds
    default_preds = (test_probs >= 0.5).astype(int)
    default_macro_f1 = f1_score(test_labels, default_preds, average="macro", zero_division=0)
    default_micro_f1 = f1_score(test_labels, default_preds, average="micro", zero_division=0)
    
    # Evaluate with optimized thresholds
    optimized_results = evaluate_with_thresholds(test_probs, test_labels, best_thresholds)
    
    logger.info(f"Test Results (0.5 threshold): Macro F1: {default_macro_f1:.4f}, "
                f"Micro F1: {default_micro_f1:.4f}")
    logger.info(f"Test Results (optimized): Macro F1: {optimized_results['macro_f1']:.4f}, "
                f"Micro F1: {optimized_results['micro_f1']:.4f}")
    
    # Generate comprehensive classification report
    logger.info("Generating detailed classification report...")
    optimized_preds = np.zeros_like(test_probs)
    for i in range(len(best_thresholds)):
        optimized_preds[:, i] = (test_probs[:, i] >= best_thresholds[i]).astype(int)
    
    # Classification report with actual label names
    print("\nClassification Report (Optimized Thresholds):")
    print(classification_report(test_labels, optimized_preds, 
                               target_names=GOEMOTIONS_LABELS, zero_division=0))
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame({
        "text": df_test["text"].tolist(),
        "true_labels": [list(arr) for arr in test_labels],
        "predicted_labels_default": [list(arr) for arr in default_preds],
        "predicted_labels_optimized": [list(arr) for arr in optimized_preds],
        "predicted_probs": [list(arr) for arr in test_probs]
    })
    
    results_df.to_csv(f"bert_finetuned_results_{timestamp}.csv", index=False)
    
    # Save per-label results for analysis
    per_label_results = pd.DataFrame({
        "label": GOEMOTIONS_LABELS,
        "threshold": best_thresholds,
        "f1_score": optimized_results['per_label_f1'],
        "pos_weight": pos_weight.tolist(),
        "support": test_labels.sum(axis=0)
    })
    per_label_results.to_csv(f"bert_finetuned_per_label_{timestamp}.csv", index=False)
    
    # Save final metrics for master results table
    final_metrics = {
        "model_type": "Fine-Tuned BERT (SOTA Baseline)",
        "macro_f1_default": float(default_macro_f1),
        "micro_f1_default": float(default_micro_f1),
        "macro_f1_optimized": float(optimized_results['macro_f1']),
        "micro_f1_optimized": float(optimized_results['micro_f1']),
        "training_loss": train_result.metrics.get('train_loss', 0),
        "training_runtime": train_result.metrics.get('train_runtime', 0),
        "training_samples_per_second": train_result.metrics.get('train_samples_per_second', 0),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "timestamp": timestamp
    }
    
    with open(f"bert_finetuned_metrics_{timestamp}.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Print final summary
    print("\n" + "="*80)
    print("FINE-TUNED BERT TRAINING COMPLETED")
    print("="*80)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Default thresholds (0.5): Macro F1: {default_macro_f1:.4f}, Micro F1: {default_micro_f1:.4f}")
    print(f"Optimized thresholds: Macro F1: {optimized_results['macro_f1']:.4f}, Micro F1: {optimized_results['micro_f1']:.4f}")
    print(f"Improvement: +{(optimized_results['macro_f1'] - default_macro_f1):.4f} Macro F1")
    print(f"Total parameters: {final_metrics['total_parameters']:,}")
    print(f"Training time: {final_metrics['training_runtime']:.1f} seconds")
    print("="*80)
    
    logger.info("Fine-tuning completed successfully!")
    
    return final_metrics

if __name__ == "__main__":
    main()