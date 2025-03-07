import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

df = pd.read_csv('data/train_submission_cleaned.csv')
df['Text'] = df['Text'].astype(str)
df['Label'] = df['Label'].astype(str)

# Prompting
df['input_text'] = "identify language: " + df['Text']
df['target_text'] = df['Label']

# Test / Train split
train_df, val_df = train_test_split(df, test_size=0.05, stratify=df['Label'], random_state=42)

train_dataset = Dataset.from_pandas(train_df[['input_text', 'target_text']])
val_dataset = Dataset.from_pandas(val_df[['input_text', 'target_text']])

# Load model and tokenizer
model_checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

# Preprocessing + Tokenization parameters
max_source_length = 256  
max_target_length = 10   

def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments set up
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_mt5_base",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Accuracy calculation
def compute_metrics(eval_preds):
    
    preds, labels = eval_preds
    decoded_preds = [tokenizer.decode(pred.tolist(), skip_special_tokens=True) for pred in preds]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = [tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in labels]

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    correct = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    accuracy = correct / len(decoded_preds)
    return {"accuracy": accuracy}

# Training
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()

# Evaluation
eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

# Save model and tokenizer
model.save_pretrained("./mt5_base_language_classifier")
tokenizer.save_pretrained("./mt5_base_language_classifier")
