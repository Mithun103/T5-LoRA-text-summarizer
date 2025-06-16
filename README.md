# 📝 T5 Text Summarizer with LoRA Fine-Tuning

This project demonstrates a complete pipeline for building a summarization system using the T5 Transformer model fine-tuned with **LoRA (Low-Rank Adaptation)** via **PEFT (Parameter-Efficient Fine-Tuning)**. It also provides a live Gradio interface and evaluation using **ROUGE** metrics.

---

## 📌 Overview

* **Model:** T5 (Text-To-Text Transfer Transformer)
* **Technique:** LoRA for efficient fine-tuning
* **Interface:** Gradio for interactive summarization
* **Evaluation:** ROUGE metric (Recall-Oriented Understudy for Gisting Evaluation)

---

💡 Powered by **Gradio** and hosted on **Hugging Face Spaces**
## 📁 Project Structure

```
t5-text-summarizer/
├── full_lora_summarizer/        # Directory to store fine-tuned model
├── LLM_LORA.ipynb               # Notebook for fine-tuning and evaluation
├── app.py                       # Gradio interface
├── README.md                    # Documentation
├── requirements.txt             # Python dependencies
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/t5-text-summarizer.git
cd t5-text-summarizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install transformers datasets peft accelerate gradio evaluate nltk rouge_score
```

---

## 🏋️ Fine-Tuning the Model with LoRA

### LoRA Configuration

LoRA adds low-rank matrices to the attention layers to reduce the number of trainable parameters while maintaining performance.

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
model = get_peft_model(model, peft_config)
```

### Dataset and Tokenization

Using the CNN/DailyMail dataset for training and evaluation:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["highlights"], max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)
```

### Training

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(2000)),
    eval_dataset=tokenized_dataset["validation"].select(range(500)),
    tokenizer=tokenizer
)

trainer.train()
```

---

## 📊 Evaluation with ROUGE

### Install Metrics

```bash
pip install evaluate nltk rouge_score
```

### Evaluation Code

```python
from evaluate import load
import numpy as np

rouge = load("rouge")
predictions = trainer.predict(tokenized_dataset["validation"].select(range(100)))
pred_ids = np.argmax(predictions.predictions, axis=-1)
decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
print("📊 ROUGE Evaluation Results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

### Eval Output

```
📊 ROUGE Evaluation Results:
rouge1: 0.0036
rouge2: 0.0000
rougeL: 0.0036
rougeLsum: 0.0036
```

---

## 🌐 Gradio UI for Live Summarization

### Run the App

```bash
python app.py
```

### Code (app.py)

```python
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = "./full_lora_summarizer"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def summarize(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

interface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="📝 T5 Summarizer with LoRA",
    description="Fine-tuned T5 model with LoRA for efficient summarization."
)

interface.launch(share=True)
```

---
### Gradio Interface Output
![image](https://github.com/user-attachments/assets/d0a00741-41e4-4ced-b2db-a114e9f719a3)

---
### 🚀 Live Demo

Try the model live in your browser!  
🔗 [**Click here to test the Summarizer App →**](https://your-live-demo-link.com)  
---
## 💡 Use Cases

* News summarization
* Academic paper summarization
* Long-form blog content
* Meeting transcripts

---

## 🔖 Acknowledgements

* Hugging Face Transformers
* PEFT by Hugging Face
* Gradio for UI
* CNN/DailyMail dataset

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙋 Contact

Created by \[MITHUN M S] — feel free to reach out!
