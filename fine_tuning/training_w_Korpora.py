import random

import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import (classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Korpora dataset and convert to Dataset
if "corpus" not in globals():
    from Korpora import Korpora
    corpus = Korpora.load("naver_changwon_ner", root_dir="./")
items = corpus.train

def convert_korpora_to_dataset(items):
    all_tokens = [item.words for item in items]
    all_tags = [item.tags for item in items]
    
    # 데이터셋을 train:validation:test = 8:1:1로 랜덤 분할
    train_tokens, temp_tokens, train_tags, temp_tags = train_test_split(
        all_tokens, all_tags, test_size=0.2, random_state=42
    )
    val_tokens, test_tokens, val_tags, test_tags = train_test_split(
        temp_tokens, temp_tags, test_size=0.5, random_state=42
    )
    
    # HuggingFace Dataset 형식으로 변환
    train_dataset = Dataset.from_dict({
        "tokens": train_tokens,
        "ner_tags": train_tags,
    })
    val_dataset = Dataset.from_dict({
        "tokens": val_tokens,
        "ner_tags": val_tags,
    })
    test_dataset = Dataset.from_dict({
        "tokens": test_tokens,
        "ner_tags": test_tags,
    })

    # DatasetDict 구성 (Optional)
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })
    return dataset

dataset = convert_korpora_to_dataset(items)
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# 2. Label Encoding
label_encoder = LabelEncoder()
flat_labels = [tag for example in train_dataset["ner_tags"] for tag in example]
label_encoder.fit(flat_labels)

label2id = {label: id for id, label in enumerate(label_encoder.classes_)}
id2label = {id: label for label, id in label2id.items()}

def encode_labels(example):
    example["labels"] = [label2id[tag] for tag in example["ner_tags"]]
    return example

train_dataset = train_dataset.map(encode_labels)
validation_dataset = validation_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)


# 3. Tokenization and label alignment
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")    # Tokenizer로는 distilBERT도 고려해 볼 것. 파라미터가 BERT의 60%, 그에 따라 속도도 60% 정도 빠르지만 성능은 유사.

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512,  # 모델의 max_position_embeddings와 일치
    )
    
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    labels = []
    
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(example["labels"][word_idx] if example["labels"][word_idx] != label2id["-"] else -100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=False)
validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=False)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=False)


# 4. Load and config the model
model = AutoModelForTokenClassification.from_pretrained(
    "monologg/koelectra-base-v3-discriminator",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
).to(device)

# 5. Config training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-koelectra-korpora-ner",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True if device.type == 'cuda' else False,  # CUDA 사용 시 mixed precision 활성화
    no_cuda=False if device.type == 'cuda' else True,  # CUDA 사용 가능 시 활성화
)

# 5-1. Config evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(-1)
    
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    
    return {
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# 6. Train the model
trainer.train()

# 7. Save the model and evaluate
trainer.save_model("./finetuned-koelectra-korpora-ner")