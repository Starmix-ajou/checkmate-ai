from dataclasses import dataclass
from typing import Dict

from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)
from transformers import (BertForTokenClassification, BertTokenizerFast,
                          Trainer, TrainingArguments)


@dataclass
class LabelMapper:
    labels: Dict[str, int] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {
                "O": 0,
                "B-PER": 1,
                "I-PER": 2,
                "B-ORG": 3,
                "I-ORG": 4,
                "B-LOC": 5,
                "I-LOC": 6,
                "B-MISC": 7,
                "I-MISC": 8,
            }
    
    def get_id(self, label: str) -> int:
        return self.labels[label]


# Fine-tuning의 단계
# 1. load dataset
datasets = load_dataset("json",
    data_files ={
        "train": "ner_dataset.jsonl",
        "valid": "ner_validation.jsonl",
        "test": "ner_test.jsonl",
    },
    split_names=["train", "valid", "test"]
)

# 2. load pre-trained model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")    # Tokenizer로는 distilBERT도 고려해 볼 것. 파라미터가 BERT의 60%, 그에 따라 속도도 60% 정도 빠르지만 성능은 유사.
model = BertForTokenClassification.from_pretrained("bert-base-cased")

# 3. preprocessing: tokenize dataset ("tokens" -> "input_ids") and align labels ("tags" -> "label_ids")
def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )
    label_all = []
    for i, word_ids in enumerate(tokenized.word_ids(batch_index=i) for i in range(len(examples["tokens"]))):
        labels=[]
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            else:
                mapper = LabelMapper()
                labels.append(mapper.get_id(examples["ner_tags"][i][word_id]))
        label_all.append(labels)
    tokenized["labels"] = label_all
    return tokenized

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# 4. define evaluation metrics
### 4-1. f1-score
### 4-2. precision
### 4-3. recall

# 5. config & execute Trainer
training_args = TrainingArguments(
    output_dir="trained_models/results",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
trainer.train() # 학습 시작

# 6. 만약 evaluation이 heuristic standard를 충족하면 모델을 저장하고 종료

# 7. 만약 evaluation이 heuristic standard를 충족하지 않으면 모델을 조정하고 다시 훈련 (feedback)
