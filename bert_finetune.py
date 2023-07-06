from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auc
    }

"""
MRPC
Microsoft Research Paraphrase Corpus
마이크로소프트에서 공개한 문장 쌍 데이터셋으로 같은지 다른지를 0과 1로 판단
0: not_same
1: same
평가
accuracy 와 F1 스코어로 측정
[출처] GLUE 데이터셋|작성자 지피
https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForNextSentencePrediction,
https://huggingface.co/sgugger/glue-mrpc?text=I+like+you.+I+love+you,
https://huggingface.co/docs/datasets/v1.0.1/loading_metrics.html 참고 
"""

# 모델 및 토크나이저 초기화
model = BertForSequenceClassification.from_pretrained("bert_trained", num_labels=2) 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# MRPC 태스크의 데이터셋 및 메트릭 로드
dataset = load_dataset("glue", "mrpc")
metric = load_metric("glue", "mrpc")

# 데이터셋 전처리
def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

# 데이터셋에 전처리 함수 적용
dataset = dataset.map(encode, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 훈련을 위한 TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate= 5e-05,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

# 훈련 및 평가를 위한 Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  
)

# 훈련
trainer.train()

# 평가
eval_results = trainer.evaluate()

# 정확도와 F1 스코어 출력
accuracy = eval_results["eval_accuracy"]
f1_score = eval_results["eval_f1"]
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
