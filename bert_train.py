from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, BertConfig
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling

def load_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read().split('\n')
    return text

dataset = load_dataset("lucadiliello/english_wikipedia")

# Define the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["maintext"], truncation=True, padding="max_length", max_length=128)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["maintext"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Initialize the model
config = BertConfig(vocab_size=30522)
model = BertForMaskedLM(config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_base_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./bert_trained")
