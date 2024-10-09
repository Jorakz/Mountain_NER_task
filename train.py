from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import ast

if torch.cuda.is_available():
    GPU = torch.cuda.get_device_name(0)
    VRAM = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    print(f"{GPU} is available with {VRAM}GB V-RAM")
    print("CUDA Version:", torch.version.cuda)
else:
    print("No GPU available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)

tag2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2}

df = pd.read_csv("dataset_redacted.csv")

df['Tokens'] = df['Tokens'].apply(ast.literal_eval)
df['Labels'] = df['Labels'].apply(ast.literal_eval)


def align_labels_with_tokens(labels, word_ids):
    new_labels = [-100] * len(word_ids)
    label_index = 0
    for i, word_id in enumerate(word_ids):
        if word_id is not None:
            if label_index < len(labels):
                new_labels[i] = labels[label_index]
            if i == 0 or word_id != word_ids[i - 1]:
                label_index += 1
    return new_labels


tokenized_inputs = tokenizer(df['Tokens'].tolist(), is_split_into_words=True, padding=True, truncation=True, return_offsets_mapping=True, max_length=512)
labels_aligned = []
for i in range(len(df)):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    labels = [tag2id[tag] for tag in df['Labels'][i]]
    labels_aligned.append(align_labels_with_tokens(labels, word_ids))
tokenized_inputs['labels'] = labels_aligned


class Mountain_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.encodings['labels'][idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


dataset = Mountain_dataset(tokenized_inputs)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
    learning_rate=3e-4,
    adam_epsilon=1e-8,
    save_total_limit=3,
    evaluation_strategy='steps',
    save_strategy='steps',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)





trainer.train()

output_dir = "model_NER"

model.save_pretrained(output_dir)

tokenizer.save_pretrained(output_dir)