import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"Logits shape: {logits.shape}")  # 打印模型輸出形狀
        print(f"Labels shape: {labels.shape}")  # 打印標籤形狀
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").squeeze(-1).squeeze(-1)  # 去掉多餘的維度
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss()  # 使用交叉熵損失
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss if loss is not None else torch.tensor(0.0)



def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}")
        return None

def extract_text_from_virustotal(virustotal_data):
    try:
        dns_records = virustotal_data['data']['attributes']['last_dns_records']
        texts = [f"{record['type']}:{record['value']}" for record in dns_records]
        return ' '.join(texts)
    except KeyError:
        print("Key not found in VirusTotal data")
        return None

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).expand(512)  # 假設模型輸出形狀為[batch_size, 512, vocab_size]
        print("Debug: Item shape:", {k: v.shape for k, v in item.items()})  # Debug line
        return item

    def __len__(self):
        return len(self.labels)

def main():
    parser = argparse.ArgumentParser(description='Train SecBERT model.')
    parser.add_argument('--data_dir', type=str, default='E:\\CTIbySecBERT\\data\\domain', help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    args = parser.parse_args()

    data_dir = args.data_dir
    batch_size = args.batch_size
    
    tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
    model = AutoModelForMaskedLM.from_pretrained('jackaduma/SecBERT')

    texts = []
    labels = []

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isdir(file_path):
            continue
        virustotal_data = read_json_file(file_path)
        if virustotal_data:
            extracted_text = extract_text_from_virustotal(virustotal_data)
            if extracted_text:
                texts.append(extracted_text)
                labels.append(1 if 'malwaredomainlist' in filename else 0)

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = MyDataset(encodings, labels)

    print("Debug: Encodings keys:", encodings.keys())
    print("Debug: Sample encoding:", encodings['input_ids'][0])
    print("Debug: Sample label:", labels[0])

    if len(dataset) == 0:
        print("The dataset is empty. Exiting.")
        return

    print("Dataset Length:", len(dataset))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
    )

    trainer = MyTrainer(  # 使用自定義的 MyTrainer
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained('./saved_model/')

if __name__ == '__main__':
    main()
