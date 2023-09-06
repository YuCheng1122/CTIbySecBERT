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
        with open(file_path, 'r', encoding='utf-8') as f:  # 指定編碼為 utf-8
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}")
        return None


def extract_text_from_virustotal(virustotal_data):
    try:
        print("Received data:", virustotal_data)  # 打印接收到的原始数据
        dns_records = virustotal_data.get('data', {}).get(
            'attributes', {}).get('last_dns_records', [])
        if not dns_records:
            print("Warning: No DNS records found.")  # 如果没有找到DNS记录，打印警告
        texts = [f"{record['type']}:{record['value']}" for record in dns_records]
        extracted_text = ' '.join(texts) if texts else None  # 合并文本或返回None
        print("Extracted text:", extracted_text)  # 打印提取出的文本
        return extracted_text
    except KeyError:
        print("Unexpected key structure in VirusTotal data")  # 键错误时打印提示
        return None


def extract_text_from_metre(new_data):
    try:
        if 'objects' in new_data and len(new_data['objects']) > 0:
            obj = new_data['objects'][0]
            tactic_refs = obj.get('tactic_refs', [])
            x_mitre_domains = obj.get('x_mitre_domains', [])
            description = obj.get('description', "")
            return ' '.join(tactic_refs + x_mitre_domains + [description])
        else:
            print('Key not found in new data')
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def read_and_extract_from_directory(directory, extraction_functions):
    texts = []
    labels = []
    print(f"Reading from directory: {directory}")  # Debug line
    for filename in os.listdir(directory):
        print(f"Processing file: {filename}")  # Debug line
        file_path = os.path.join(directory, filename)
        if os.path.isdir(file_path):
            continue
        for key, func in extraction_functions.items():
            if key in filename:
                # Debug line
                print(f"Using extraction function for key: {key}")
                data = read_json_file(file_path)
                if data:
                    extracted_text = func(data)
                    if extracted_text:
                        texts.append(extracted_text)
                        labels.append(
                            1 if 'malwaredomainlist' in filename else 0)
    return texts, labels


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        seq_len = len(self.encodings['input_ids'][idx])
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(
            self.labels[idx], dtype=torch.long).expand(seq_len)
        return item

    def __len__(self):
        return len(self.labels)


def main():
    parser = argparse.ArgumentParser(description='Train SecBERT model.')
    parser.add_argument('--data_dir', type=str,
                        default='E:\\CTIbySecBERT\\data\\domain', help='Path to the data directory.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Training batch size.')
    args = parser.parse_args()

    data_dir = args.data_dir
    batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
    model = AutoModelForMaskedLM.from_pretrained('jackaduma/SecBERT')

    all_texts = []
    all_labels = []
    extraction_functions_domain = {
        'alexa.com': extract_text_from_virustotal,
        'github.com': extract_text_from_virustotal,
        'malwaredomainlist.com': extract_text_from_virustotal,
        'openphish.com': extract_text_from_virustotal,
        'phishtank.com': extract_text_from_virustotal,
        'virustotal.com': extract_text_from_virustotal}
    extraction_functions_metre = {
        'enterprise-attack': extract_text_from_metre,
        'ics-attack': extract_text_from_metre,
        'mobile-attack': extract_text_from_metre,
        'pre-attack': extract_text_from_metre
    }

    directories = [
        ('E:\\CTIbySecBERT\\data\\domain', extraction_functions_domain),
        ('E:\\CTIbySecBERT\\data\\metre', extraction_functions_metre)
    ]

    all_texts = []
    all_labels = []
    for directory, funcs in directories:
        texts, labels = read_and_extract_from_directory(directory, funcs)
        all_texts.extend(texts)
        all_labels.extend(labels)

    encodings = tokenizer(all_texts, truncation=True,
                          padding=True, max_length=512)
    dataset = MyDataset(encodings, all_labels)

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
