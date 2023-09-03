# 导入所需的库
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    # 初始化 SecBERT 模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
    model = AutoModelForMaskedLM.from_pretrained('jackaduma/SecBERT')

    # 准备数据和标签（这里只是一个示例，你应该使用你自己的数据）
    texts = ['示例文本1', '示例文本2']
    labels = [0, 1]  # 0 表示“非恶意”，1 表示“恶意”

    # 数据预处理
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = MyDataset(encodings, labels)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # 开始微调
    trainer.train()

    # 保存微调后的模型
    model.save_pretrained('./saved_model/')


if __name__ == '__main__':
    main()
