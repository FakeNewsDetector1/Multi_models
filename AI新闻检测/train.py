import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
import json
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

# 配置路径
local_model_path = "models/roberta-base"
real_json_path = "../数据集/mirage-news/metadata/train.json"
output_json_path = "../数据集/mirage-news/metadata/train_with_fake.json"
real_image_dir = "../数据集/mirage-news/origin"
fake_image_dir = "../数据集/mirage-news/fake_images"
gpt2_path = "models/gpt2"
sd_path = "models/stable-diffusion-v1-4"
#
# # 创建目录
# os.makedirs(fake_image_dir, exist_ok=True)
#
# # 加载模型
# print("Loading models...")
# text_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
# text_model = GPT2LMHeadModel.from_pretrained(gpt2_path)
# sd_pipe = StableDiffusionPipeline.from_pretrained(sd_path,
#                                                   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
# sd_pipe = sd_pipe.to("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def generate_text(prompt):
#     inputs = text_tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True)
#     outputs = text_model.generate(
#         inputs.input_ids,
#         max_length=128,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         pad_token_id=text_tokenizer.eos_token_id
#     )
#     return text_tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#
# def generate_image(prompt, save_path):
#     image = sd_pipe(prompt[:77]).images[0]  # 截断到77字符（SD的限制）
#     image.save(save_path)
#
#
# # 处理原始数据
# with open(real_json_path, "r", encoding="utf-8") as f:
#     original_data = json.load(f)
#
# new_data = []
#
# for idx, item in enumerate(tqdm(original_data[:4999]), start=0):  # 生成 0 到 4999
#     fake_text = generate_text(item["text"][:50])  # 使用前50字符作为提示
#     fake_img_path = os.path.join(fake_image_dir, f"{idx}.jpg")
#     generate_image(item["text"][:50], fake_img_path)
#
#     new_data.append({
#         "id": idx,
#         "image": os.path.join("fake_images_text", f"{idx}.jpg"),
#         "text": fake_text
#     })
#
# # 保存新数据
# with open(output_json_path, "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=2)
#
# print(f"Generated {len(new_data)} fake samples")


class NewsDataset(Dataset):
    def __init__(self, json_path, real_image_dir, fake_image_dir, text_model=local_model_path, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.real_image_dir = real_image_dir
        self.fake_image_dir = fake_image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        # 处理图片路径
        if "fake_images" in item['image']:
            image_path = os.path.join(self.fake_image_dir, os.path.basename(item['image']))
        else:
            image_path = os.path.join(self.real_image_dir, os.path.basename(item['image']).replace("\\", "/"))

        # 文本处理
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        # 图像处理
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return text_inputs, image, torch.tensor(label, dtype=torch.long)

class MultiModalModel(nn.Module):
    def __init__(self, text_model=local_model_path, image_model='resnet50'):
        super().__init__()
        # 从本地加载文本模型
        config = AutoConfig.from_pretrained(text_model)
        self.text_encoder = AutoModel.from_pretrained(
            text_model,
            config=config
        )
        self.text_proj = nn.Linear(config.hidden_size, 512)  # 使用配置中的维度

        # 图像编码器
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(2048, 512)

        # 融合层
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=2)
        self.classifier = nn.Linear(512, 2)

    def forward(self, text_inputs, images):
        # 明确指定输入参数
        text_outputs = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        ).last_hidden_state  # 形状: [batch_size, seq_len, hidden_dim]

        # 获取CLS token特征
        text_features = self.text_proj(text_outputs[:, 0, :])  # [batch_size, 512]

        # 图像特征提取（保持不变）
        image_features = self.image_encoder(images)  # [batch_size, 512]

        # 特征融合（调整维度处理）
        fused_features = self.transformer(
            text_features.unsqueeze(1),  # 添加序列维度 [batch_size, 1, 512]
            image_features.unsqueeze(1)  # 添加序列维度 [batch_size, 1, 512]
        ).squeeze(1)  # 移除序列维度 [batch_size, 512]

        return self.classifier(fused_features)

def train(model, dataloader, epochs=5, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            text_inputs, images, labels = batch
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(text_inputs, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_model.pth2")

def evaluate(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for text_inputs, images in dataloader:
            text_inputs = {k: v.squeeze(0).to(device) for k, v in text_inputs.items()}
            images = images.to(device)

            logits = model(text_inputs, images)
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend(pred_labels)
            labels.extend([1] * len(pred_labels))  # 假设所有样本都是真实数据

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"Evaluation Results:\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}")


if __name__ == '__main__':
    # 初始化数据集
    dataset = NewsDataset(
        json_path="../数据集/mirage-news/metadata/train_with_label_finally.json",
        real_image_dir="../数据集/mirage-news/origin",
        fake_image_dir="../数据集/mirage-news/fake_images",
        text_model=local_model_path
    )

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    model = MultiModalModel()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    # 训练模型
    train(model, dataloader, epochs=10)

    # 最终评估
    evaluate(model, dataloader)

    # 保存最终模型
    torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
               "final_multimodal_model.pth")