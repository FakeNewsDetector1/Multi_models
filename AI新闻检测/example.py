import torch
from PIL import Image
import json
from train import MultiModalModel
import torch.nn as nn
from transformers import AutoTokenizer
from torchvision import models, transforms


# 配置路径
local_model_path = "models/roberta-base"


class AIGCDetector:
    def __init__(self, model_path, text_model_path=local_model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型（必须与训练时的结构完全一致）
        self.model = MultiModalModel(text_model_path)

        # 加载权重时处理DataParallel前缀
        state_dict = torch.load(model_path, map_location=self.device)

        # 移除DataParallel的'module.'前缀（如果存在）
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # 加载修正后的权重
        self.model.load_state_dict(state_dict)

        # 多GPU支持（如果可用）
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.model.eval()

        # 其余初始化保持不变...
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess_text(self, text):
        return self.text_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.image_transform(image).unsqueeze(0)  # 添加batch维度

    def predict(self, text, image_path):
        # 数据预处理
        text_inputs = self.preprocess_text(text)
        image_tensor = self.preprocess_image(image_path)

        # 转移到设备
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        image_tensor = image_tensor.to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.model(text_inputs, image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        # 解析结果
        ai_prob = probabilities[0][0].item()  # 假设第0类是AI生成
        human_prob = probabilities[0][1].item()

        return {
            "AI_generated_prob": ai_prob,
            "human_prob": human_prob,
            "prediction": "AI生成" if ai_prob > human_prob else "人工创作"
        }


if __name__ == '__main__':
    # 使用示例
    detector = AIGCDetector(
        model_path="best_model.pth",
        text_model_path=local_model_path
    )

    # 测试样本
    test_text = "Senator Christopher J. Dodd, with his wife and daughter, announces his divorce and resignation from the Senate on Wednesday."
    test_image_path = "D:\PyCharmProjects\伪新闻检测\数据集\mirage-news\origin\\1.jpg"  # 替换为真实路径

    # 进行预测
    result = detector.predict(test_text, test_image_path)

    # 格式化输出
    print("检测结果：")
    print(f"文本长度：{len(test_text)}字符")
    print(f"图片路径：{test_image_path}")
    print("-" * 40)
    print(f"AI生成概率：{result['AI_generated_prob']:.2%}")
    print(f"人工创作概率：{result['human_prob']:.2%}")
    print(f"最终判断：{result['prediction']}")