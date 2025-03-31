import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from train import MultiModalModel
from tqdm import tqdm
from train import NewsDataset

def test(model, train_loader, test_loader, device="cuda"):
    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))
    model = model.to(device)
    model.eval()

    def evaluate(loader, name="Train"):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for text_inputs, images, labels in tqdm(loader, desc=f"Evaluating {name}"):
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                images = images.to(device)

                outputs = model(text_inputs, images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())  # 使用真实标签

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"\n{name} Set Evaluation:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        return acc

    # 在训练集和测试集上分别评估
    train_acc = evaluate(train_loader, "Training")
    test_acc = evaluate(test_loader, "Testing")

    # 过拟合判断
    if train_acc - test_acc > 0.2:  # 训练精度比测试高20%以上
        print("\n警告：模型可能过拟合！")
        print(f"训练集与测试集准确率差距: {train_acc - test_acc:.2%}")
    else:
        print("\n模型泛化能力正常")
        print(f"训练集与测试集准确率差距: {train_acc - test_acc:.2%}")


if __name__ == '__main__':
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalModel()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)

    # 创建测试数据集（需要准备测试集的元数据）
    test_dataset = NewsDataset(
        json_path="../数据集/mirage-news/metadata/test_with_label_finally.json",  # 请替换为真实测试集路径
        real_image_dir="../数据集/mirage-news/origin",  # 测试集真实图片目录
        fake_image_dir="../数据集/mirage-news/origin"  # 测试集生成图片目录
    )

    # 创建数据加载器
    train_loader = DataLoader(
        NewsDataset(json_path="../数据集/mirage-news/metadata/train_with_label_finally.json",
                    real_image_dir="../数据集/mirage-news/origin",
                    fake_image_dir="../数据集/mirage-news/fake_images"),
        batch_size=16,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    # 运行测试
    test(model, train_loader, test_loader, device)