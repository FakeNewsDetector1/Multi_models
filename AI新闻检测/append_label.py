import json

# 输入 JSON 文件路径
input_json_path = "../数据集/mirage-news/metadata/test2.json"
output_json_path = "../数据集/mirage-news/metadata/test_with_label_finally1.json"

# 读取原始 JSON 文件
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历数据并添加 label: 0
for item in data:
    item["label"] = 0

# 保存带有 label 的新 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Updated JSON file saved to {output_json_path}")
