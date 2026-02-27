import argparse
import torch

# 解析命令行参数
parser = argparse.ArgumentParser(description='Inspect PyTorch model structure')
parser.add_argument('--input', type=str, required=True, help='Path to the model file (.pth)')
args = parser.parse_args()

# 加载模型
model_path = args.input
print(f"Loading model from: {model_path}")

# 加载模型参数
model_data = torch.load(model_path, map_location='cpu', weights_only=False)

# 检查模型数据结构
if isinstance(model_data, dict):
    print(f"Model structure: Dictionary")
    print(f"Keys: {list(model_data.keys())}\n")
    
    # 获取state_dict
    if 'model' in model_data:
        state_dict = model_data['model']
    elif 'state_dict' in model_data:
        state_dict = model_data['state_dict']
    else:
        state_dict = model_data
else:
    state_dict = model_data

# 统计信息
print("=" * 100)
print(f"{'#':<5} {'Parameter Name':<70} {'Shape':<25}")
print("=" * 100)

total_params = 0
param_list = []

for name, param in state_dict.items():
    if isinstance(param, torch.Tensor):
        shape = tuple(param.shape)
        num_params = param.numel()
        total_params += num_params
        param_list.append((name, shape, num_params, param.dtype))

# 排序
param_list.sort(key=lambda x: x[0])

# 打印
for idx, (name, shape, num_params, dtype) in enumerate(param_list, 1):
    shape_str = str(shape)
    print(f"{idx:<5} {name:<70} {shape_str:<25}")

# 汇总
print("=" * 100)
print(f"\nSummary:")
print(f"  Total layers/parameters: {len(param_list)}")
print(f"  Total parameter count: {total_params:,}")
print(f"  Total parameters (Million): {total_params / 1e6:.2f}M")
print(f"  Total parameters (Billion): {total_params / 1e9:.4f}B")

# 数据类型统计
dtypes = {}
for _, _, _, dtype in param_list:
    dtypes[str(dtype)] = dtypes.get(str(dtype), 0) + 1

print(f"\nData types:")
for dtype, count in dtypes.items():
    print(f"  {dtype}: {count} parameters")
