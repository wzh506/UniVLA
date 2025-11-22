import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 调小张量尺寸，减少显存压力
x = torch.randn((512, 512, 16), device=device, requires_grad=True)
y = torch.randn((512, 512, 16), device=device, requires_grad=True)

print("开始持续占用GPU... 按 Ctrl+C 退出")
try:
    while True:
        z = torch.matmul(x, y.transpose(1, 2))
        z = z.relu()
        z.sum().backward()  # 不要传参数
        x.grad.zero_()
        y.grad.zero_()
        time.sleep(0.005)
except KeyboardInterrupt:
    print("测试结束")