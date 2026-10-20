import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# -----------------------
# 1. 加载 ViT ReID 模型
# -----------------------
model = torch.load("your_vit_model.pth", map_location="cpu")
model.eval()

# -----------------------
# 2. Hook：获得每层 attention
# -----------------------
attn_maps = []

def get_attn_hook(module, input, output):
    # output shape: [batch, heads, tokens, tokens]
    attn_maps.append(output.detach().cpu())

# 注册所有 transformer block 的注意力输出
for blk in model.backbone.blocks:
    blk.attn.attn_drop.register_forward_hook(get_attn_hook)

# -----------------------
# 3. 预处理输入图片
# -----------------------
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img_path = "test.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# -----------------------
# 4. 前向传播，获取注意力
# -----------------------
with torch.no_grad():
    _ = model(input_tensor)

# -----------------------
# 5. Attention Rollout
# -----------------------
# attn_maps: list of [1, heads, tokens, tokens]
attn = attn_maps[0].squeeze(0)
# 平均多头
attn = attn.mean(0)   # [tokens, tokens]

# 使用残差连接方式 rollout
rollout = attn + torch.eye(attn.size(0))
for i in range(1, len(attn_maps)):
    attn_i = attn_maps[i].squeeze(0).mean(0)
    attn_i = attn_i + torch.eye(attn_i.size(0))
    rollout = attn_i @ rollout

# 取 cls token 对 patch tokens 的注意力
cls_attn = rollout[0, 1:]   # 去掉 cls 本身

# -----------------------
# 6. 转换成热力图
# -----------------------
h, w = 16, 8   # ViT patch grid (256/16=16, 128/16=8) 如不同 patch size 需修改
cam = cls_attn.reshape(h, w).numpy()
cam = cam / cam.max()

# resize 到原图大小
cam = cv2.resize(cam, (img.width, img.height))

# 转为 heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# 融合
result = heatmap * 0.4 + np.array(img) * 0.6
cv2.imwrite("vit_rollout_cam.jpg", result)

print("Saved to vit_rollout_cam.jpg")
