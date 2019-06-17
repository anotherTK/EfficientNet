import json
from PIL import Image
import argparse
import torch
from torchvision import transforms

from EfficientNet import  EfficientNet


def parse_args():
    parser = argparse.ArgumentParser(description="ArgumentParser for EfficientNet")
    
    parser.add_argument('--arch', '-a', default='efficientnet-b0', help="Choose the network")
    parser.add_argument('--img', default='demo/img.jpg', help="Image path")
    parser.add_argument('--cls', default='demo/labels_map.txt', help="Class identification")

    args = parser.parse_args()
    return args

def main(args):

    # 加载训练好的模型
    model = EfficientNet.from_pretrained(args.arch)
    # 处理图像
    tfms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 模型输入
    img = tfms(Image.open(args.img)).unsqueeze(0)

    # 加载模型类别
    labels_map = json.load(open(args.cls))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # 分类
    model.eval()
    with torch.no_grad():
        outputs = model(img)

    # 预测输出
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))

if __name__ == "__main__":
    args = parse_args()
    main(args)
