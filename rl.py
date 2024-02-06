import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# 假设 YourModelWithLoRA 是一个包含LoRA参数的预训练模型
from your_model_with_lora import YourModelWithLoRA
import matplotlib.pyplot as plt

class SimplePolicyGradient:
    def __init__(self, parameters, lr=0.01):
        self.optimizer = optim.Adam(parameters, lr=lr)

    def update(self, log_probs, rewards):
        policy_loss = [-log_prob * reward for log_prob, reward in zip(log_probs, rewards)]
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


def visualize_image_and_collect_feedback(image_tensor):
    """
    显示图片并收集反馈。
    参数:
    - image_tensor: 一个形状为(C, H, W)的张量，代表图片。

    返回:
    - reward: 根据人类反馈得到的奖励值。
    """
    # 将张量转换为numpy数组，并调整为HWC格式用于显示
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean  # 反归一化
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title("Evaluate the object detection")
    plt.show()

    # 收集反馈
    feedback = input("Is the detected object correct? (yes/no): ")
    reward = 1 if (feedback.lower() == 'yes' or feedback.lower() == 'y') else -1
    return reward

def virtual_feedback(prediction, ground_truth):
    # 这个函数需要实现为接收模型预测和真实标签，返回奖励
    return +1 if prediction == ground_truth else -1


def main():
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YourModelWithLoRA().to(device)
    model.load_state_dict(torch.load("path_to_your_model_with_lora"))
    model.eval()  # 设置为评估模式

    # 冻结非LoRA参数
    for param in model.parameters():
        param.requires_grad = False
    # 启用LoRA参数
    lora_params = [param for name, param in model.named_parameters() if 'lora' in name]
    for param in lora_params:
        param.requires_grad = True

    # 实例化Policy Gradient
    pg_agent = SimplePolicyGradient(lora_params, lr=0.01)

    # 数据加载器
    dataloader = DataLoader(...)  # 假设你已经有了数据加载器

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        # case1: we obtain the ground-truth
        # for img, ground_truth in dataloader:
        #     img = img.to(device)
        #     ground_truth = ground_truth.to(device)
        #
        #     # 获取模型输出和log概率
        #     logits = model(img)
        #     log_probs = F.log_softmax(logits, dim=-1)
        #     action = torch.argmax(log_probs, dim=-1)  # 假设action是预测的类别
        #
        #     # 获取人类反馈
        #     reward = virtual_feedback(action.item(), ground_truth.item())
        #
        #
        #     # 使用Policy Gradient更新LoRA参数
        #     pg_agent.update([log_probs[0, action]], [torch.tensor(reward)])

        # case2: purely depends on human feedback
        for img, _ in dataloader:  # 假设没有ground truth，或者你不使用它
            img = img.to(device)
            logits = model(img)
            log_probs = F.log_softmax(logits, dim=-1)
            action = torch.argmax(log_probs, dim=-1)  # 假设action是预测的类别

            # 可视化图片并收集反馈
            reward = visualize_image_and_collect_feedback(img.cpu().squeeze(0))

            # 使用Policy Gradient更新LoRA参数
            pg_agent.update([log_probs[0, action]], [torch.tensor([reward], dtype=torch.float, device=device)])


        print(f"Epoch {epoch + 1} completed.")

    # 保存模型
    torch.save(model.state_dict(), "path_to_your_finetuned_model_with_lora.pth")

if __name__ == "__main__":
    main()