import torch

def modify_weight_keys(pretrained_weights_path, output_path):
    """
    修改预训练权重文件中的键名，将特定前缀的键进行重命名
    
    参数:
        pretrained_weights_path: 预训练权重文件路径
        output_path: 修改后保存的文件路径
    """
    # 加载预训练权重
    pretrained_weights = torch.load(pretrained_weights_path)['state_dict']
    
    # 创建新的状态字典
    new_state_dict = {}
    
    # 遍历原始权重字典
    for key, value in pretrained_weights.items():
        # 检查键是否以"blocks."开头
        if key.startswith("blocks."):
            # 将"blocks."替换为"blocks"
            new_key = key.replace("blocks.", "blocks", 1)
            new_state_dict[new_key] = value
        else:
            # 保持其他键不变
            new_state_dict[key] = value
    
    # 保存修改后的权重
    torch.save(new_state_dict, output_path)
    print(f"修改后的权重已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_path = "/home/zhangxiaoshuai/Pretrained/repghostnet_0_5x_43M_66.95.pth.tar"
    output_path = "/home/zhangxiaoshuai/Pretrained/repghostnet_0_5x_43M_66.95_modify.pth.tar"
    
    # 执行修改
    modify_weight_keys(input_path, output_path)