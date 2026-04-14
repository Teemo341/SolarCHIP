import torch
from torch.utils.data import Dataset

class RandomImageDataset(Dataset):
    def __init__(self, modal_list=['hmi','0094','0131','0171','0193','0211','0304','0335','1600','1700','4500'],
                 log1p_scale=1,
                 load_imgs=False,
                 enhance_list=[224,0.5,90],
                 time_interval=[0,5400],
                 time_step=1):
        """随机多模态数据集，初始化参数与 SolarDataset.multimodal_dataset 保持一致。"""
        self.modal_list = modal_list
        self.log1p_scale = log1p_scale
        self.load_imgs = load_imgs
        self.enhance_list = enhance_list
        self.time_interval = time_interval
        self.time_step = time_step

        start, end = int(time_interval[0]), int(time_interval[1])
        step = int(time_step)
        if step <= 0:
            raise ValueError('time_step should be positive')
        if end <= start:
            raise ValueError('time_interval should satisfy end > start')

        self.exist_idx = torch.arange(start, end, step).tolist()
        self.image_size = int(enhance_list[0])

    def __len__(self):
        return len(self.exist_idx)

    def __getitem__(self, idx):
        image_dict = {}
        for modal in self.modal_list:
            image_dict[modal] = torch.randn(1, self.image_size, self.image_size)
        return image_dict


if __name__ == "__main__":
    dataset = RandomImageDataset()
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch in dataloader:
        print(batch['hmi'].shape)  # 输出 (4, 1, 224, 224)
        print(batch['0094'].shape)  # 输出 (4, 1, 224, 224)
        break
