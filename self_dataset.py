from torch.utils.data import Dataset, DataLoader
import pymysql
import numpy as np
import torch
from PIL import Image
import io

class YOLOSQLDataset(Dataset):
    def __init__(self, db_config, img_size=640, subset='train'):
        self.img_size = img_size
        self.subset = subset
        self.conn = pymysql.connect(**db_config)
        self.cursor = self.conn.cursor()
        self.data = self.load_data()
        self._labels = self.load_labels()

    def load_data(self):
        sql = "SELECT id, img_data, annotation FROM images WHERE subset = %s"
        self.cursor.execute(sql, (self.subset,))
        return self.cursor.fetchall()

    def load_labels(self):
        labels = []
        for row in self.data:
            id, img_blob, annotation = row
            bboxes = []
            cls_list = []  # 👈 新增
            if annotation:
                lines = annotation.strip().split('\n')
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        bboxes.append([class_id, x, y, w, h])
                        cls_list.append(class_id)  # 👈 添加 class_id 到 cls_list
            labels.append({
                'bboxes': np.array(bboxes, dtype=np.float32),
                'cls': np.array(cls_list, dtype=np.float32)  # 👈 新增 cls 字段
            })
        return labels

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id, img_blob, annotation = self.data[index]
        image = Image.open(io.BytesIO(img_blob)).convert("RGB")
        orig_w, orig_h = image.size
        ori_shape = (orig_h, orig_w)

        # resize（确保 padding 时保持比例一致）
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).float() / 255.0

        label = self._labels[index]
        bboxes = label['bboxes']
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes, dtype=np.float32)

        if bboxes.ndim == 1:  # ⛔ 避免 [cls, x, y, w, h] 这种变成 1D 的情况
            bboxes = np.expand_dims(bboxes, axis=0)

        if bboxes.shape[0] == 0:
            target = torch.zeros((0, 5), dtype=torch.float32)
        else:
            target = torch.tensor(bboxes, dtype=torch.float32)
        # if len(label['bboxes']) == 0:
        #     target = torch.zeros((0, 5), dtype=torch.float32)
        # else:
        #     target = torch.tensor(label['bboxes'], dtype=torch.float32)  # shape [N, 5]

        # ratio_pad 可以预设为 ((1.0,), (0, 0))，如果你没有做 letterbox
        ratio_pad = ((1.0,), (0, 0))

        return image, target, ori_shape, ratio_pad

    @staticmethod
    def collate_fn(batch):
        imgs, targets, ori_shapes, ratio_pads = list(zip(*batch))

        # [B, 3, H, W]
        imgs = torch.stack(imgs, dim=0)

        batch_idx_list = []
        cls_list = []
        bboxes_list = []
        im_files = []

        for i, boxes in enumerate(targets):
            if boxes.numel() == 0:
                continue

            if boxes.ndim == 1:  # 只包含一个目标时，升维
                boxes = boxes.unsqueeze(0)

            batch_idx = torch.full((boxes.shape[0],), i, dtype=torch.int64)
            batch_idx_list.append(batch_idx)
            cls_list.append(boxes[:, 0].long())
            bboxes_list.append(boxes[:, 1:5].float())
            im_files.append(f"db_image_{i}.jpg")

        if batch_idx_list:
            batch_idx = torch.cat(batch_idx_list, dim=0)
            cls = torch.cat(cls_list, dim=0)
            bboxes = torch.cat(bboxes_list, dim=0)
        else:
            batch_idx = torch.empty((0,), dtype=torch.int64)
            cls = torch.empty((0,), dtype=torch.int64)
            bboxes = torch.empty((0, 4), dtype=torch.float32)

        # 👇 👇 👇 重点：确保 cls 是至少 1D 的 tensor
        if cls.ndim == 0:
            cls = cls.unsqueeze(0)

        return {
            "img": imgs,
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": list(im_files),
            "ori_shape": list(ori_shapes),
            "ratio_pad": list(ratio_pads),
        }


def custom_collate_fn(batch):
    imgs, targets, ori_shapes, ratio_pads = list(zip(*batch))

    # [B, 3, H, W]
    imgs = torch.stack(imgs, dim=0)

    batch_idx_list = []
    cls_list = []
    bboxes_list = []
    im_files = []

    for i, boxes in enumerate(targets):
        if boxes.numel() == 0:
            continue

        # 🔧 修复：避免 1D tensor 导致后续错误
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        # i 表示 batch 中的第几张图片，boxes.shape[0] 是该图中的目标数
        batch_idx = torch.full((boxes.shape[0],), i, dtype=torch.int64)
        batch_idx_list.append(batch_idx)

        cls = boxes[:, 0].long()
        if cls.ndim == 0:
            cls = cls.unsqueeze(0)
        cls_list.append(cls)

        bboxes = boxes[:, 1:5].float()
        if bboxes.ndim == 1:
            bboxes = bboxes.unsqueeze(0)
        bboxes_list.append(bboxes)

        im_files.append(f"db_image_{i}.jpg")  # 模拟 filename

    # 若 batch 中没有目标，也不能崩
    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list, dim=0)
        cls = torch.cat(cls_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)
    else:
        batch_idx = torch.empty((0,), dtype=torch.int64)
        cls = torch.empty((0,), dtype=torch.int64)
        bboxes = torch.empty((0, 4), dtype=torch.float32)

    if cls.ndim == 0:
        cls = cls.unsqueeze(0)
    return {
        "img": imgs,  # [B, 3, H, W]
        "batch_idx": batch_idx,  # [N]
        "cls": cls,  # [N]
        "bboxes": bboxes,  # [N, 4]
        "im_file": list(im_files),  # [B] => 用于 plot_training_samples
        "ori_shape": list(ori_shapes),  # [B] => (h, w)
        "ratio_pad": list(ratio_pads),  # [B] => ((gain,), (pad_w, pad_h))
    }
class Dataloader:
    def __init__(self, db_config, batch_size=16, img_size=640):
        self.dataset = YOLOSQLDataset(db_config, img_size)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 建议从0开始，确认无线程问题后可调高
            collate_fn=custom_collate_fn
        )

    def get_loader(self):
        return self.dataloader
