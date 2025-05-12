import torch

def get_iou_mat(bboxes):
    n_boxes = len(bboxes)

    bboxes_col = bboxes.unsqueeze(0).expand(n_boxes, -1, -1)
    bboxes_row = bboxes.unsqueeze(1).expand(-1, n_boxes, -1)
    area0 = (bboxes_col[..., 2] - bboxes_col[..., 0]) * (bboxes_col[..., 3] - bboxes_col[..., 1])
    area1 = (bboxes_row[..., 2] - bboxes_row[..., 0]) * (bboxes_row[..., 3] - bboxes_row[..., 1])
    tl_x = torch.maximum(bboxes_col[..., 0], bboxes_row[..., 0])
    tl_y = torch.maximum(bboxes_col[..., 1], bboxes_row[..., 1])
    br_x = torch.minimum(bboxes_col[..., 2], bboxes_row[..., 2])
    br_y = torch.minimum(bboxes_col[..., 3], bboxes_row[..., 3])
    area_inter = (br_x - tl_x).clamp(0) * (br_y - tl_y).clamp(0)
    area_union = area0 + area1 - area_inter
    iou = area_inter / area_union
    return iou