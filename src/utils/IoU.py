import torch


def IoU_yolo(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, S: int) -> torch.Tensor:
    """
    Calcula el IoU entre cajas codificadas en formato YOLO (como las produce VocDataset),
    SIN necesitar el dataframe original: todo se deriva del propio tensor + S.

    Formato de entrada (por celda):
        x, y -> posición del centro RELATIVA a la celda, en [0, 1]
        w, h -> ancho/alto RELATIVOS a la imagen completa, en [0, 1]

    Args:
        boxes_pred:   (N, S, S, B, 4) -> (x, y, w, h)
        boxes_target: (N, S, S, B, 4) -> (x, y, w, h)
        S: número de celdas por lado de la grid

    Returns:
        iou: (N, S, S, B, 1)
    """
    device = pred_boxes.device

    # grid_i -> índice de columna (eje x), grid_j -> índice de fila (eje y)
    grid_i = torch.arange(S, device=device).float().view(1, 1, S, 1, 1)
    grid_j = torch.arange(S, device=device).float().view(1, S, 1, 1, 1)
   

    def to_corners(boxes: torch.Tensor):
        x = boxes[..., 0:1]
        y = boxes[..., 1:2]
        w = boxes[..., 2:3]
        h = boxes[..., 3:4]

        # Centro en coordenadas normalizadas a TODA la imagen (para poder comparar
        # ambas cajas en el mismo sistema de referencia, aunque x,y sean cell-relative)
    
        # x = (cx / S_w) - i --> cx = (i + x) * S_w pero esto seria sin normalizar a image size
        # como debemos normalizar para poder operar con w y h que si lo estan, acabamso con la formula de abajo
        cx = (grid_i + x) /S
        cy = (grid_j + y) / S

        x_top_left = cx - (w / 2)
        y_top_left = cy - (h / 2)
        x_bottom_right = cx + (w / 2)
        y_bottom_right = cy + (h / 2)

        return x_top_left, y_top_left, x_bottom_right, y_bottom_right

    p_x1, p_y1, p_x2, p_y2 = to_corners(pred_boxes)
    t_x1, t_y1, t_x2, t_y2 = to_corners(target_boxes)

    # Calculamos Intersection Area
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Calculamos Union Area
    p_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    t_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)

    union = p_area + t_area - inter_area + 1e-6

    IoU = inter_area / union # (N, S, S, B, 1), IoU is in B dimension.

    return IoU