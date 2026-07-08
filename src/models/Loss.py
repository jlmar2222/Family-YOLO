import torch
import torch.nn as nn

from src.utils.IoU import IoU_yolo


class YoloLossv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLossv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.alpha_coord = 5.0

        self.alpha_noobj = 0.5

    def forward(self, predictions, target):
        # Reshape predictions to (batch_size, S, S, B*5 + C)
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        # Split predictions into components
        pred_boxes = predictions[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5) # (batch_size, S, S, B, 5)
        pred_classes = predictions[..., self.B * 5:]

        # Split target into components
        target_boxes = target[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5) # CREO QUE DEBERIA DE SER 4? O ESTA BIEN ASI REVISAR
        target_classes = target[..., self.B * 5:]


        ious = IoU_yolo(
            pred_boxes[..., :4], target_boxes[..., :4], self.S
        ) 

        best_iou, best_box_idx = torch.max(ious, dim=3, keepdim=True)  # (N,S,S,1,1)

        best_iou = best_iou.detach()

        box_range = torch.arange(self.B, device=predictions.device).view(1, 1, 1, self.B, 1)
        responsible = (box_range == best_box_idx).float()  # one-hot sobre B, (N,S,S,B,1)

        # existe objeto/bbox en la celda? miramos si la celda tiene confidence = 1 en el target
        exists_box = target_boxes[..., 0:1, 4:5]  # (N,S,S,1,1)

        obj_mask = responsible * exists_box 

        noobj_mask = 1.0 - obj_mask  

        exist_object = exists_box.squeeze(3) # (N,S,S,1)

        # Proper Loss (Paper Loss):

        # (1) Center Loss
        coord_loss = torch.sum(obj_mask*(pred_boxes[..., :2] - target_boxes[..., :2]) ** 2)
        # (2) Size Loss        
        size_loss = torch.sum(obj_mask*((torch.sign(pred_boxes[..., 2:4]) * torch.sqrt(torch.abs(pred_boxes[..., 2:4]) + 1e-6)) - torch.sqrt(target_boxes[..., 2:4])) ** 2)

        # (3) Confidence Loss
        conf_loss_obj = torch.sum(obj_mask*(pred_boxes[..., 4:5] - best_iou) ** 2)
        conf_loss_noobj = torch.sum(noobj_mask*(pred_boxes[..., 4:5] - target_boxes[..., 4:5]) ** 2)

        # (4) Class Loss
        class_loss = torch.sum(exist_object*(pred_classes - target_classes) ** 2)

        # Sum everything up
        total_loss = (self.alpha_coord * coord_loss + self.alpha_coord * size_loss +
                    (conf_loss_obj + self.alpha_noobj * conf_loss_noobj) + class_loss)
        
        return total_loss
    
