import torch
import torch.nn as nn


class YoloLossv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLossv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.alpha_coord = 5.0
        self_alpha_noobj = 0.5

    def forward(self, predictions, target):
        # Reshape predictions to (batch_size, S, S, B*5 + C)
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        # Split predictions into components
        pred_boxes = predictions[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B * 5:]

        # Split target into components
        target_boxes = target[..., :self.B * 5].view(-1, self.S, self.S, self.B, 5)
        target_classes = target[..., self.B * 5:]



        # FALTA INTEGRAR LO DE LA IoU Y PONER SOLO EL QUE GANE DE TODA LA DIMENSION B. 

        # Proper Loss (Paper Loss):

        # (1) Center Loss
        coord_loss = torch.sum((pred_boxes[..., :2] - target_boxes[..., :2]) ** 2)
        # (2) Size Loss
        size_loss = torch.sum((pred_boxes[..., 2:4] - target_boxes[..., 2:4]) ** 2)
        # (3) Confidence Loss
        conf_loss = torch.sum((pred_boxes[..., 4] - target_boxes[..., 4]) ** 2)
        
        # (4) Class Loss
        class_loss = torch.sum((pred_classes - target_classes) ** 2)


        # CUIDADO CON LA CONF LOSS QUE HAY QUE VALIDAR SI HAY OBJETO O NO HAY OBJETO
        # TODO ESTO LO PODEMOS HACER MAÑANA

        total_loss = self.alpha_coord * coord_loss + self.alpha_coord * size_loss + (conf_loss + self.alpha_noobj * conf_loss) + class_loss
        
        return total_loss