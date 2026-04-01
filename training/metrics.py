import torch

class SegmentationMetrics:
    def __init__(self, num_classes=7, ignore_index=6):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
        
    def reset(self):
        self.total_correct = 0.0
        self.total_valid = 0.0
        
        self.intersections = torch.zeros(self.num_classes)
        self.unions = torch.zeros(self.num_classes)
        self.sum_areas = torch.zeros(self.num_classes)
        
    def update(self, preds, labels):
        # preds: [B, C, H, W], labels: [B, H, W]
        pred_classes = preds.argmax(dim=1)
        valid_mask = (labels != self.ignore_index)
        
        # Pixel accuracy accumulation
        self.total_correct += (pred_classes[valid_mask] == labels[valid_mask]).sum().item()
        self.total_valid += valid_mask.sum().item()
        
        # IoU and Dice accumulation per class
        for c in range(self.num_classes):
            if c == self.ignore_index: continue
            
            pred_c = (pred_classes == c)
            gt_c = (labels == c)
            
            self.intersections[c] += (pred_c & gt_c & valid_mask).sum().item()
            self.unions[c] += ((pred_c | gt_c) & valid_mask).sum().item()
            self.sum_areas[c] += (pred_c & valid_mask).sum().item() + (gt_c & valid_mask).sum().item()
            
    def compute(self):
        acc = self.total_correct / self.total_valid if self.total_valid > 0 else 0.0
        
        # Macro averaging
        iou_list = []
        dice_list = []
        
        for c in range(self.num_classes):
            if c == self.ignore_index: continue
            
            if self.unions[c] > 0:
                iou_list.append((self.intersections[c] / self.unions[c]).item())
            if self.sum_areas[c] > 0:
                dice_list.append((2.0 * self.intersections[c] / self.sum_areas[c]).item())
                
        mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
        mean_dice = sum(dice_list) / len(dice_list) if dice_list else 0.0
        
        return {
            "acc": acc,
            "iou": mean_iou,
            "dice": mean_dice
        }
