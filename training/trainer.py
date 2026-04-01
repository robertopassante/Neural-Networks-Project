import os
import shutil
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from .metrics import SegmentationMetrics

class SatMAETrainer:
    def __init__(self, model, train_loader, device, lr=1e-4, checkpoint_dir="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Path specificato verso Drive, fungerà da appoggio se esiste e/o scrivibile
        self.drive_checkpoint_dir = "/content/drive/MyDrive/satellite_segmentation/checkpoints"
        self.results_dir = "training_results"
        
        # Setup and criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=6)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.metric_tracker = SegmentationMetrics(num_classes=7, ignore_index=6)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Variabili di stato (Early Stopping e Resume)
        self.best_iou = 0.0
        self.patience_counter = 0
        self.patience_limit = 3
        self.start_epoch = 0

    def load_checkpoint(self):
        """Cerca l'ultimo checkpoint nel drive o se assente nel path locale per resume automatico."""
        checkpoints_to_try = [
            os.path.join(self.drive_checkpoint_dir, "latest_checkpoint.pth"),
            os.path.join(self.checkpoint_dir, "latest_checkpoint.pth")
        ]
        
        for ckpt_path in checkpoints_to_try:
            if os.path.exists(ckpt_path):
                print(f"[*] Trovato checkpoint, ripristino in corso da: {ckpt_path}")
                try:
                    checkpoint = torch.load(ckpt_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
                    if 'best_iou' in checkpoint:
                        self.best_iou = checkpoint['best_iou']
                    print(f"[✓] Modello e Ottimizzatore ripristinati con successo! Si riparte dall'epoca {self.start_epoch + 1}.")
                    return True
                except Exception as e:
                    print(f"[!] Errore nel caricamento del file {ckpt_path}: {e}")
        
        print(f"[*] Nessun checkpoint esistente trovato. Inizio addestramento da zero.")
        return False

    def save_checkpoint(self, name, epoch, is_best=False):
        """Salva lo stato completo localmente e tenta un backup in cloud/drive."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou
        }
        
        # Salvataggio nel path locale
        local_path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        torch.save(state, local_path)
        
        if is_best:
            best_local = os.path.join(self.checkpoint_dir, "best_model.pth")
            # Evita SameFileError se stiamo già salvando con nome "best_model"
            if os.path.abspath(local_path) != os.path.abspath(best_local):
                shutil.copyfile(local_path, best_local)
            
        # Copia silenziosa in Google Drive (Non bloccherà l'esecuzione in locale/windows)
        drive_base = os.path.dirname(self.drive_checkpoint_dir)
        if os.path.exists(drive_base):
            try:
                os.makedirs(self.drive_checkpoint_dir, exist_ok=True)
                drive_path = os.path.join(self.drive_checkpoint_dir, f"{name}.pth")
                if os.path.abspath(local_path) != os.path.abspath(drive_path):
                    shutil.copyfile(local_path, drive_path)
                if is_best:
                    best_drive = os.path.join(self.drive_checkpoint_dir, "best_model.pth")
                    if os.path.abspath(local_path) != os.path.abspath(best_drive):
                        shutil.copyfile(local_path, best_drive)
            except Exception:
                pass # Eventuali permessi negati non devono interrompere il training

    def visualize_predictions(self, epoch, images, masks, outputs, metrics):
        """Salva un collage visivo Reale/Realtà/Predizione con Metriche a fine epoca."""
        try:
            import matplotlib
            matplotlib.use('Agg') # Evita errori di visualizzazione GUI in env headless
            
            # Preleviamo l'immagine 0 del batch
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            
            # De-normalizzazione Imagenet Standard per visualizzare RGB coerentemente
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            true_mask = masks[0].cpu().numpy()
            pred_mask = outputs[0].argmax(dim=0).cpu().numpy()
            
            # Ottengo le metriche globali computate e formatto in percentuale
            m_dice = metrics.get('dice', 0.0) * 100
            m_iou = metrics.get('iou', 0.0) * 100

            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            
            axes[0].set_title('Satellite (Immagine Reale)', fontsize=14)
            axes[0].imshow(img)
            axes[0].axis('off')

            axes[1].set_title('Dataset (Maschera Originale)', fontsize=14)
            axes[1].imshow(true_mask, cmap='nipy_spectral', vmin=0, vmax=6)
            axes[1].axis('off')

            axes[2].set_title(f'Predizione del Modello\nIoU: {m_iou:.2f}% | Dice: {m_dice:.2f}%', fontsize=14)
            axes[2].imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=6)
            axes[2].axis('off')

            plt.tight_layout()
            save_path = os.path.join(self.results_dir, f"epoch_{epoch+1}_visual_result.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[!] Impossibile disegnare le figure per questa epoca: {e}")

    def train_epoch(self, epoch_idx, total_epochs):
        self.model.train()
        running_loss = 0.0
        self.metric_tracker.reset()
        
        # Barra TQDM con formato barra fluido e indicazione Loss running
        desc_str = f"Epoca {epoch_idx+1}/{total_epochs}"
        pbar = tqdm(self.train_loader, desc=desc_str, unit="batch", leave=False, 
                    bar_format="{l_bar}{bar:20}{r_bar}")
        
        # Variabili cache utilizzate poi per disegnare the graph a fine epoch
        last_images, last_masks, last_outputs = None, None, None

        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            # Aggiornamento accumulativo metriche
            running_loss += loss.item() * images.size(0)
            self.metric_tracker.update(outputs.detach(), masks.detach())
            
            # Postfix dinamico per mostrare all'utente in real time la perdita sul batch
            pbar.set_postfix({'Live Loss': f"{loss.item():.4f}"})
            
            # Memo dell'ultimo intero batch processato
            last_images, last_masks, last_outputs = images.detach(), masks.detach(), outputs.detach()
            
        # Ottengo le metriche dell'intera Epoca elaborate correttamente dal tracker globale
        final_metrics = self.metric_tracker.compute()
        epoch_stats = {
            "loss": running_loss / len(self.train_loader.dataset),
            "acc": final_metrics["acc"],
            "iou": final_metrics["iou"],
            "dice": final_metrics["dice"]
        }
        
        # Trigger del plotter delle matrici su pyplot
        if last_images is not None:
             self.visualize_predictions(epoch_idx, last_images, last_masks, last_outputs, epoch_stats)
             
        return epoch_stats

    def train(self, num_epochs=10):
        print(f"[*] Inizio pipeline di training su device: {self.device}")
        
        # Tenta di ristabilire l'addestramento da dove lo si è fermato (se esiste file su gdrive/locale)
        self.load_checkpoint()
        
        if self.start_epoch >= num_epochs:
            print(f"[!] Attenzione: Il modello ha già traguardato le {num_epochs} epoche imposte nel checkpoint. Training bypassato.")
            return

        try:
            for epoch in range(self.start_epoch, num_epochs):
                
                # Esecuzione Epoca e recupero statistiche esatte
                stats = self.train_epoch(epoch, num_epochs)
                
                # Stampa Dettagliata Richiesta di Fine Epoca
                print(f"--- Info Epoca: {epoch+1:02d}/{num_epochs} ---")
                print(f"  > Loss Totale Dataset : {stats['loss']:.5f}")
                print(f"  > IoU Generale (Macro): {stats['iou']*100:.2f}%")
                print(f"  > Dice Coefficient    : {stats['dice']*100:.2f}%")
                
                current_iou = stats['iou']
                
                # Snapshot backup generico di update epoca su locale/cloud 
                self.save_checkpoint("latest_checkpoint", epoch)
                
                # Logica per l'Early Stopping ed Exporting the Best model basato sull'IoU
                if current_iou > self.best_iou:
                    print(f"  [✓] Nuovo Top IoU Raggiunto! ({self.best_iou*100:.2f}% -> {current_iou*100:.2f}%)")
                    self.best_iou = current_iou
                    self.patience_counter = 0
                    self.save_checkpoint("best_model", epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    print(f"  [!] L'IoU non è migliorato per {self.patience_counter} epoche (Pazienza limite: {self.patience_limit}).")
                    
                # Scatta l'interruttore dell'Early Stopping
                if self.patience_counter >= self.patience_limit:
                    print(f"\n[!] EARLY STOPPING CONSEGUITO all'Epoca {epoch+1}. Terminazione forzata anti-overfitting.")
                    break
                    
                print("-" * 35)
                
            print("\n[✓] Sessione di Addestramento conclusa in toto.")
            
        except KeyboardInterrupt:
            # Safe interject se viene pigiato CTRL-C
            current_ep = locals().get('epoch', self.start_epoch)
            print(f"\n[!] INTERROTTO MANUALMENTE: Inizializzo backup d'emergenza all'epoca {current_ep+1}...")
            self.save_checkpoint("latest_checkpoint", current_ep)
            print("[✓] Modello salvato nel file di latest_checkpoint locale/gdrive in sicurezza.")
