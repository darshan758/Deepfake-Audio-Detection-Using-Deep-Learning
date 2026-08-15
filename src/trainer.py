import torch
from tqdm import tqdm
import os

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.start_epoch = 0  # to track resumed epoch

    def train(self, train_loader, val_loader=None, epochs=15, save_dir="../saved"):
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

        # ✅ Try to resume from checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epoch = checkpoint["epoch"]
            print(f"🔁 Resuming training from epoch {self.start_epoch + 1}...\n")
        else:
            print("🚀 Starting new training session...\n")

        # ✅ Continue training from where we left off
        for epoch in range(self.start_epoch + 1, self.start_epoch + epochs + 1):
            self.model.train()
            running_loss = 0.0

            progress = tqdm(train_loader, desc=f"Epoch [{epoch}/{self.start_epoch + epochs}]")
            for batch_x, batch_y in progress:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = running_loss / len(train_loader)
            print(f"✅ Epoch [{epoch}/{self.start_epoch + epochs}] | Avg Loss: {avg_loss:.4f}")

            # ✅ Validation
            if val_loader:
                val_loss, acc = self.evaluate(val_loader)
                print(f"🧪 Validation -> Loss: {val_loss:.4f} | Accuracy: {acc:.2f}%")

            # ✅ Save checkpoint properly
            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }, checkpoint_path)
            print(f"💾 Checkpoint saved successfully at epoch {epoch}\n")

        print("🎉 Training completed successfully!")

    def evaluate(self, data_loader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / len(data_loader)
        acc = 100 * correct / total
        return avg_loss, acc
