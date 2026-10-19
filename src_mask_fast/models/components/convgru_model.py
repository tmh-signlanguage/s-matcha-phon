import torch
import torch.nn as nn


class ConvGRUPredictor(nn.Module):
    def __init__(self, input_dim=2, conv_hidden=64, gru_hidden=256, 
                 num_clusters=1024, num_gru_layers=3, dropout=0.3):
        super().__init__()
        
        # Multi-layer 1D convolutions for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, conv_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            conv_hidden, 
            gru_hidden, 
            num_gru_layers,
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # Classifier head with intermediate layer
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_clusters)
        )
        
    def forward(self, x):
        # x: [batch, seq_len=30, 2]
        batch_size = x.size(0)
        
        # Conv expects [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, 2, 30]
        x = self.conv_layers(x)  # [batch, conv_hidden, 30]
        x = x.transpose(1, 2)  # [batch, 30, conv_hidden]
        
        # GRU
        x, _ = self.gru(x)  # [batch, 30, gru_hidden*2]
        
        # Classify each timestep
        logits = self.classifier(x)  # [batch, 30, 1024]
        
        return logits



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # pred: [batch*seq_len, num_clusters]
        # target: [batch*seq_len]
        n_classes = pred.size(-1)
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        
        # True label gets (1-smoothing), others share smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


def train_model(model, train_loader, val_loader, num_epochs=150, 
                lr=0.001, use_label_smoothing=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)  # [batch, 30, 1024]
            
            # Reshape for loss computation
            batch_size, seq_len, num_classes = logits.shape
            loss = criterion(
                logits.reshape(-1, num_classes), 
                y.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Accuracy
            pred = logits.argmax(dim=-1)
            train_correct += (pred == y).sum().item()
            train_total += y.numel()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), 
                    y.reshape(-1)
                )
                val_loss += loss.item()
                
                pred = logits.argmax(dim=-1)
                val_correct += (pred == y).sum().item()
                val_total += y.numel()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, 'best_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model
