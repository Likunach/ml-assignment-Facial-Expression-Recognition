import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

def train(model, train_loader, val_loader, config, device='cpu', checkpoint_dir=None):
    run_name = config['run_name']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config.get('weight_decay', 0)
    optimizer_name = config.get('optimizer', 'adam')
    use_scheduler = config.get('scheduler', False)

    # optimizer
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    criterion = nn.CrossEntropyLoss()

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )

    model.to(device)
    best_val_acc = 0.0

    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    for epoch in tqdm(range(epochs), desc=f'{run_name} | Epochs'):
        # ── training ──
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        batch_bar = tqdm(train_loader, desc=f'  Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for images, labels in batch_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # ── validation ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'  Epoch {epoch+1}/{epochs} [Val]', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        if scheduler:
            scheduler.step(val_acc)

        # ── wandb logging ──
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        # ── save best ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

        tqdm.write(
            f'Epoch {epoch+1}/{epochs} | '
            f'train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}'
        )

    print(f'Best val_acc: {best_val_acc:.4f} — saved to {checkpoint_path}')
    return best_val_acc