#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# XAND-Ray — Chest X-Ray Screening Research.
# © 2026 XOREngine
# https://github.com/XOREngine/xand-ray


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
import mlflow
import sys
from tqdm import tqdm
import pandas as pd
import inspect
import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.trainConfig import trainConfig
from src.data.chestXrayDataset import ChestXRayDataset
from src.data.imageTransforms import get_train_transforms, get_val_transforms
from src.models.binaryClassifier import BinaryClassifier
from src.models.focalLoss import MultiTaskLoss
from src.utils.aucMetrics import compute_metrics


class TeeStream:
    def __init__(self, original):
        self.original = original
        self.buffer = []
    
    def write(self, data):
        self.original.write(data)
        if '\r' not in data or '\n' in data:
            self.buffer.append(data.replace('\r', ''))
    
    def flush(self):
        self.original.flush()
    
    def get_log(self):
        return "".join(self.buffer)


tee_stdout = None
mlflow_run_active = False
interrupt_handled = False


def save_log_and_exit(status="KILLED", reason="user_interrupt"):
    global tee_stdout, mlflow_run_active
    if mlflow_run_active:
        if tee_stdout:
            mlflow.log_text(tee_stdout.get_log(), "console_log.txt")
        mlflow.set_tag("abort_reason", reason)
        mlflow.end_run(status=status)
    sys.exit(0)


def handle_interrupt(signum, frame):
    global interrupt_handled
    if interrupt_handled:
        return
    interrupt_handled = True
    print("\n[INTERRUPTED] Saving log and closing MLflow...")
    save_log_and_exit(status="KILLED", reason="user_interrupt")


def worker_init_fn(worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps, clip_value, aux_tasks):
    model.train()
    
    running_losses = {'total': 0.0, 'disease': 0.0, 'device': 0.0}
    for task in aux_tasks:
        running_losses[task] = 0.0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch['image'].to(device)
        
        targets = {
            'disease': batch['disease_label'].to(device).unsqueeze(1),
            'device': batch['device_label'].to(device).unsqueeze(1),
        }
        
        if 'projection' in aux_tasks:
            targets['projection'] = batch['projection_label'].to(device).unsqueeze(1)
        if 'sex' in aux_tasks:
            targets['sex'] = batch['sex_label'].to(device).unsqueeze(1)
        if 'age' in aux_tasks:
            targets['age'] = batch['age_normalized'].to(device).unsqueeze(1)
        
        with autocast(device_type=device.type, enabled=device.type == 'cuda'):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses['total'] / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_losses['total'] += losses['total'].item()
        running_losses['disease'] += losses['disease'].item()
        if 'device' in losses:
            running_losses['device'] += losses['device'].item()
        for task in aux_tasks:
            if task != 'device' and task in losses:
                running_losses[task] += losses[task].item()
    
    n = len(dataloader)
    avg_losses = {k: v / n for k, v in running_losses.items()}
    peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0.0
    
    return avg_losses, peak_vram


def validate(model, dataloader, criterion, device, aux_tasks, desc="Validation"):
    model.eval()
    
    running_loss = 0.0
    preds = {'disease': [], 'device': []}
    targets_all = {'disease': [], 'device': []}
    
    for task in aux_tasks:
        preds[task] = []
        targets_all[task] = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['image'].to(device)
            
            targets = {
                'disease': batch['disease_label'].to(device).unsqueeze(1),
                'device': batch['device_label'].to(device).unsqueeze(1),
            }
            
            if 'projection' in aux_tasks:
                targets['projection'] = batch['projection_label'].to(device).unsqueeze(1)
            if 'sex' in aux_tasks:
                targets['sex'] = batch['sex_label'].to(device).unsqueeze(1)
            if 'age' in aux_tasks:
                targets['age'] = batch['age_normalized'].to(device).unsqueeze(1)
            
            outputs = model(images)
            losses = criterion(outputs, targets)
            running_loss += losses['total'].item()
            
            preds['disease'].extend(torch.sigmoid(outputs['disease']).cpu().numpy().flatten())
            targets_all['disease'].extend(targets['disease'].cpu().numpy().flatten())
            
            if 'device' in outputs:
                preds['device'].extend(torch.sigmoid(outputs['device']).cpu().numpy().flatten())
                targets_all['device'].extend(targets['device'].cpu().numpy().flatten())
            
            for task in aux_tasks:
                if task != 'device' and task in outputs:
                    if task == 'age':
                        preds[task].extend(outputs[task].cpu().numpy().flatten())
                    else:
                        preds[task].extend(torch.sigmoid(outputs[task]).cpu().numpy().flatten())
                    targets_all[task].extend(targets[task].cpu().numpy().flatten())
    
    avg_loss = running_loss / len(dataloader)
    
    metrics = {
        'disease': compute_metrics(targets_all['disease'], preds['disease']),
        'device': compute_metrics(targets_all['device'], preds['device']) if preds['device'] else {'auc': 0.0},
    }
    
    for task in aux_tasks:
        if task not in ['age', 'device'] and preds[task]:
            metrics[task] = compute_metrics(targets_all[task], preds[task])
    
    return avg_loss, metrics


def print_dataset_table(datasets, disease_col):
    all_cols = set()
    for _, df in datasets:
        all_cols.update(df.columns)

    has_device = 'device_label' in all_cols
    has_proj = 'projection_label' in all_cols
    has_sex = 'sex_label' in all_cols
    has_patients = 'patient_id' in all_cols

    headers = ["Dataset", "Samples", "POS", "NEG", "%POS"]
    widths = [11, 7, 6, 6, 5]
    if has_device:
        headers.append("Dev%")
        widths.append(5)
    if has_proj:
        headers.append("PA%")
        widths.append(5)
    if has_sex:
        headers.append("M%")
        widths.append(5)
    if has_patients:
        headers.append("Patients")
        widths.append(8)

    def hline(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def row(vals):
        cells = []
        for v, w in zip(vals, widths):
            cells.append(f" {v:>{w}} ")
        return "│" + "│".join(cells) + "│"

    print(hline("┌", "┬", "┐"))
    print(row(headers))
    print(hline("├", "┼", "┤"))

    for label, df in datasets:
        n = len(df)
        pos = int(df[disease_col].sum())
        neg = n - pos
        pct = f"{pos/n*100:.1f}" if n > 0 else "0.0"

        vals = [label, f"{n:,}", f"{pos:,}", f"{neg:,}", pct]

        if has_device:
            if 'device_label' in df.columns:
                vals.append(f"{df['device_label'].mean()*100:.0f}")
            else:
                vals.append("-")
        if has_proj:
            if 'projection_label' in df.columns:
                vals.append(f"{df['projection_label'].mean()*100:.0f}")
            else:
                vals.append("-")
        if has_sex:
            if 'sex_label' in df.columns:
                vals.append(f"{df['sex_label'].mean()*100:.0f}")
            else:
                vals.append("-")
        if has_patients:
            if 'patient_id' in df.columns:
                vals.append(f"{df['patient_id'].nunique():,}")
            else:
                vals.append("-")

        print(row(vals))

    print(hline("└", "┴", "┘"))


def print_config(cfg, csv_train, csv_valid_split, csv_valid_real, csv_test_real, train_df, val_split_df, val_real_df, test_real_df, disease_col):
    aux_tasks = cfg.loss.aux_tasks

    print("\n" + "="*70)
    print("CONFIG")
    print("="*70)

    print(f"[data]     disease: {cfg.data.disease} | u_policy: {cfg.data.u_policy}")
    print(f"[data]     input_size: {cfg.data.input_size} | batch_size: {cfg.data.batch_size}")
    print(f"[data]     num_workers: {cfg.data.num_workers} | pin_memory: {cfg.data.pin_memory}")

    print(f"[model]    backbone: {cfg.model.backbone} | pretrained: {cfg.model.pretrained}")
    print(f"[model]    multitask: {cfg.model.multitask} | grad_ckpt: {cfg.model.gradient_checkpointing}")

    print(f"[loss]     type: {cfg.loss.disease_type} | \u03b3: {cfg.loss.focal_gamma}")
    print(f"[loss]     aux_tasks: {aux_tasks}")

    print(f"[training] epochs: {cfg.training.epochs} | lr: {cfg.training.lr} | wd: {cfg.training.weight_decay}")
    print(f"[training] grad_accum: {cfg.training.gradient_accumulation} | grad_clip: {cfg.training.gradient_clip} | early_stop: {cfg.training.early_stopping_patience}")

    print(f"[optim]    betas: {cfg.optimizer.betas}")
    print(f"[sched]    min_lr: {cfg.scheduler.min_lr}")

    print(f"[hardware] device: {cfg.hardware.device} | gpu: {cfg.hardware.gpu_name}")
    print(f"[mlflow]   experiment: {cfg.mlflow.experiment_name}")

    print("-"*70)
    print(f"[dataset]  name: {cfg.data.dataset_name} | disease_col: {disease_col}")
    print(f"[dataset]  csv_train: {csv_train}")
    print(f"[dataset]  csv_valid_split: {csv_valid_split}")
    print(f"[dataset]  csv_valid_real: {csv_valid_real}")
    print(f"[dataset]  csv_test_real: {csv_test_real if csv_test_real else 'N/A'}")
    print("-"*70)
    datasets = [
        ("train", train_df),
        ("valid_split", val_split_df),
        ("valid_real", val_real_df),
    ]
    if test_real_df is not None:
        datasets.append(("test_real", test_real_df))
    print_dataset_table(datasets, disease_col)

    print("="*70 + "\n")


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, config_path, aux_tasks):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config_path': config_path,
        'aux_tasks': list(aux_tasks.keys()) if aux_tasks else [],
    }, checkpoint_path)


def apply_image_overrides(df, overrides):
    for pair in overrides:
        old, new = pair.split('=', 1)
        df['image_path'] = df['image_path'].str.replace(old, new, regex=False)
        ext_map = {'.jpg': '.png', '.jpeg': '.png'}
        for old_ext, new_ext in ext_map.items():
            mask = df['image_path'].str.endswith(old_ext)
            if mask.any():
                df.loc[mask, 'image_path'] = df.loc[mask, 'image_path'].str[:-len(old_ext)] + new_ext
    return df


def main(config_path: str, disease: str, csv_train: str, csv_valid_split: str, csv_valid_real: str, csv_test_real: str = None, resume_from: str = None, gpu_index: int = 0, images_root_override: list = None):
    global tee_stdout, mlflow_run_active, interrupt_handled
    
    tee_stdout = TeeStream(sys.stdout)
    sys.stdout = tee_stdout
    
    signal.signal(signal.SIGINT, handle_interrupt)
    
    cfg = trainConfig.from_yaml(config_path, disease, gpu_index)
    device = torch.device(cfg.hardware.device)
    aux_tasks = cfg.loss.aux_tasks
    
    train_df = pd.read_csv(csv_train)
    val_split_df = pd.read_csv(csv_valid_split)
    val_real_df = pd.read_csv(csv_valid_real)
    test_real_df = pd.read_csv(csv_test_real) if csv_test_real else None
    
    if images_root_override:
        print(f"[OVERRIDE] Rules: {images_root_override}")
        apply_image_overrides(train_df, images_root_override)
        apply_image_overrides(val_split_df, images_root_override)
        sample = train_df['image_path'].iloc[0]
        print(f"[OVERRIDE] Sample: {sample}")
        if not Path(sample).exists():
            raise FileNotFoundError(f"Override path not found: {sample}")
    
    disease_cols = [c for c in train_df.columns if c.startswith('disease_')]
    if not disease_cols:
        raise ValueError("No disease column found")
    disease_col = disease_cols[0]
    
    print_config(cfg, csv_train, csv_valid_split, csv_valid_real, csv_test_real, train_df, val_split_df, val_real_df, test_real_df, disease_col)
    
    disease_slug = cfg.data.disease.lower().replace(' ', '_')
    run_name = f"{disease_slug}_{cfg.model.backbone}_{cfg.data.input_size}px_{cfg.data.u_policy}"
    if aux_tasks:
        run_name += f"_aux{len(aux_tasks)}"
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow_run_active = True
        print(f"[mlflow]   run_name: {run_name}")
        print(f"[mlflow]   run_id: {run.info.run_id}")
        print("-"*70)
        
        mlflow.log_params({
            'backbone': cfg.model.backbone,
            'pretrained': cfg.model.pretrained,
            'multitask': cfg.model.multitask,
            'aux_tasks': list(aux_tasks.keys()) if aux_tasks else [],
        })
        mlflow.log_params({
            'epochs': cfg.training.epochs,
            'lr': cfg.training.lr,
            'weight_decay': cfg.training.weight_decay,
            'gradient_accumulation': cfg.training.gradient_accumulation,
            'gradient_clip': cfg.training.gradient_clip,
            'early_stopping_patience': cfg.training.early_stopping_patience,
        })
        mlflow.log_params({
            'disease_type': cfg.loss.disease_type,
            'focal_gamma': cfg.loss.focal_gamma,
        })
        for task, weight in aux_tasks.items():
            mlflow.log_param(f"aux_{task}_weight", weight)
        
        mlflow.log_params({
            'disease': cfg.data.disease, 
            'u_policy': cfg.data.u_policy,
            'input_size': cfg.data.input_size,
            'batch_size': cfg.data.batch_size,
        })
        mlflow.log_params({
            'dataset_name': cfg.data.dataset_name,
            'csv_train': csv_train,
            'csv_valid_split': csv_valid_split,
            'csv_valid_real': csv_valid_real,
            'csv_test_real': csv_test_real if csv_test_real else 'N/A',
            'disease_col': disease_col,
            'train_samples': len(train_df),
            'valid_split_samples': len(val_split_df),
            'valid_real_samples': len(val_real_df),
            'test_real_samples': len(test_real_df) if test_real_df is not None else 0,
            'train_pos_ratio': train_df[disease_col].mean(),
            'valid_split_pos_ratio': val_split_df[disease_col].mean(),
            'valid_real_pos_ratio': val_real_df[disease_col].mean(),
        })
        if test_real_df is not None:
            mlflow.log_param('test_real_pos_ratio', test_real_df[disease_col].mean())
        
        mlflow.set_tag("dataset", cfg.data.dataset_name)
        mlflow.set_tag("resolution", cfg.data.input_size)
        mlflow.set_tag("disease", cfg.data.disease)
        mlflow.set_tag("gpu", cfg.hardware.gpu_name)
        mlflow.set_tag("aux_tasks", ",".join(aux_tasks.keys()) if aux_tasks else "none")
        mlflow.set_tag("has_test_real", "yes" if csv_test_real else "no")
        if cfg.mlflow.description:
            mlflow.set_tag("description", cfg.mlflow.description)
        
        train_transforms = get_train_transforms(cfg.data.input_size)
        val_transforms = get_val_transforms(cfg.data.input_size)
        
        mlflow.log_text(inspect.getsource(get_train_transforms), "transforms/train_transforms.py")
        mlflow.log_text(inspect.getsource(get_val_transforms), "transforms/val_transforms.py")
        
        train_dataset = ChestXRayDataset(train_df, transforms=train_transforms)
        val_split_dataset = ChestXRayDataset(val_split_df, transforms=val_transforms)
        val_real_dataset = ChestXRayDataset(val_real_df, transforms=val_transforms)
        test_real_dataset = ChestXRayDataset(test_real_df, transforms=val_transforms) if test_real_df is not None else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            worker_init_fn=worker_init_fn,
        )
        
        val_split_loader = DataLoader(
            val_split_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            worker_init_fn=worker_init_fn,
        )
        
        val_real_loader = DataLoader(
            val_real_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            worker_init_fn=worker_init_fn,
        )
        
        test_real_loader = None
        if test_real_dataset is not None:
            test_real_loader = DataLoader(
                test_real_dataset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                worker_init_fn=worker_init_fn,
            )
        
        print(f"Creating model: {cfg.model.backbone}")
        if aux_tasks:
            print(f"Auxiliary tasks: {list(aux_tasks.keys())}")
        
        model = BinaryClassifier(
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            multitask=cfg.model.multitask,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
            aux_tasks=aux_tasks,
        ).to(device)
        
        if resume_from:
            print(f"Resuming from {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        criterion = MultiTaskLoss(cfg.loss)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            betas=cfg.optimizer.betas,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.scheduler.min_lr,
        )
        
        scaler = GradScaler(enabled=device.type == 'cuda')
        
        best_auc_split = 0.0
        best_auc_valid_real = 0.0
        best_auc_test_real = 0.0
        auc_test_at_best_valid = 0.0
        patience_counter = 0
        
        checkpoint_dir = Path('outputs/checkpoints')
        
        ckpt_best_split = checkpoint_dir / f"{disease_slug}_best_split.pt"
        ckpt_best_valid_real = checkpoint_dir / f"{disease_slug}_best_valid_real.pt"
        ckpt_peak_test_real = checkpoint_dir / f"{disease_slug}_peak_test_real.pt"
        
        for epoch in range(cfg.training.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.training.epochs}")
            print("-"*50)
            
            train_losses, peak_vram = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                cfg.training.gradient_accumulation,
                cfg.training.gradient_clip,
                aux_tasks,
            )
            
            val_split_loss, metrics_split = validate(
                model, val_split_loader, criterion, device, aux_tasks, desc="Val Split",
            )
            
            val_real_loss, metrics_valid_real = validate(
                model, val_real_loader, criterion, device, aux_tasks, desc="Val Real",
            )
            
            metrics_test_real = None
            if test_real_loader is not None:
                _, metrics_test_real = validate(
                    model, test_real_loader, criterion, device, aux_tasks, desc="Test Real",
                )
            
            scheduler.step()
            
            auc_split = metrics_split['disease']['auc']
            auc_valid_real = metrics_valid_real['disease']['auc']
            auc_test_real = metrics_test_real['disease']['auc'] if metrics_test_real else 0.0
            
            auc_pr_split = metrics_split['disease']['auc_pr']
            auc_pr_valid_real = metrics_valid_real['disease']['auc_pr']
            auc_pr_test_real = metrics_test_real['disease']['auc_pr'] if metrics_test_real else 0.0
            
            device_auc_split = metrics_split['device']['auc']
            device_auc_valid_real = metrics_valid_real['device']['auc']
            device_auc_test_real = metrics_test_real['device']['auc'] if metrics_test_real else 0.0
            
            current_lr = optimizer.param_groups[0]['lr']
            
            rows = [
                ("Loss", f"{val_split_loss:.4f}", f"{val_real_loss:.4f}", "-"),
                ("AUC disease", f"{auc_split:.4f}", f"{auc_valid_real:.4f}", f"{auc_test_real:.4f}"),
                ("AUC-PR disease", f"{auc_pr_split:.4f}", f"{auc_pr_valid_real:.4f}", f"{auc_pr_test_real:.4f}"),
                ("AUC device", f"{device_auc_split:.4f}", f"{device_auc_valid_real:.4f}", f"{device_auc_test_real:.4f}"),
            ]
            for task in aux_tasks:
                if task not in ['age', 'device'] and task in metrics_split:
                    rows.append((
                        f"AUC {task}",
                        f"{metrics_split[task]['auc']:.4f}",
                        f"{metrics_valid_real[task]['auc']:.4f}" if task in metrics_valid_real else "-",
                        "-"
                    ))
            
            print(f"\nTrain Loss: {train_losses['total']:.4f} (disease: {train_losses['disease']:.4f})")
            print("┌─────────────────┬─────────────┬────────────┬───────────┐")
            print("│ Metric          │ Valid Split │ Valid Real │ Test Real │")
            print("├─────────────────┼─────────────┼────────────┼───────────┤")
            for name, vs, vr, tr in rows:
                print(f"│ {name:<15} │ {vs:>11} │ {vr:>10} │ {tr:>9} │")
            print("├─────────────────┼─────────────┴────────────┴───────────┤")
            print(f"│ LR: {current_lr:.2e}    │         VRAM: {peak_vram:.2f} GB              │")
            print("└─────────────────┴──────────────────────────────────────┘")
            
            mlflow.log_metric("train_loss", train_losses['total'], step=epoch)
            mlflow.log_metric("train_disease_loss", train_losses['disease'], step=epoch)
            mlflow.log_metric("val_split_loss", val_split_loss, step=epoch)
            mlflow.log_metric("val_real_loss", val_real_loss, step=epoch)
            mlflow.log_metric("disease_auc_split", auc_split, step=epoch)
            mlflow.log_metric("disease_auc_pr_split", auc_pr_split, step=epoch)
            mlflow.log_metric("device_auc_split", device_auc_split, step=epoch)
            mlflow.log_metric("disease_auc_valid_real", auc_valid_real, step=epoch)
            mlflow.log_metric("disease_auc_pr_valid_real", auc_pr_valid_real, step=epoch)
            mlflow.log_metric("device_auc_valid_real", device_auc_valid_real, step=epoch)
            
            if metrics_test_real:
                mlflow.log_metric("disease_auc_test_real", auc_test_real, step=epoch)
                mlflow.log_metric("disease_auc_pr_test_real", auc_pr_test_real, step=epoch)
                mlflow.log_metric("device_auc_test_real", device_auc_test_real, step=epoch)
            
            mlflow.log_metric("lr", current_lr, step=epoch)
            mlflow.log_metric("vram_peak_gb", peak_vram, step=epoch)
            
            for task in aux_tasks:
                if task not in ['age', 'device'] and task in metrics_split:
                    mlflow.log_metric(f"{task}_auc_split", metrics_split[task]['auc'], step=epoch)
                if task not in ['age', 'device'] and task in metrics_valid_real:
                    mlflow.log_metric(f"{task}_auc_valid_real", metrics_valid_real[task]['auc'], step=epoch)
            
            mlflow.log_text(tee_stdout.get_log(), "console_log.txt")
            
            checkpoint_saved = []
            
            if auc_split > best_auc_split:
                best_auc_split = auc_split
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch,
                    {'auc_split': auc_split, 'auc_valid_real': auc_valid_real, 'auc_test_real': auc_test_real},
                    ckpt_best_split, config_path, aux_tasks
                )
                checkpoint_saved.append(f"{ckpt_best_split.name} (AUC={auc_split:.4f})")
            else:
                patience_counter += 1
            
            if auc_valid_real > best_auc_valid_real:
                best_auc_valid_real = auc_valid_real
                auc_test_at_best_valid = auc_test_real
                save_checkpoint(
                    model, optimizer, epoch,
                    {'auc_split': auc_split, 'auc_valid_real': auc_valid_real, 'auc_test_real': auc_test_real},
                    ckpt_best_valid_real, config_path, aux_tasks
                )
                checkpoint_saved.append(f"{ckpt_best_valid_real.name} (AUC={auc_valid_real:.4f})")
            
            if metrics_test_real and auc_test_real > best_auc_test_real:
                best_auc_test_real = auc_test_real
                # save_checkpoint(
                #     model, optimizer, epoch,
                #     {'auc_split': auc_split, 'auc_valid_real': auc_valid_real, 'auc_test_real': auc_test_real},
                #     ckpt_peak_test_real, config_path, aux_tasks
                # )
                # checkpoint_saved.append(f"{ckpt_peak_test_real.name} (AUC={auc_test_real:.4f})")
            
            if checkpoint_saved:
                print(f"[CKPT] Saved: {', '.join(checkpoint_saved)}")
            
            print(f"[BEST] Split: {best_auc_split:.4f} | Valid Real: {best_auc_valid_real:.4f} | Test@Valid: {auc_test_at_best_valid:.4f} | Peak Test: {best_auc_test_real:.4f}")
            print(f"[INFO] Patience: {patience_counter}/{cfg.training.early_stopping_patience}")
            
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"\n[STOP] Early stopping triggered after {epoch+1} epochs")
                break
        
        mlflow.log_metric("final_best_auc_split", best_auc_split)
        mlflow.log_metric("final_best_auc_valid_real", best_auc_valid_real)
        mlflow.log_metric("final_auc_test_at_best_valid", auc_test_at_best_valid)
        mlflow.log_metric("peak_auc_test_real", best_auc_test_real)
        
        for ckpt_path in [ckpt_best_split, ckpt_best_valid_real]:
            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path))
        
        mlflow.log_text(tee_stdout.get_log(), "console_log.txt")
        mlflow_run_active = False
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best AUC Split:              {best_auc_split:.4f}")
        print(f"Best AUC Valid Real:         {best_auc_valid_real:.4f}")
        print(f"AUC Test @ Best Valid:       {auc_test_at_best_valid:.4f}")
        print(f"Peak AUC Test Real:          {best_auc_test_real:.4f}")
        print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--disease', type=str, required=True, help='Disease name (e.g., "Cardiomegaly")')
    parser.add_argument('--csv-train', type=str, required=True, help='Path to train CSV')
    parser.add_argument('--csv-valid-split', type=str, required=True, help='Path to validation split CSV')
    parser.add_argument('--csv-valid-real', type=str, required=True, help='Path to validation real CSV')
    parser.add_argument('--csv-test-real', type=str, default=None, help='Path to test real CSV (optional)')
    parser.add_argument('--images-root-override', type=str, nargs='+', metavar='OLD=NEW',
                        help='Replace image root paths: --images-root-override /old/train=/new/train_512 /old/valid=/new/valid_512')
    parser.add_argument('--resume', type=str, required=False)
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index (default: 0)')
    args = parser.parse_args()
    
    main(args.config, args.disease, args.csv_train, args.csv_valid_split, args.csv_valid_real, args.csv_test_real, args.resume, args.gpu, args.images_root_override)