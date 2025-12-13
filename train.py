# -*- coding:utf-8 -*-

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

from models.layers import SimpleTARNN, Loss
from utils import checkmate as cm
from utils import data_helper as dh
from utils import param_parser as parser
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix


args = parser.parameter_parser()
OPTION = dh.option()
logger = dh.logger_fn("ptlog", "logs/{0}-{1}.log".format('Train' if OPTION == 'T' else 'Restore', time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using the device is {device}")

def train():
    """Training SimpleTARNN model for single text difficulty prediction."""
    dh.tab_printer(args, logger)
    
    # Load BERT tokenizer if using BERT
    logger.info("Loading BERT tokenizer...")
    if args.bert_mod == 'local':
        tokenizer = dh.load_bert_tokenizer(local_path=args.bert_path)
    else:
        tokenizer = dh.load_bert_tokenizer(model_name=args.bert_name)
    vocab_size = tokenizer.vocab_size
    embedding_size = args.max_length
    pretrained_embedding = None

    # Load data
    logger.info("Loading training data...")
    train_data = dh.load_question_data_single(
        data_file=args.train_file,
        tokenizer=tokenizer,
        task_type=args.task_type,
        max_length=args.max_length,
        include_knowledge=args.include_knowledge,
        include_analysis=args.include_analysis
    )
    
    logger.info("Loading validation data...")
    val_data = dh.load_question_data_single(
        data_file=args.validation_file,
        tokenizer=tokenizer,
        task_type=args.task_type,
        max_length=args.max_length,
        include_knowledge=args.include_knowledge,
        include_analysis=args.include_analysis
    )

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = dh.QuestionDataset(train_data, device, args.task_type)
    val_dataset = dh.QuestionDataset(val_data, device, args.task_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Init network
    logger.info("Initializing model...")
    net = SimpleTARNN(
        args=args,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        pretrained_embedding=pretrained_embedding,
        task_type=args.task_type,
        num_classes=args.num_classes,
        use_bert=args.use_bert,
        bert_hidden_size=768 if args.use_bert else None
    ).to(device)

    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(f"{param_tensor}\t{net.state_dict()[param_tensor].size()}")

    criterion = Loss(task_type=args.task_type)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)

    if OPTION == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info(f"Writing to {out_dir}\n")
    elif OPTION == 'R':
        timestamp = input("[Input] Please input the checkpoints model you want to restore: ")
        while not (timestamp.isdigit() and len(timestamp) == 10):
            timestamp = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        saver = cm.BestCheckpointSaver(save_dir=out_dir, num_to_keep=args.num_checkpoints, maximize=False)
        logger.info(f"Writing to {out_dir}\n")
        checkpoint = torch.load(os.path.join(out_dir, "best_checkpoints"))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info("Training...")
    writer = SummaryWriter(os.path.join(out_dir, 'summary'))

    def eval_model(val_loader, epoch):
        """Evaluate on the validation set."""
        net.eval()
        eval_loss = 0.0
        true_labels = []
        predicted_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                labels = batch['labels']
                
                logits, scores = net(input_ids, attention_mask, token_type_ids)
                
                if args.task_type == 'regression':
                    loss = nn.MSELoss()(scores.squeeze(), labels)
                    true_labels.extend(labels.cpu().numpy())
                    predicted_scores.extend(scores.squeeze().cpu().numpy())
                else:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    true_labels.extend(labels.cpu().numpy())
                    predicted_scores.extend(torch.argmax(scores, dim=1).cpu().numpy())
                
                eval_loss += loss.item()

        eval_loss = eval_loss / len(val_loader)
        
        # Calculate metrics based on task type
        if args.task_type == 'regression':
            # Regression metrics
            eval_rmse = mean_squared_error(true_labels, predicted_scores) ** 0.5
            eval_r2 = r2_score(true_labels, predicted_scores)
            eval_pcc, eval_doa = dh.regression_eval(true_labels, predicted_scores)
            
            logger.info(f"Validation - Loss: {eval_loss:.4f} | PCC: {eval_pcc:.4f} | DOA: {eval_doa:.4f} | "
                       f"RMSE: {eval_rmse:.4f} | R2: {eval_r2:.4f}")
            
            writer.add_scalar('validation/loss', eval_loss, epoch)
            writer.add_scalar('validation/PCC', eval_pcc, epoch)
            writer.add_scalar('validation/DOA', eval_doa, epoch)
            writer.add_scalar('validation/RMSE', eval_rmse, epoch)
            writer.add_scalar('validation/R2', eval_r2, epoch)
            
            cur_value = eval_rmse  # Use RMSE for checkpoint saving (lower is better)
            
        else:
            # Classification metrics
            accuracy = accuracy_score(true_labels, predicted_scores)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predicted_scores, average='weighted', zero_division=0
            )
            conf_matrix = confusion_matrix(true_labels, predicted_scores, labels=range(args.num_classes))
            
            logger.info(f"Validation - Loss: {eval_loss:.4f} | Accuracy: {accuracy:.4f} | "
                       f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            writer.add_scalar('validation/loss', eval_loss, epoch)
            writer.add_scalar('validation/accuracy', accuracy, epoch)
            writer.add_scalar('validation/precision', precision, epoch)
            writer.add_scalar('validation/recall', recall, epoch)
            writer.add_scalar('validation/f1', f1, epoch)
            
            cur_value = accuracy  # Use accuracy for checkpoint saving (higher is better)
        
        return cur_value

    # Training loop
    for epoch in tqdm(range(args.epochs), desc="Epochs", leave=True):
        net.train()
        epoch_loss = 0.0
        
        batches = trange(len(train_loader), desc=f"Epoch {epoch+1}", leave=True)
        for batch_cnt, batch in zip(batches, train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
            
            logits, scores = net(input_ids, attention_mask, token_type_ids)
            
            if args.task_type == 'regression':
                loss = nn.MSELoss()(scores.squeeze(), labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches.set_description(f"Epoch {epoch+1} (Loss={loss.item():.4f})")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('training/loss', avg_epoch_loss, epoch)
        
        # Evaluation
        cur_value = eval_model(val_loader, epoch)
        saver.handle(cur_value, net, optimizer, epoch)
    
    writer.close()
    logger.info('Training Finished.')
    
    # Save final model
    final_model_path = os.path.join(out_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }, final_model_path)
    logger.info(f'Final model saved to {final_model_path}')


if __name__ == "__main__":
    train()