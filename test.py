# -*- coding:utf-8 -*-

import os
import time
import torch
import json
import numpy as np

from models.layers import SimpleTARNN, Loss
from utils import checkmate as cm
from utils import data_helper as dh
from utils import param_parser as parser
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("ptlog", "logs/Test-{0}.log".format(time.asctime()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CPT_DIR = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
SAVE_DIR = os.path.abspath(os.path.join(os.path.curdir, "outputs", MODEL))

def test():
    """Test SimpleTARNN model for single text difficulty prediction."""
    logger.info("Loading Data...")
    
    # Load BERT tokenizer if using BERT
    logger.info("Loading BERT tokenizer...")
    if args.bert_mod == 'local':
        tokenizer = dh.load_bert_tokenizer(local_path=args.bert_path)
    else:
        tokenizer = dh.load_bert_tokenizer(model_name=args.bert_name)
    
    vocab_size = tokenizer.vocab_size

    logger.info("Data processing...")
    test_data = dh.load_question_data_single(
        data_file=args.test_file,
        tokenizer=tokenizer,
        include_knowledge=args.include_knowledge,
        include_analysis=args.include_analysis
    )

    logger.info("Creating dataset...")
    test_dataset = dh.QuestionDataset(test_data, device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"Test samples: {len(test_dataset)}")

    # Initialize model
    net = SimpleTARNN(
        args=args,
        vocab_size=vocab_size,
        bert_hidden_size=768 if args.use_bert else None
    ).to(device)

    # Load best checkpoint
    checkpoint_file = cm.get_best_checkpoint(CPT_DIR, select_maximum_value=False)
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    logger.info(f"Loaded checkpoint from: {checkpoint_file}")

    logger.info("Testing...")
    test_loss = 0.0
    true_labels = []
    predicted_scores = []
    predicted_labels = []  # For classification task
    question_ids = []
    subjects = []
    question_types = []
    original_difficulties = []
    
    criterion = Loss()

    batches = trange(len(test_loader), desc="Testing", leave=True)
    with torch.no_grad():
        for batch_cnt, batch in zip(batches, test_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
            
            logits, scores = net(input_ids, attention_mask, token_type_ids)
            
            loss = criterion(logits, labels)
            batch_preds = torch.argmax(scores, dim=1).cpu().numpy()
            predicted_labels.extend(batch_preds)
            
            test_loss += loss.item()
            true_labels.extend(labels.cpu().numpy())
            predicted_scores.extend(batch_preds)
            
            # Collect additional information
            question_ids.extend(batch['question_id'])
            subjects.extend(batch['subject'])
            question_types.extend(batch['question_type'])
            original_difficulties.extend(batch['original_difficulty'])
            
            batches.set_description(f"Testing (Loss={loss.item():.4f})")

    test_loss = test_loss / len(test_loader)
    
    # Calculate metrics based on task type
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='macro', zero_division=0
    )
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=range(args.num_classes))
        
    logger.info("Test Results:")
    logger.info(f"Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted - Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"Macro - Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f} | F1: {f1_macro:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
    # Print per-class metrics
    class_metrics = {}
    for class_idx in range(args.num_classes):
        class_mask = np.array(true_labels) == class_idx
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(
                np.array(true_labels)[class_mask], 
                np.array(predicted_labels)[class_mask]
            )
            class_metrics[class_idx] = {
                'count': np.sum(class_mask),
                'accuracy': class_acc
            }
        
    logger.info("\nPer-class metrics:")
    for class_idx, metrics in class_metrics.items():
        logger.info(f"Class {class_idx}: Count={metrics['count']}, Accuracy={metrics['accuracy']:.4f}")
        
    # Print some examples
    logger.info("\nSample Predictions (first 10):")
    for i in range(min(10, len(true_labels))):
        logger.info(f"ID: {question_ids[i]}, True: {true_labels[i]}, "
                    f"Pred: {predicted_labels[i]}, Correct: {true_labels[i] == predicted_labels[i]}")
    
    logger.info('Test Finished.')

    # Create prediction file
    logger.info('Creating prediction file...')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Save detailed predictions
    predictions_file = os.path.join(SAVE_DIR, 'detailed_predictions.json')
    detailed_results = []
    
    for i in range(len(true_labels)):
        result = {
            'question_id': int(question_ids[i]) if question_ids[i] is not None else None,
            'subject': str(subjects[i]) if subjects[i] is not None else None,
            'question_type': str(question_types[i]) if question_types[i] is not None else None,
            'original_difficulty': str(original_difficulties[i]) if original_difficulties[i] is not None else None,
            'true_label': float(true_labels[i]),
            'predicted_score': int(predicted_labels[i]),
            'text_length': int(test_data['text_lengths'][i])
        }
        detailed_results.append(result)
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Detailed predictions saved to: {predictions_file}")
    
    # Also save CSV format for compatibility
    dh.create_prediction_file(
        save_dir=SAVE_DIR,
        identifiers=question_ids,
        predictions=predicted_scores if args.task_type == 'regression' else predicted_labels,
        task_type=args.task_type
    )
    
    # Save evaluation metrics
    metrics_file = os.path.join(SAVE_DIR, 'evaluation_metrics.json')
    metrics = {
        'loss': float(test_loss),
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation metrics saved to: {metrics_file}")
    logger.info('All Finished.')


if __name__ == "__main__":
    test()