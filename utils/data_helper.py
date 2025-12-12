#-*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import math
import logging
import json
import torch
import numpy as np
import pandas as pd

from scipy import stats
from texttable import Texttable
from transformers import BertTokenizer, BertModel


def option():
    """
    Choose training or restore pattern.

    Returns:
        The OPTION
    """
    OPTION = input("[Input] Train or Restore? (T/R): ")
    while not (OPTION.upper() in ['T', 'R']):
        OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger
        input_file: The logger file path
        level: The logger level
    Returns:
        The logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(save_dir, identifiers, predictions, task_type):
    """
    Create the prediction file.

    Args:
        save_dir: The all classes predicted results provided by network
        identifiers: The data record id
        predictions: The predict scores
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    preds_file = os.path.abspath(os.path.join(save_dir, 'predictions.csv'))
    out = pd.DataFrame()
    out["id"] = identifiers
    out["predictions"] = [round(float(i), 4) for i in predictions]
    out.to_csv(preds_file, index=None)


def load_bert_tokenizer(model_name='bert-base-chinese'):
    """
    加载BERT分词器

    Args:
        model_name: BERT模型名字
    
    Returns:
        BERT分词器实例
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer

def regression_eval(true_label, pred_label):
    """
    Calculate the PCC & DOA.

    Args:
        true_label: The true labels
        pred_label: The predicted labels
    Returns:
        The value of PCC & DOA
    """
    # compute pcc
    pcc, _ = stats.pearsonr(pred_label, true_label)
    if math.isnan(pcc):
        print('[Error]: PCC=nan', true_label, pred_label)
    # compute doa
    n = 0
    correct_num = 0
    for i in range(len(true_label) - 1):
        for j in range(i + 1, len(true_label)):
            if (true_label[i] > true_label[j]) and (pred_label[i] > pred_label[j]):
                correct_num += 1
            elif (true_label[i] == true_label[j]) and (pred_label[i] == pred_label[j]):
                continue
            elif (true_label[i] < true_label[j]) and (pred_label[i] < pred_label[j]):
                correct_num += 1
            n += 1
    if n == 0:
        print(true_label)
        return -1, -1
    doa = correct_num / n
    return pcc, doa


def get_diff_map():
    """
    获得难度到数值的映射
    
    Returns:
        Dict[str, int]
    """
    difficult_mapping = {
        '极难': 4,
        '困难': 3,
        '较难': 3,  # 同义词
        '难': 3,    # 同义词
        '中等': 2,
        '一般': 1,
        '较易': 0,
        '容易': 0,  # 同义词
        '简单': 0,  # 同义词
        # 扩展其他可能写法
        '非常难': 4,
        '非常容易': 0,
        '极容易': 0
    }
    return difficult_mapping


def conv_diff_to_value(difficulty_str, task_type, difficulty_map):
    """
    将难度字符串转换为数值
 
    Args:
        difficulty_str: 难度字符串
        task_type: 任务类型 ('classification' 或 'regression')
        difficulty_map: 自定义难度映射
    
    Returns:
        Union[int, float]: 数值标签
    """
    if difficulty_map is None:
        difficulty_map = get_diff_map()
    
    default_value = 2 if task_type == 'classification' else 2.0
    difficulty_str = difficulty_str.strip()
    
    if difficulty_str in difficulty_map:
        value = difficulty_map[difficulty_str]
    else:
        matched = False
        for key, val in difficulty_map.items():
            if key in difficulty_str:
                value = val
                matched = True
                break
        
        if not matched:
            value = default_value
    
    return int(value) if task_type == 'classification' else float(value)


def prepare_text(question_data, include_knowledge=True, include_analysis=False):
    """
    准备模型输入的文本
    
    Args:
        question_data: 题目数据字典
        include_knowledge: 是否包含知识点
        include_analysis: 是否包含解析（注意：训练时不应包含解析）
    
    Returns:
        str: 处理后的文本
    """
    text = question_data.get('ques_content', '')
    if include_knowledge and 'ques_knowledges' in question_data and question_data['ques_knowledges']:
        knowledge_text = ' '.join(question_data['ques_knowledges'])
        text = text + " [知识点] " + knowledge_text
    
    if include_analysis and 'ques_analyze' in question_data:
        analysis_text = question_data['ques_analyze']
        text = text + " [解析] " + analysis_text
    
    return text


def load_question_data_single(data_file, tokenizer, task_type, max_length = 256,
                             include_knowledge = True, include_analysis = False, difficulty_map = None):
    """
    加载单文本题目数据（适应你的数据格式）
    
    Args:
        data_file: 数据文件路径
        tokenizer: 分词器(BERT或自定义)
        task_type: 任务类型 ('classification' 或 'regression')
        max_length: 最大序列长度
        include_knowledge: 是否包含知识点
        include_analysis: 是否包含解析
        difficulty_map: 难度映射
    
    Returns:
        Dict[str, List]: 处理后的数据
    """
    processed_data = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'labels': [],
        'question_ids': [],
        'subjects': [],
        'question_types': [],
        'original_difficulties': [],
        'text_lengths': []
    }
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                question_data = json.loads(line.strip())
                text = prepare_text(question_data, include_knowledge, include_analysis)

                if hasattr(tokenizer, 'encode_plus'):
                    encoded = tokenizer.encode_plus(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=max_length,
                        return_tensors='pt'
                    )
                    
                    processed_data['input_ids'].append(encoded['input_ids'].squeeze(0))
                    processed_data['attention_mask'].append(encoded['attention_mask'].squeeze(0))
                    
                    if 'token_type_ids' in encoded:
                        processed_data['token_type_ids'].append(encoded['token_type_ids'].squeeze(0))
                    else:
                        processed_data['token_type_ids'].append(torch.zeros(max_length, dtype=torch.long))
                
                else:
                    raise NotImplementedError("Currently only BERT tokenizer is supported for single text mode")

                difficulty_str = question_data.get('ques_difficulty', '一般')
                label_value = conv_diff_to_value(difficulty_str, task_type, difficulty_map)
                
                processed_data['labels'].append(label_value)
                processed_data['original_difficulties'].append(difficulty_str)
                processed_data['question_ids'].append(line_num)
                processed_data['subjects'].append(question_data.get('subject', ''))
                processed_data['question_types'].append(question_data.get('ques_type', ''))
                
                text_length = min(len(text), max_length)
                processed_data['text_lengths'].append(text_length)
                
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON解析错误(第{line_num}行): {e}")
                continue
            except Exception as e:
                print(f"[Warning] 处理第{line_num}行时出错: {e}")
                continue
    
    return processed_data


def classification_eval(true_labels, pred_labes, num_classes=5):
    """
    分类评估任务(难度等级0-4)
    Args:
        true_labels: 真实标签(0-4)
        pred_labels: 预测标签(0-4)
        num_classes: 类别数目
    Returns:
        accuracy, precision, recall, f1, confusion_matrix
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    acc = accuracy_score(true_labels, pred_labes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labes, average='weighted', zero_division=0
    )
    conf_matrix = confusion_matrix(true_labels, pred_labes, labels=range(num_classes))
    return acc, precision, recall, f1, conf_matrix


class QuestionDataset(torch.utils.data.Dataset):
    """
    单文本题目数据集类（适应你的数据格式）
    """
    def __init__(self, data, device, task_type):
        """
        初始化数据集
        
        Args:
            data: 处理后的数据字典
            device: 设备 ('cpu' 或 'cuda')
            task_type: 任务类型 ('classification' 或 'regression')
        """
        self.device = device
        self.task_type = task_type
        
        self.input_ids = torch.stack(data['input_ids'])
        self.attention_mask = torch.stack(data['attention_mask'])
        self.token_type_ids = torch.stack(data['token_type_ids'])
        
        if task_type == 'regression':
            self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        else:  # classification
            self.labels = torch.tensor(data['labels'], dtype=torch.long)
        
        self.question_ids = data['question_ids']
        self.subjects = data['subjects']
        self.question_types = data['question_types']
        self.original_difficulties = data['original_difficulties']
        self.text_lengths = data['text_lengths']
        
        print(f"数据集已创建: {len(self.labels)} 个样本")
        print(f"任务类型: {task_type}")
        print(f"标签类型: {self.labels.dtype}")
 
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        """获取单个样本"""
        sample = {
            'input_ids': self.input_ids[idx].to(self.device),
            'attention_mask': self.attention_mask[idx].to(self.device),
            'token_type_ids': self.token_type_ids[idx].to(self.device),
            'labels': self.labels[idx].to(self.device),
            'question_id': self.question_ids[idx],
            'subject': self.subjects[idx],
            'question_type': self.question_types[idx],
            'original_difficulty': self.original_difficulties[idx]
        }
        
        return sample