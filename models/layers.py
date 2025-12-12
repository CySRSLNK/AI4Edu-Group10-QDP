# -*- coding:utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BiRNNLayer(nn.Module):
    def __init__(self, input_units, rnn_type, rnn_layers, rnn_hidden_size, dropout_keep_prob):
        super(BiRNNLayer, self).__init__()
        if rnn_layers == 1 and dropout_keep_prob > 0:
            dropout_keep_prob = 0
            print(f"[Warning] Single-layer RNN detected, setting dropout to 0 (was {dropout_keep_prob})")

        if rnn_type == 'LSTM':
            self.bi_rnn = nn.LSTM(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                  batch_first=True, bidirectional=True, dropout=dropout_keep_prob)
        if rnn_type == 'GRU':
            self.bi_rnn = nn.GRU(input_size=input_units, hidden_size=rnn_hidden_size, num_layers=rnn_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout_keep_prob)

    def forward(self, input_x):
        rnn_out, _ = self.bi_rnn(input_x)
        rnn_avg = torch.mean(rnn_out, dim=1)
        return rnn_out, rnn_avg


class AttentionLayer(nn.Module):
    def __init__(self, num_units, att_unit_size, att_type):
        super(AttentionLayer, self).__init__()
        self.att_type = att_type

    def forward(self, input_x, input_y):
        if self.att_type == 'normal':
            attention_matrix = torch.matmul(input_y, input_x.transpose(1, 2))
            scaling_factor = torch.sqrt(torch.tensor(input_x.size(2), dtype=torch.float32, device=input_x.device))
            attention_matrix = attention_matrix / scaling_factor
            attention_weight = torch.softmax(attention_matrix, dim=2)
            attention_visual = torch.mean(attention_matrix, dim=1)
            attention_out = torch.matmul(attention_weight, input_x)
            # TODO
            attention_out = torch.mean(attention_out, dim=1)
        if self.att_type == 'cosine':
            cos_matrix = []
            seq_len = list(input_y.size())[-2]
            normalized_x = F.normalize(input_x, p=2, dim=2)
            for t in range(seq_len):
                new_input_y = torch.unsqueeze(input_y[:, t, :], dim=1)
                normalized_y = F.normalize(new_input_y, p=2, dim=2)
                # cos_similarity: [batch_size, seq_len_1]
                cos_similarity = torch.sum(torch.mul(normalized_y, normalized_x), dim=2)
                cos_matrix.append(cos_similarity)
            # attention_matrix: [batch_size, seq_len_2, seq_len_1]
            attention_matrix = torch.stack(cos_matrix, dim=1)
            attention_visual = torch.mean(attention_matrix, dim=1)
            attention_out = torch.mul(torch.unsqueeze(attention_visual, dim=-1), input_x)
            attention_out = torch.mean(attention_out, dim=1)
        if self.att_type == 'mlp':
            alpha_matrix = []
            seq_len = list(input_y.size())[-2]
            for t in range(seq_len):
                u_t = torch.matmul(torch.unsqueeze(input_y[:, t, :], dim=1), input_x.transpose(1, 2))
                # u_t: [batch_size, 1, seq_len_1]
                u_t = torch.tanh(u_t)
                alpha_matrix.append(u_t)
            attention_matrix = torch.cat(alpha_matrix, dim=1)
            attention_matrix = torch.squeeze(attention_matrix, dim=2)
            attention_weight = F.softmax(attention_matrix, dim=1)
            attention_visual = torch.mean(attention_weight, dim=1)
            attention_out = torch.mul(torch.unsqueeze(attention_visual, dim=-1), input_x)
            attention_out = torch.mean(attention_out, dim=1)
        return attention_visual, attention_out


class HighwayLayer(nn.Module):
    def __init__(self, in_units, out_units):
        super(HighwayLayer, self).__init__()
        self.highway_linear = nn.Linear(in_features=in_units, out_features=out_units, bias=True)
        self.highway_gate = nn.Linear(in_features=in_units, out_features=out_units, bias=True)

    def forward(self, input_x):
        highway_g = torch.relu(self.highway_linear(input_x))
        highway_t = torch.sigmoid(self.highway_gate(input_x))
        highway_out = torch.mul(highway_g, highway_t) + torch.mul((1 - highway_t), input_x)
        return highway_out


class Loss(nn.Module):
    def __init__(self, task_type='regression'):
        super(Loss, self).__init__()
        self.task_type = task_type
        
        if task_type == 'regression':
            self.MSELoss = nn.MSELoss(reduce=True, size_average=True)
        else:
            # 对于分类任务，使用交叉熵损失
            self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, predict_y, input_y):
        # 根据任务类型选择损失函数
        if self.task_type == 'regression':
            # 回归任务：使用MSE损失
            f_loss = self.MSELoss(predict_y[0], input_y[0])
            b_loss = self.MSELoss(predict_y[1], input_y[1])
            losses = f_loss + b_loss
        else:
            # 分类任务：使用交叉熵损失
            f_loss = self.CrossEntropyLoss(predict_y[0], input_y[0].long())
            b_loss = self.CrossEntropyLoss(predict_y[1], input_y[1].long())
            losses = f_loss + b_loss
        
        return losses


class SimpleTARNN(nn.Module):
    """
    简化版的TARNN,用于单文本输入(适用于你的数据格式)
    """
    def __init__(self, args, vocab_size, embedding_size, pretrained_embedding=None, 
                 task_type='regression', num_classes=5, use_bert=False,bert_hidden_size=768):
        super(SimpleTARNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.pretrained_embedding = pretrained_embedding
        self.task_type = task_type
        self.num_classes = num_classes
        self.use_bert = use_bert
        self.bert_hidden_size = bert_hidden_size
        if self.use_bert:
            from transformers import BertModel
            if args.bert_mod == 'local':
                self.bert = BertModel.from_pretrained(args.bert_path)
            else:
                self.bert = BertModel.from_pretrained(args.bert_name)
        else:
            self.bert = None
        self._setup_layers()

    def _setup_embedding_layer(self):
        """
        Creating Embedding layers.
        """
        if self.use_bert:
            # 如果使用BERT，嵌入层将在外部处理
            self.embedding = None
        else:
            if self.pretrained_embedding is None:
                embedding_weight = torch.FloatTensor(np.random.uniform(-1, 1, size=(self.vocab_size, self.embedding_size)))
                embedding_weight = torch.nn.Parameter(embedding_weight, requires_grad=True)
            else:
                if self.args.embedding_type == 0:
                    embedding_weight = torch.from_numpy(self.pretrained_embedding).float()
                if self.args.embedding_type == 1:
                    embedding_weight = torch.nn.Parameter(torch.from_numpy(self.pretrained_embedding).float(), requires_grad=True)
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, _weight=embedding_weight)

    def _setup_bi_rnn_layer(self):
        """
        Creating Bi-RNN Layer.
        """
        if self.use_bert:
            # BERT的输出维度通常是768
            rnn_input_size = self.bert_hidden_size
        else:
            rnn_input_size = self.embedding_size
            
        self.word_bi_rnn = BiRNNLayer(input_units=rnn_input_size, rnn_type=self.args.rnn_type,
                                 rnn_layers=self.args.rnn_layers, rnn_hidden_size=self.args.rnn_dim,
                                 dropout_keep_prob=self.args.dropout_rate)
        
    def _setup_hierarchical_attention(self):
        """
        添加层次化注意力层
        """
        self.word_attention = AttentionLayer(
            num_units=self.args.attention_dim,
            att_unit_size=self.args.attention_dim,
            att_type=self.args.attention_type
        )

    def _setup_highway_layer(self):
        """
         Creating Highway Layer.
         """
        self.highway = HighwayLayer(in_units=self.args.fc_dim, out_units=self.args.fc_dim)

    def _setup_fc_layer(self):
        """
         Creating FC Layer.
         """
        # BiRNN输出是双向的，所以维度是2 * rnn_dim
        self.fc1 = nn.Linear(in_features=self.args.rnn_dim * 2, out_features=self.args.fc_dim, bias=True)
        
        # 根据任务类型设置输出层
        if self.task_type == 'regression':
            self.out = nn.Linear(in_features=self.args.fc_dim, out_features=1, bias=True)
        else:  # classification
            self.out = nn.Linear(in_features=self.args.fc_dim, out_features=self.num_classes, bias=True)

    def _setup_dropout(self):
        """
         Adding Dropout.
         """
        self.dropout = nn.Dropout(self.args.dropout_rate)

    def _setup_layers(self):
        """
        Creating layers of model.
        """
        self._setup_embedding_layer()
        self._setup_bi_rnn_layer()
        self._setup_hierarchical_attention()
        self._setup_highway_layer()
        self._setup_fc_layer()
        self._setup_dropout()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, sentence_boundaries=None):
        """
        前向传播，适用于单文本输入
        """
        if self.use_bert:
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            embedded_sentence = bert_outputs.last_hidden_state
        else:
            embedded_sentence = self.embedding(input_ids)
        
        word_rnn_out, word_rnn_avg = self.word_bi_rnn(embedded_sentence)
        word_attention_visual, word_attention_out = self.word_attention(word_rnn_out, word_rnn_out)
        combined = word_attention_out

        # Fully Connected Layer
        fc_out = self.fc1(combined)

        # Highway Layer
        highway_out = self.highway(fc_out)

        # Dropout
        h_drop = self.dropout(highway_out)

        # 输出层
        if self.task_type == 'regression':
            logits = self.out(h_drop).squeeze()
            scores =  logits
        else:  # classification
            logits = self.out(h_drop)
            scores = F.softmax(logits, dim=1)  # 对于分类任务，使用softmax

        return logits, scores