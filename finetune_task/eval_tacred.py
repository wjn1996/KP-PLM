# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import simplejson as json
from pathlib import Path

import math
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# from knowledge_bert.typing import BertTokenizer as BertTokenizer_label
# from knowledge_bert.tokenization import BertTokenizer
# from knowledge_bert.modeling import BertForEntityTyping
# from knowledge_bert.optimization import BertAdam
# from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from transformers.models.roberta import RobertaTokenizer, RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta import RobertaForSequenceClassification

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)




def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        # self.input_ent = input_ent
        # self.ent_mask = ent_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())


class TacredProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "dev")

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
                    # print(line['text'][x[1]:x[2]].encode("utf-8"))
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=line['ann'], label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""

    label_list = sorted(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}

    # entity2id = {}
    # with open("kg_embed/entity2id.txt") as fin:
    #     fin.readline()
    #     for line in fin:
    #         qid, eid = line.strip().split('\t')
    #         entity2id[qid] = int(eid)

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        t_name = ex_text_a[t[1]:t[2]]
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + "# "+h_name+" #" + \
                ex_text_a[h[2]:t[1]] + "$ "+t_name+" $" + ex_text_a[t[2]:]
        else:
            ex_text_a = ex_text_a[:t[1]] + "$ "+t_name+" $" + \
                ex_text_a[t[2]:h[1]] + "# "+h_name+" #" + ex_text_a[h[2]:]

        ent_pos = [x for x in example.text_b if x[-1] > threshold]
        for x in ent_pos:
            cnt = 0
            if x[1] > h[2]:
                cnt += 2
            if x[1] >= h[1]:
                cnt += 2
            if x[1] >= t[1]:
                cnt += 2
            if x[1] > t[2]:
                cnt += 2
            x[1] += cnt
            x[2] += cnt
        # tokens_a, entities_a = tokenizer.tokenize(ex_text_a, ent_pos)
        tokens_a = tokenizer.tokenize(ex_text_a)

        tokens_b = None
        if False:
            tokens_b, entities_b = tokenizer.tokenize(
                example.text_b[0], [x for x in example.text_b[1] if x[-1] > threshold])
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, entities_a,
                               entities_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                # entities_a = entities_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # ents = ["UNK"] + entities_a + ["UNK"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            # ents += entities_b + ["UNK"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # input_ent = []
        # ent_mask = []
        # for ent in ents:
        #     if ent != "UNK" and ent in entity2id:
        #         input_ent.append(entity2id[ent])
        #         ent_mask.append(1)
        #     else:
        #         input_ent.append(-1)
        #         ent_mask.append(0)
        # ent_mask[0] = 1

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        # padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        # input_ent += padding_
        # ent_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(input_ent) == max_seq_length
        # assert len(ent_mask) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            # logger.info("ents: %s" % " ".join(
            #     [str(x) for x in ents]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          # input_ent=input_ent,
                          # ent_mask=ent_mask,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def eval_result(pred_result, labels, na_id):
    correct = 0
    total = len(labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    for i in range(total):
        if labels[i] == pred_result[i]:
            correct += 1
            if labels[i] != na_id:
                correct_positive += 1
        if labels[i] != na_id:
            gold_positive += 1
        if pred_result[i] != na_id:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    result = {'acc': acc, 'micro_p': micro_p,
              'micro_r': micro_r, 'micro_f1': micro_f1}
    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_bin",
                        default=None,
                        type=int,
                        nargs="+",
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--threshold', type=float, default=.3)

    args = parser.parse_args()

    processors = TacredProcessor

    num_labels_task = 42

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    processor = processors()
    num_labels = num_labels_task
    label_list = None

    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    label_list = sorted(label_list)
    # vecs = []
    # vecs.append([0]*100)
    # with open("kg_embed/entity2vec.vec", 'r') as fin:
    #     for line in fin:
    #         vec = line.strip().split('\t')
    #         vec = [float(x) for x in vec]
    #         if len(vec) != 100:
    #             diff = 100 - len(vec)
    #             vec = vec + [0 for _ in range(diff)]
    #         vecs.append(vec)
    # embed = torch.tensor(vecs, dtype=torch.float)
    # embed = torch.nn.Embedding.from_pretrained(embed)
    #
    # logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    # del vecs

    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if "pytorch_model.bin_" in x]

    if args.model_bin is None:
        file_mark = []
        for x in filenames:
            file_mark.append([x, True])
            file_mark.append([x, False])
    else:
        file_mark = [['pytorch_model.bin_{}'.format(i), True] for i in args.model_bin]

    eval_examples = processor.get_dev_examples(args.data_dir)
    dev = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.threshold)
    eval_examples = processor.get_test_examples(args.data_dir)
    test = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.threshold)

    for x, mark in file_mark:
        print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, state_dict=model_state_dict, num_labels=len(label_list))
        model.to(device)

        if mark:
            eval_features = dev
            output_file = os.path.join(
                args.output_dir, "eval_pred_{}.txt".format(x.split("_")[-1]))
            output_file_ = os.path.join(
                args.output_dir, "eval_gold_{}.txt".format(x.split("_")[-1]))
        else:
            eval_features = test
            output_file = os.path.join(
                args.output_dir, "test_pred_{}.txt".format(x.split("_")[-1]))
            output_file_ = os.path.join(
                args.output_dir, "test_gold_{}.txt".format(x.split("_")[-1]))
        fpred = open(output_file, "w")
        fgold = open(output_file_, "w")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # zeros = [0 for _ in range(args.max_seq_length)]
        # zeros_ent = [0 for _ in range(100)]
        # zeros_ent = [zeros_ent for _ in range(args.max_seq_length)]
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long)
        # all_ent = torch.tensor(
        #     [f.input_ent for f in eval_features], dtype=torch.long)
        # all_ent_masks = torch.tensor(
        #     [f.ent_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred_all, label_all = [], []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
            # input_ent = embed(input_ent+1)  # -1 -> 0
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            # input_ent = input_ent.to(device)
            # ent_mask = ent_mask.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                # tmp_eval_loss = model(
                #     input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids)
                # logits = model(input_ids, segment_ids,
                #                input_mask, input_ent, ent_mask)
                output = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=label_ids,
                    return_dict=False
                )
                tmp_eval_loss, logits = output[0], output[1]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            pred = np.argmax(logits, axis=1)
            for a, b in zip(pred, label_ids):
                pred_all.append(a)
                label_all.append(b)
                fgold.write("{}\n".format(label_list[b]))
                fpred.write("{}\n".format(label_list[a]))

            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        result = eval_result(pred_all, label_all, label_list.index("NA"))

        if mark:
            output_eval_file = os.path.join(
                args.output_dir, "eval_results_{}.txt".format(x.split("_")[-1]))
        else:
            output_eval_file = os.path.join(
                args.output_dir, "test_results_{}.txt".format(x.split("_")[-1]))

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
'''
CUDA_VISIBLE_DEVICES=6,7  python3 eval_tacred.py --do_eval --do_lower_case --data_dir data/tacred --model_name_or_path ../output/roberta-base-temp/checkpoint-32000/ --max_seq_length 256 --train_batch_size 32 --learning_rate 2e-5  --num_train_epochs 4.0 --output_dir output_tacred_32000 --fp16 --loss_scale 128 --threshold 0.4
CUDA_VISIBLE_DEVICES=6,7  python3 eval_tacred.py --do_eval --do_lower_case --data_dir data/tacred --model_name_or_path roberta-base --max_seq_length 256 --train_batch_size 32 --learning_rate 2e-5  --num_train_epochs 4.0 --output_dir output_tacred_roberta --fp16 --loss_scale 128 --threshold 0.4
'''