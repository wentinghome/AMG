'''
Goal: get the embedding from the pretrained unilm-mlm-epoch4 model.
input: the model, the input ids, and the input mask.
output: the sentence level representation. [bs, 768]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
from pathlib import Path
import sys
sys.path.append('../')
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling_mlm import BertForSeq2SeqPreviousContext
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from nn.data_parallel import DataParallelImbalance
import biunilm.seq2seq_loader_mlm as seq2seq_loader

import torch
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def get_prev_entity_hidden(model, input_ids, input_mask):
def get_prev_entity_hidden(model,tokenizer, input_ids, input_mask,
                           max_a_len=30):
    '''
    input is a list consisting the ids of several entities, not padded.
    each entity is like: ['[ENTITY_CLS]' ... '[ENTITY_SEP]']
    '''


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")

    # decoding parameter
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--max_seq_length", default=367, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', default=True,
                        # action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for decoding.")

    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    # parser.add_argument('--forbid_duplicate_ngrams', default=True)
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Ignore the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=64,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--not_predict_token', type=str, default=None,
                        help="Do not predict the tokens during decoding.")

    args = parser.parse_args()


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # tokenizer = BertTokenizer.from_pretrained(
    #     args.bert_model, do_lower_case=args.do_lower_case)

    # tokenizer = BertTokenizer.from_pretrained(
    #     args.bert_model, do_lower_case=args.do_lower_case,
    #     never_split=("[UNK]", "[SEP]", "[X_SEP]", "[PAD]", "[CLS]", "[MASK]",
    #                  "[ENTITY_CLS]", "[ENTITY_SEP]"))

    tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(
        seq2seq_loader.Preprocess4Seq2seqDecoder(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids,
                                                 args.max_seq_length,
                                                 max_tgt_length=args.max_tgt_length,
                                                 new_segment_ids=args.new_segment_ids,
                                                 mode="s2s", num_qkv=args.num_qkv,
                                                 s2s_special_token=args.s2s_special_token,
                                                 s2s_add_segment=args.s2s_add_segment,
                                                 s2s_share_segment=args.s2s_share_segment,
                                                 pos_shift=args.pos_shift))

    ##############################Start the function main body###############################
    # _chunk = input_lines[next_i:next_i + args.batch_size]
    # _chunk_str = input_str_lines[next_i:next_i + args.batch_size]
    # buf_id = [x[0] for x in _chunk]
    # buf = [x[1] for x in _chunk]
    # next_i += args.batch_size
    # max_a_len = max([len(x) for x in buf])
    instances = []
    for instance in [(x, max_a_len) for x in input_ids]:
        for proc in bi_uni_pipeline:
            instances.append(proc(instance))
    with torch.no_grad():
        batch = seq2seq_loader.batch_list_to_batch_tensors(
            instances)
        batch = [t.to(device) if t is not None else None for t in batch]
        input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
        batch_avg_hidden = model(input_ids, token_type_ids,
                                 position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)

    return batch_avg_hidden

if __name__ == "__main__":

    # 2. load the unilm-epoch-4 model to prepare for the
    PATH = "/data0/CKPT/unilm-mlm-get-prev-para-embed-epoch4-model/" \
           "unilm-mlm-epoch-4-entire-model.pt"
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased', do_lower_case=False,
    never_split=("[UNK]", "[SEP]", "[X_SEP]", "[PAD]", "[CLS]", "[MASK]", "[ENTITY_CLS]", "[ENTITY_SEP]"))

    # Load
    prev_unilm_mlm_e4_model = torch.load(PATH)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    prev_unilm_mlm_e4_model.to(device)
    prev_unilm_mlm_e4_model.eval()

    data = [[20, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,19,0,0,0],
            [0]*20,
            [20, 1, 2, 3,4,5,6,7,8,19,0,0,0,0,0,0,0,0,0,0]]
    input_ids = torch.tensor(data)

    input_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1, 1,0,0,0],
                  [0]*20,
                 [1, 1, 1, 1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]]
    input_mask = torch.tensor(input_mask)
    get_prev_entity_hidden(prev_unilm_mlm_e4_model,tokenizer, input_ids, input_mask)
    print("~")