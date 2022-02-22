from random import randint, shuffle, choice
from random import random as rand
import math
import torch
import os
from biunilm.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
import pickle
import numpy as np
from itertools import groupby
import csv
# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

def truncate_tokens_pair(tokens_a, tokens_b, tokens_c, tokens_d, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    # (tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
    #                                                   max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
    num_truncated_a = [0, 0]
    # b,c,d : tgt, mem_tgt_span_label,mem_tgt_span_label
    num_truncated_b = [0, 0]
    num_truncated_c = [0, 0]
    num_truncated_d = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = (tokens_b, tokens_c, tokens_d)
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = (tokens_b, tokens_c, tokens_d)
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = (tokens_b, tokens_c, tokens_d)
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            if isinstance(trunc_tokens, tuple):
                del trunc_tokens[0][0] # truncate token_b
                del trunc_tokens[1][0] # truncate token_c
                del trunc_tokens[2][0] # truncate token_d
            else:
                del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            if isinstance(trunc_tokens, tuple):
                trunc_tokens[0].pop() # pop token_b
                trunc_tokens[1].pop() # pop token_c
                trunc_tokens[2].pop() # pop token_d
            else:
                trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b, num_truncated_c, num_truncated_d

# def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
#     num_truncated_a = [0, 0]
#     num_truncated_b = [0, 0]
#     while True:
#         if len(tokens_a) + len(tokens_b) <= max_len:
#             break
#         if (max_len_a > 0) and len(tokens_a) > max_len_a:
#             trunc_tokens = tokens_a
#             num_truncated = num_truncated_a
#         elif (max_len_b > 0) and len(tokens_b) > max_len_b:
#             trunc_tokens = tokens_b
#             num_truncated = num_truncated_b
#         elif trunc_seg:
#             # truncate the specified segment
#             if trunc_seg == 'a':
#                 trunc_tokens = tokens_a
#                 num_truncated = num_truncated_a
#             else:
#                 trunc_tokens = tokens_b
#                 num_truncated = num_truncated_b
#         else:
#             # truncate the longer segment
#             if len(tokens_a) > len(tokens_b):
#                 trunc_tokens = tokens_a
#                 num_truncated = num_truncated_a
#             else:
#                 trunc_tokens = tokens_b
#                 num_truncated = num_truncated_b
#         # whether always truncate source sequences
#         if (not always_truncate_tail) and (rand() < 0.5):
#             del trunc_tokens[0]
#             num_truncated[0] += 1
#         else:
#             trunc_tokens.pop()
#             num_truncated[1] += 1
#     return num_truncated_a, num_truncated_b


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file_src, file_entity_emb_dir, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1,
                 sent_reverse_order=False, bi_uni_pipeline=[],
                 max_len_key_entity=20, max_len_key_entity_tk = 30, memsize=15, use_kwmem=True, dim=768,
                 max_len_a=300,
                 split='train',
                 data_avaliable = -1):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []
        self.max_len_key_entity = max_len_key_entity # max num of entities
        self.max_len_key_entity_tk = max_len_key_entity_tk # max num tokens of each entity
        self.memsize = memsize
        self.use_kwmem = use_kwmem
        self.dim = dim
        # self.max_len_a = max_len_a  # the max key word length
        self.file_entity_emb_dir = file_entity_emb_dir
        self.h = max_len_key_entity # max num of previous entity
        self.split = split
        self.data_avaliable = data_avaliable
        self.num_summary_tks = 0
        src_tokenized_list = []
        tgt_tokenized_list = []
        with open(file_src, "r", encoding='utf-8') as f_src:
            # 1) get the input_data file; and the previous paragraph embedding file
            with open(file_src, "rb") as f:
                self.data = f.readlines()

        if self.data_avaliable != -1 and self.split=='train':
            # if need a small portion of the data
            if self.data_avaliable > len(self.data):
                print("exceed the max len of data")
            else:
                data_index = np.random.permutation(len(self.data))[:self.data_avaliable]
                # print([i for i in data_index])
                self.data = [self.data[i] for i in data_index]

        # calculate the number of summary tokens.

        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.data)

    def prepare_key_entity(self, key_words):
        tk_id = []
        mask = []
        entity_end_tk = '[ENTITY_SEP]'
        for entity in key_words:
            # 1) tokenize
            entity_tk = self.tokenizer.tokenize(entity.strip())
            if len(entity_tk) > self.max_len_key_entity_tk:
                entity_tk = entity_tk[:self.max_len_key_entity_tk - 1] + [entity_end_tk]

            entity_tk_ids = self.tokenizer.convert_tokens_to_ids(entity_tk)
            entity_mask = [1] * len(entity_tk)
            # 2) pad
            # if len(entity_tk) < self.max_len_key_entity_tk:
            n_pad = self.max_len_key_entity_tk - len(entity_tk)
            entity_tk_ids.extend([0] * n_pad)
            entity_mask.extend([0] * n_pad)
            # 3) add to the example res list
            tk_id.append(entity_tk_ids)
            mask.append(entity_mask)
        num_pad_entity = self.max_len_key_entity - len(key_words)
        pad_entity = [[0]*self.max_len_key_entity_tk]
        tk_id.extend(pad_entity*num_pad_entity)
        mask.extend(pad_entity*num_pad_entity)
        return tk_id,mask

    def __getitem__(self, idx):
        # A). Read data
        # 1.1 ) get csv_data : [doc_id], [keywords], [table_str], [memory_id]
        csv_data = self.data[idx].decode("utf-8", "ignore").strip().split('\t')
        docid = int(csv_data[0])  # doc id starting from 0
        key_words = csv_data[1].split('[SEP]')[:self.max_len_key_entity] # only got so many keywords entity
        len_effective_kw = len(key_words)
        # 1.2) prepare the keyword (tokenize and pad) for the entity level keyword computation.
        key_entity_tk, key_entity_mask = self.prepare_key_entity(key_words)

        table_data = csv_data[2]
        summary_data = csv_data[3]
        memory_id_label = csv_data[4]
        assert len(summary_data.split()) == len(memory_id_label.split())
        # 2.1) tokenize the table data
        # 2.2) map summary data and memory_id_label on the token level
        table_tk, table_words, table_token_list_of_list = self.tokenizer.tokenize_return_word_token_list(table_data.strip())
        summary_tk, summary_words, summary_token_list_of_list = self.tokenizer.tokenize_return_word_token_list(summary_data.strip())
        # self.num_summary_tks += len(summary_tk)

        _, memory_id_words, _ = self.tokenizer.tokenize_return_word_token_list(memory_id_label.strip())
        assert len(summary_words) == len(memory_id_words)
        # 2.3) get the token level memory id
        memory_id_tk = self.mem_tgt_span_assign(summary_tk, summary_token_list_of_list, memory_id_words)
        # 2.4) new added: get the table input span mask
        span_id_tk = self.span_tgt_span_assign(summary_tk)

        # B). Unilm info
        # 3) get the unilm related data information.
        proc = choice(self.bi_uni_pipeline)
        instance = proc((table_tk, summary_tk, memory_id_tk, span_id_tk, self.h)) # this includes the memory_id for [src, tgt]
        # C). Memory info
        # 4) get the memory related information
        # 4.1) get the mem and mmask initilization part
        if self.use_kwmem:
            # use keyword memory(entity level), initialize a empty mem [num_updates, 20+10, 768]
            # num_updates means : store all the possible memory for an example in an matrix
            mem = torch.torch.empty(self.h, self.max_len_key_entity + self.memsize, self.dim).normal_(std=.02)
            mmask = torch.zeros(self.max_len_key_entity + self.memsize).long()
            # the keyword mem mask is set according to the effective length
            mmask[0:len_effective_kw] = torch.ones(len_effective_kw).long()
        else:
            mem = torch.torch.empty(self.h, self.memsize, self.dim).normal_(std=.02)
            mmask = torch.zeros(self.memsize).long()
        mmask[-self.memsize:] = torch.ones(self.memsize).long()

        # 4.2) get the prev and pmask res
        # if self.split =='train': train_500_split = 'train_500'
        prev_embed_pkl_path = os.path.join(self.file_entity_emb_dir,
                                           self.split + '_' + str(docid) + '.pkl')
        with open(prev_embed_pkl_path, 'rb') as doc_history_f:
            history = pickle.load(doc_history_f)

        prev = torch.zeros(self.h, 1, self.dim).float()  # .long() [10,1,768]
        pmask = torch.zeros(self.h, 1).long()  # [10, 1] - for the entire example, how many previous ones are filled.

        num_history = len(history)  # pid now is the total length of the effective keywords
        for p in range(1, min(num_history, self.h)):  # at most, the model accept h=10 paragraph
            prev[p, 0, :] = history[p][-1]  # the last item in the current pid tuple
            pmask[p, 0] = torch.ones(1).long()  # the last item's pmask

        # # 4) get the current doc, pid previous paragraph embedding
        # assert pid == history[pid][1]  # history[p][1]  stores the pid of the current doc
        # prevvmat = history[pid][-1]

        # return original instance, and the (input_memory_id(in th), mem, mmask, prev, pmask)
        # result = instance
        result = instance + (mem, mmask, prev, pmask, key_entity_tk, key_entity_mask)
        return result
        # # C). Memory info
        # # 4) get the memory related information
        # # 4.1) get the mem and mmask initilization part
        # if self.use_kwmem:
        #     # use keyword memory(entity level), initialize a empty mem [num_updates, 20+10, 768]
        #     # num_updates means : store all the possible memory for an example in an matrix
        #     mem = torch.torch.empty(self.h, self.max_len_key_entity + self.memsize, self.dim).normal_(std=.02)
        #     mmask = torch.zeros(self.max_len_key_entity + self.memsize).long()
        #     # the keyword mem mask is set according to the effective length
        #     mmask[0:len_effective_kw] = torch.ones(len_effective_kw).long()
        # else:
        #     mem = torch.torch.empty(self.h, self.memsize, self.dim).normal_(std=.02)
        #     mmask = torch.zeros(self.memsize).long()
        # mmask[-self.memsize:] = torch.ones(self.memsize).long()
        #
        # # 4.2) get the prev and pmask res
        # prev_embed_pkl_path = os.path.join(self.file_entity_emb_dir, self.split+'_' + str(docid)
        #                                    + '_encoded_unilm-mlm-4.pkl')
        # with open(prev_embed_pkl_path, 'rb') as doc_history_f:
        #     history = pickle.load(doc_history_f)
        #
        # prev = torch.zeros(self.h, 1, self.dim).float()  # .long() [10,1,768]
        # pmask = torch.zeros(self.h, 1).long()            #         [10, 1] - for the entire example, how many previous ones are filled.
        #
        # pid = len_effective_kw - 1 # pid now is the total length of the effective keywords
        # for p in range(1, min(pid+1, self.h+1)):  # at most, the model accept h=10 paragraph
        #     prev[p - 1, 0, :] = history[p][-1]  # the last item in the current pid tuple
        #     pmask[p - 1, 0] = torch.ones(1).long() # the last item's pmask
        #
        # # # 4) get the current doc, pid previous paragraph embedding
        # # assert pid == history[pid][1]  # history[p][1]  stores the pid of the current doc
        # # prevvmat = history[pid][-1]
        #
        # # return original instance, and the (input_memory_id(in th), mem, mmask, prev, pmask)
        # # result = instance
        # result = instance + (mem, mmask, prev, pmask,key_entity_tk, key_entity_mask)
        # return result

    def mem_tgt_span_assign(self, tgt_tk, tgt_token_list_of_list, tgt_span_split_words):
        '''
        return: tgt_span_label - extend the word level label from tgt_span_split_words to token level,
                refer to the structure of tgt_token_list_of_list
        params:
                tgt_tk: tgt token
                tgt_token_list_of_list: tgt tokens in a nested list, the 1st layer is the word level; the 2nd layer is token level.
                tgt_span_split_words:   tgt span split word for word level
        '''
        assert len(tgt_token_list_of_list) == len(tgt_span_split_words)
        span_label = [len(word_tokens_ls) * [tgt_span_split_words[id]] for id, word_tokens_ls in
                      enumerate(tgt_token_list_of_list)]
        span_label_flatten = [item for sublist in span_label for item in sublist]
        assert len(span_label_flatten) == len(tgt_tk)
        return span_label_flatten

    def span_tgt_span_assign(self, tgt_tk):
        '''
        param:
            1). tgt_tk: ['name', 'is', '[ENTITY_CLS]', 'r', '##og', '##er', 'la', '##ka', '[ENTITY_SEP]', ';',
        'position', 'is', '[ENTITY_CLS]', 'h', '##b', '[ENTITY_SEP]', ';',
        return:
            list of 0 for non-span, and 1 for within span.
        '''
        span_label = []
        entity_cls = '[ENTITY_CLS]'
        entity_sep = '[ENTITY_SEP]'

        within_span = False
        span_id = 0
        for token in tgt_tk:
            # start of the span
            if token == entity_cls:
                within_span = True
                span_id += 1
                span_label.append(span_id)
            elif token == entity_sep:
                within_span = False
                span_label.append(span_id)
            else:
                if within_span is True:
                    span_label.append(span_id)
                else:
                    span_label.append(0)

        assert len(span_label)==len(tgt_tk)

        return span_label

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False,
                 mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s",
                 has_oracle=False,
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, tokens_b, tokens_b_memory_id, tokens_b_span_id, max_num_entity = instance
        tokens_b_memory_id = [int(item) if int(item) < max_num_entity else max_num_entity - 1 for item in
                              tokens_b_memory_id]  # constrain the max reference of Memory

        num_truncated_a, num_truncated_b,\
        num_truncated_c_memory_id,num_truncated_d_spanmask_id = truncate_tokens_pair(tokens_a, tokens_b,
                                                                                          tokens_b_memory_id, tokens_b_span_id,
                                                                                          self.max_len - 3,
                                                                                          max_len_a=self.max_len_a,
                                                                                          max_len_b=self.max_len_b, trunc_seg=self.trunc_seg,
                                                                                          always_truncate_tail=self.always_truncate_tail)
        truncated_token_b = tokens_b[:]
        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        tokens_memory_id = [0]*(len(tokens_a)+1) + [0] + tokens_b_memory_id + [tokens_b_memory_id[-1]]
        tokens_span_id = [0]*(len(tokens_a)+1) + [0] + tokens_b_span_id + [0]
        assert len(tokens) == len(tokens_memory_id) == len(tokens_span_id)
        original_tokens = tokens


        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+['[SEP]'])
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(
                1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        tokens_memory_id.extend([0] * n_pad)
        tokens_span_id.extend([0] * n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        span_input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)

        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            span_input_mask[:, :len(tokens_a) + 2].fill_(1)  # 第一个segment 都是1

            second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            # span_mask
            # 2) the span-mask self attention for masked language model pretraining.
            # notice: the span label should be a. truncated, b. padded and c. added special token with token_a
            # assert len(tokens_span_id) == len(input_ids)
            # span_input_mask = self.create_span_input_mask(tokens_span_id, len(tokens_a)+len(tokens_b)+3) # todo: this function has problem. need to change.
            assert len(tokens_b) == len(tokens_b_span_id)
            tgt_input_mask = self.create_tgt_span_mask(tokens_b_span_id, second_st, second_end)
            span_input_mask[second_st:second_end, second_st:second_end].copy_(tgt_input_mask)
            assert len(tokens_b) == len(tokens_b_span_id)

        else:
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None

        return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids, masked_pos,
                masked_weights, -1, self.task_idx, span_input_mask, tokens_memory_id)

    def create_tgt_span_mask(self, tgt_span_labels_original, second_st, second_end):
        '''
        Goal: This distinguish between 0 and non-0 for the tgt_span_labels,
        and generate a tgt mask for attention to not attend to the words within the same span.
        Params:
            tgt_span_labels: 1 1 1 0 0 2 2 2 3 3 0 4 0 0 5 5
            second_st: the beginning of the second sentence.
            second_end: the end of the second sentence.
        '''
        # torch.tensor()
        # torch.zeros((self.max_len, self.max_len), dtype=torch.long)
        # note: give [SPE] a id->0, so that [SEP] can attend to all the information before it.

        tgt_span_labels = tgt_span_labels_original + [0] # noteL here, put an "0 " at the end, so the eos token can get the all 1 attention
        assert second_end-second_st == len(tgt_span_labels_original)+1
        pre_span_label = 0
        pre_row_list = [0] * 512
        span_label_list = []

        for i, span_label in enumerate(tgt_span_labels):
            span_label = int(span_label)
            current_row_list = [0] * 512
            if span_label == 0:
                # if the current lable is 0, then attend all of values before
                current_row_list[:i + 1] = [1] * (i + 1)
                # print("1-lenof current_row_list: ", len(current_row_list))
            elif span_label > 0:
                # if the current lable is 1, consider the previous span level
                if pre_span_label == 0:
                    current_row_list[:i + 1] = [1] * (i + 1)
                    # print("2-lenof current_row_list: ", len(current_row_list))
                elif pre_span_label > 0:
                    # 2 cases:
                    # 1) in the same span: the current label is the same as the previous one
                    if span_label == pre_span_label:
                        current_row_list[:i] = pre_row_list.copy()[:i]
                        current_row_list[i - 1] = 0  # change the previous one to 0
                        current_row_list[i] = 1
                        # print("3-lenof current_row_list: ", len(current_row_list))
                    # 2) start a new span: the current label is different from the previous one.
                    #                       attend all
                    else:
                        current_row_list[:i + 1] = [1] * (i + 1)
            span_label_list.append(current_row_list)
            # update:
            pre_span_label = span_label
            pre_row_list = current_row_list.copy()
        # # slice
        # span_label_list_slice = [span_list_zero_pad[i][:second_end - second_st] for i in range(second_end - second_st)]
        span_label_list_slice = [span_label_list[i][:second_end - second_st] for i in range(second_end - second_st)]
        assert len(span_label_list_slice) == second_end - second_st # num_rows == len(token_b) + 1(last [SEP])
        assert len(span_label_list_slice[0]) == second_end - second_st
        # prepare for the tensor datatype
        span_label_tensor = torch.tensor(span_label_list_slice)
        return span_label_tensor

    def create_span_input_mask(self, span_label, effctive_input_len):
        # 1) get stats for the different span_id in span_dict
        groups = groupby(span_label)
        result = [(label, sum(1 for _ in group)) for label, group in groups]
        span_dict = {}
        for item in result:
            key = item[0]
            value = item[1]
            span_dict[key] = value
        # 2) generate span mask
        prev_span_key = 0
        span_label_list = []
        for pos, span_label_key in enumerate(span_label):
            current_span_attn = [1] * effctive_input_len + [0] * (self.max_len-effctive_input_len)
            if span_label_key > 0:
                span_len = span_dict[span_label_key]
                if span_label_key != prev_span_key:
                    prev_span_key = span_label_key
                    span_start_pos = pos
                    span_end_pos = span_start_pos + span_len
                current_span_attn[span_start_pos: span_end_pos] = [0] * (span_end_pos-span_start_pos)
            span_label_list.append(current_span_attn)
        # prepare for the tensor datatype
        span_label_tensor = torch.tensor(span_label_list)
        return span_label_tensor

class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s", num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, max_a_len = instance

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]']
        else:
            padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                _enc_seg1 = 0 if self.s2s_share_segment else 4
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [
                            0] + [1]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                    else:
                        segment_ids = [
                            4] + [6]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                else:
                    segment_ids = [4]*(len(padded_tokens_a)) + \
                        [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        return (input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx)

class Preprocess4Seq2seqMemDecoder(Pipeline):
    """
    @ Wenting Zhao
    @ 0116 14:48
    Goal: Pre-processing steps for testing step

        # 1) for training, the seq loader gives the following memory related variables:
        # input_memory_id, mem, mmask, prev, pmask, key_entity_tk, key_entity_mask
        # 2) for the testing, I need the following additional vars.
        # mem:   [bs, 35, 768]
        # mmask: [bs, 35]
        # prev_entity_hidden: [bs, 768]
        # key_entity_tk:      [bs, self.h, max_cell_token]
        # key_entity_mask:    [bs, self.h]
     """
    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False,
                 max_len_key_entity=20, max_len_key_entity_tk=30, memsize=15, dim=768, tokenizer=None):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        # memory related attributes:
        # read the file into memory
        self.ex_list = []
        self.max_len_key_entity = max_len_key_entity  # max num of entities
        self.max_len_key_entity_tk = max_len_key_entity_tk  # max num tokens of each entity
        self.memsize = memsize
        self.dim = dim
        self.h = max_len_key_entity  # max num of previous entity
        self.tokenizer = tokenizer

    def prepare_key_entity_start_from_token(self, tokens_a):
        entity_start_tk = '[ENTITY_CLS]'
        entity_end_tk = '[ENTITY_SEP]'
        # 1) get the keywords list (word level) and store in key_words
        key_words = []
        single_kw = []
        in_span = False
        # 1) save all the entities into a list
        for item in tokens_a:
            if item == entity_start_tk:
                in_span = True
                single_kw.append(item)
            elif in_span is True:
                single_kw.append(item)
                if item == entity_end_tk:
                    in_span = False
                    key_words.append(single_kw)
                    single_kw = []
        key_words = key_words[:self.max_len_key_entity]
        # 2) get the keywords list (token level)
        tk_id = []
        mask = []
        for entity_tk in key_words:
            # 1) tokenize
            if len(entity_tk) > self.max_len_key_entity_tk:
                entity_tk = entity_tk[:self.max_len_key_entity_tk - 1] + [entity_end_tk]  # only get the beginning 30 tks

            entity_tk_ids = self.tokenizer.convert_tokens_to_ids(entity_tk)
            entity_mask = [1] * len(entity_tk)
            # 2) pad
            # if len(entity_tk) < self.max_len_key_entity_tk:
            n_pad = self.max_len_key_entity_tk - len(entity_tk)
            entity_tk_ids.extend([0] * n_pad)
            entity_mask.extend([0] * n_pad)
            # 3) add to the example res list
            tk_id.append(entity_tk_ids)
            mask.append(entity_mask)
        key_words = key_words[:self.h] # is this alright?

        # 3) get the truncated key word list
        effective_key_word_len = len(key_words)
        num_pad_entity = self.max_len_key_entity - effective_key_word_len
        pad_entity = [[0]*self.max_len_key_entity_tk]
        tk_id.extend(pad_entity*num_pad_entity)
        mask.extend(pad_entity*num_pad_entity)
        return tk_id,mask,effective_key_word_len

    def __call__(self, instance):
        # mem:   [bs, 35, 768]
        # mmask: [bs, 35]
        # prev_entity_hidden: [bs, 768]
        # key_entity_tk:      [bs, self.h, max_cell_token]
        # key_entity_mask:    [bs, self.h]
        tokens_a, max_a_len = instance

        # 1) key word recognition from the tokens_a
        # key_entity_tk:      [bs, self.h, max_cell_token]
        # key_entity_mask:    [bs, self.h]
        key_entity_tk, key_entity_mask, effective_key_entity_num = self.prepare_key_entity_start_from_token(tokens_a)

        # 2) mem:
        mem = torch.torch.empty(self.max_len_key_entity + self.memsize, self.dim).normal_(std=.02) # shape [35,768]
        mmask = torch.zeros(self.max_len_key_entity + self.memsize).long()
        # the keyword mem mask is set according to the effective length
        mmask[0:effective_key_entity_num] = torch.ones(effective_key_entity_num).long()

        # 3) prev_entity_hidden:
        prev_entity_hidden = torch.zeros(1, self.dim).long()

        # todo: check whether the res is expected.

        # Add Special Tokens
        if self.s2s_special_token:
            padded_tokens_a = ['[S2S_CLS]'] + tokens_a + ['[S2S_SEP]']
        else:
            padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                _enc_seg1 = 0 if self.s2s_share_segment else 4
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [
                            0] + [1]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                    else:
                        segment_ids = [
                            4] + [6]*(len(padded_tokens_a)-1) + [5]*(max_len_in_batch - len(padded_tokens_a))
                else:
                    segment_ids = [4]*(len(padded_tokens_a)) + \
                        [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        if self.num_qkv > 1:
            mask_qkv = [0]*(len(padded_tokens_a)) + [1] * \
                (max_len_in_batch - len(padded_tokens_a))
        else:
            mask_qkv = None

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # mem:   [bs, 35, 768]
        # mmask: [bs, 35]
        # prev_entity_hidden: [bs, 768]
        # key_entity_tk:      [bs, self.h, max_cell_token]
        # key_entity_mask:    [bs, self.h]
        res = (input_ids, segment_ids, position_ids, input_mask, mask_qkv, self.task_idx)
        res = res + (mem, mmask, prev_entity_hidden, key_entity_tk, key_entity_mask) # [20,35,768]
        return res