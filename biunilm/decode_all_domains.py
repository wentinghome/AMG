"""BERT finetuning runner."""

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling_0321_low_rouge import BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import biunilm.seq2seq_loader as seq2seq_loader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main(model_recover_path, output_path,src_file,beam_size,span_ckpt_path,length_penalty):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--span_ckpt_path", default=span_ckpt_path, type=str, required=False)

    parser.add_argument("--bert_model", default='bert-base-cased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--model_recover_path", default=model_recover_path, type=str,
                        help="The file path of fine-tuned pretraining model.")

    parser.add_argument("--max_seq_length", default=367, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n" # todo: change here to accomdate the GPU1
                             "than this will be padded.")
    parser.add_argument("--max_len_key_entity", default=20, type=int,
                        help="The maximum num of key entity.")
    parser.add_argument("--max_len_key_entity_tk", default=30, type=int,
                        help="The maximum num of key entity.")

    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--src_file",
                        default=src_file, type=str,
                        help="The input data file name.")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", default=output_path, type=str, help="output file")
    parser.add_argument("--split", type=str, default=split,
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', default=False,
                        # action='store_true',
                        help="Whether the input is tokenized.")
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
    parser.add_argument('--beam_size', type=int, default=beam_size,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=length_penalty,
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

    print("args.model_recover_path", args.model_recover_path)
    print("args.src_file", args.src_file)
    print("args.output_file", args.output_file)

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case,
    never_split=("[UNK]", "[SEP]", "[X_SEP]", "[PAD]", "[CLS]", "[MASK]", "[ENTITY_CLS]", "[ENTITY_SEP]"))

    tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqMemDecoder(list(tokenizer.vocab.keys()),
                                                                    tokenizer.convert_tokens_to_ids, args.max_seq_length,
                                                                    max_tgt_length=args.max_tgt_length,
                                                                    new_segment_ids=args.new_segment_ids,
                                                                    mode="s2s", num_qkv=args.num_qkv,
                                                                    s2s_special_token=args.s2s_special_token,
                                                                    s2s_add_segment=args.s2s_add_segment,
                                                                    s2s_share_segment=args.s2s_share_segment,
                                                                    pos_shift=args.pos_shift,
                                                                    tokenizer=tokenizer))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
        (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]", "[S2S_SOS]"])

    def _get_token_id_set(s):
        r = None
        if s:
            w_list = []
            for w in s.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            r = set(tokenizer.convert_tokens_to_ids(w_list))
        return r

    forbid_ignore_set = _get_token_id_set(args.forbid_ignore_word)
    not_predict_set = _get_token_id_set(args.not_predict_token)
    print(args.model_recover_path)
    # 1) notice: this actually use a different model from the the mlm model.
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
                                                      num_rel=pair_num_relation, type_vocab_size=type_vocab_size,
                                                      task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                      length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
                                                      forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                                                      forbid_ignore_set=forbid_ignore_set, not_predict_set=not_predict_set,
                                                      ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
                                                      max_position_embeddings=args.max_seq_length, ffn_type=args.ffn_type,
                                                      num_qkv=args.num_qkv, seg_emb=args.seg_emb, pos_shift=args.pos_shift,
                                                      max_len_key_entity=args.max_len_key_entity,
                                                      max_len_key_entity_tk=args.max_len_key_entity_tk,
                                                      tokenizer=tokenizer,span_ckpt_path=args.span_ckpt_path)
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

        with open(args.src_file, encoding="utf-8") as fin:
            input_lines = [x.strip() for x in fin.readlines()]
            if args.subset > 0:
                logger.info("Decoding subset: %d", args.subset)
                input_lines = input_lines[:args.subset]
        data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
        input_lines = [data_tokenizer.tokenize(
            x)[:max_src_length] for x in input_lines]
        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx,\
                        mem, mmask, prev_entity_hidden, key_entity_tk, key_entity_mask = batch
                    memory_args = mem, mmask, prev_entity_hidden, key_entity_tk, key_entity_mask

                    traces = model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv,
                                   device=device, memory_args=memory_args, mode='test')
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_lines[buf_id[i]] = output_sequence
                        # print("output_sequence: ", output_sequence)
                        if args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                pbar.update(1)
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

        if args.need_score_traces:
            with open(fn_out + ".trace.pickle", "wb") as fout_trace:
                pickle.dump(
                    {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
                for x in score_trace_list:
                    pickle.dump(x, fout_trace)


if __name__ == "__main__":
    # decode the validation set and find one with the highest score.

    # num_train_epochs = 2
    # split_list = ["test"]
    # num_list = [100, 50]
    # domain_list = ["humans", "songs"]
    # domain_folder_list = ["1-wiki-human-folder", "2-wiki-song-folder"]

    num_train_epochs = 50
    split_list = ["test"]
    num_list = [200, 100, 50]
    domain_list = ["humans", "songs", "books"]
    domain_folder_list = ["1-wiki-human-folder", "2-wiki-song-folder", "3-wiki-books-folder"]

    beam_size = 3
    length_penalty = 1

    for domain, domain_folder in zip(domain_list, domain_folder_list):
        for num in num_list:
            for split in split_list:
                model_check_point_dir = '/data0/temp_folder_for_Table-SpanMem_fine-tune_0515/'+\
                                        domain+'/' + 'example_'+ str(num) +'_saved_model'
                span_pt_path = '/data0/'+domain_folder+'/4_pre_entity_embed_folder/' \
                               'CKPT/'+domain + '-unilm-span-mlm-entire-model.pt'

                output_dir = '/data0/temp_folder_for_Table-SpanMem_fine-tune_0515/'+\
                                        domain+'/' + 'example_'+ str(num) +'_predict_test'

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                test_path = '/data0/' + domain_folder + '/1_original_data_500/example-' + str(num) + '/' + \
                split + '/' + split + '.box.original.att_value_linear_span'

                model_check_point_path = os.path.join(model_check_point_dir, "model." + str(num_train_epochs) + ".bin")
                print(model_check_point_path)
                assert Path(model_check_point_path).exists(
                ), "model_check_point_path doesn't exist"

                output_path = os.path.join(output_dir, domain+'_'+str(num)+"_beam_"+str(beam_size)+
                                           "_lp_" + str(length_penalty) +"_ckpt_" + str(num_train_epochs) + ".txt")
                main(model_check_point_path, output_path=output_path,
                     src_file=test_path, beam_size=beam_size, span_ckpt_path=span_pt_path,
                     length_penalty=length_penalty)
