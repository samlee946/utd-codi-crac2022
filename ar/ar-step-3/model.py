import torch
import torch.nn as nn
from transformers import BertModel
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init
import higher_order as ho


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device
        self.debug = config['debug']

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        # self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])
        # MODIFIED
        self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']
        # if config['use_type_for_singletons']:
        #

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config['use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'], [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config['use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] else None

        self.span_type_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=config['num_entity_types']) if config['use_type_for_singletons'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config['coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config['higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'], [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if config['higher_order'] == 'cluster_merging' else None


        self.update_steps = 0  # Internal use for debug

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_pronoun,
                                 is_training, candidate_starts, candidate_ends, span_speaker, span_pronoun,
                                 gold_starts=None, gold_ends=None, gold_types=None, gold_mention_cluster_map=None):

        if is_pronoun is not None:
            assert is_pronoun.shape[0] == candidate_starts.shape[0]

        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            assert gold_types is not None
            do_loss = True

        # Get token emb
        mention_doc, _ = self.bert(input_ids, attention_mask=input_mask)  # [num words, bert_emb_size] -> not correct # [num seg, num max tokens, emb size]
        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)

            if not conf['non_overlaps']:
                overlapping_gold_spans = same_span.view(same_span.shape).sum(0).gt(1)
                for idx in torch.nonzero(overlapping_gold_spans): # has overlapping spans in gold
                    nonzero_list_span = same_span[:, idx[0]].nonzero()
                    selected_one = torch.randint(nonzero_list_span.shape[0], (1,))
                    selected_one = nonzero_list_span[selected_one]
                    same_span[nonzero_list_span, idx] = 0
                    same_span[selected_one, idx] = 1

            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float), same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long), 0)  # [num candidates]; non-gold span has label 0

            candidate_types = torch.matmul(torch.unsqueeze(gold_types, 0).to(torch.float), same_span.to(torch.float))
            candidate_types = torch.squeeze(candidate_types.to(torch.long), 0)  # [num candidates]; non-gold span has type 0

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        # candidate_emb_list = [span_start_emb, span_end_emb]
        candidate_span_emb = torch.cat([span_start_emb, span_end_emb], dim=1)
        if is_training == 2:
            del span_start_emb
            del span_end_emb
        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_span_emb = torch.cat([candidate_span_emb, candidate_width_emb], dim=1)
            # candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        if len(candidate_starts) > 7000 and is_training == 2:
            # this is to use CPU during inference just in case of OOM
            candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device='cpu'), 0).repeat(num_candidates, 1)
            candidate_starts_cpu = candidate_starts.to('cpu')
            candidate_ends_cpu = candidate_ends.to('cpu')
            candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts_cpu, 1)) & (
                        candidate_tokens <= torch.unsqueeze(candidate_ends_cpu, 1))
            if conf['model_heads']:
                token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
            else:
                token_attn = torch.ones(num_words, dtype=torch.float, device='cpu')  # Use avg if no attention
            token_attn = token_attn.to('cpu')
            candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn,0)
            candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
            head_attn_emb_cpu = torch.matmul(candidate_tokens_attn, mention_doc.to('cpu'))
            torch.cuda.empty_cache()
            candidate_span_emb_cpu = torch.cat([candidate_span_emb.to('cpu'), head_attn_emb_cpu], dim=1)
            candidate_span_emb = candidate_span_emb_cpu.to(device)
        else:
            candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
            candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
            if conf['model_heads']:
                token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
            else:
                token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
            candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
            candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
            head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
            candidate_span_emb = torch.cat([candidate_span_emb, head_attn_emb], dim=1)
        # candidate_emb_list.append(head_attn_emb)
        # candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        # import ipdb
        # ipdb.set_trace()

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # Get span type score
        if conf['use_type_for_singletons']:
            candidate_mention_type_raw_scores = torch.squeeze(self.span_type_ffnn(candidate_span_emb), 1)
            candidate_mention_type_scores, candidate_mention_type_ids = torch.topk(candidate_mention_type_raw_scores, 1)

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        if conf['is_second_step']:
            num_top_spans = len(candidate_starts)
            selected_idx = torch.arange(0, num_top_spans, dtype=torch.long, device=device)
        else:
            num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
            selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu, candidate_ends_cpu, num_top_spans)
            assert len(selected_idx_cpu) == num_top_spans
            selected_idx = torch.tensor(selected_idx_cpu, device=device)
        # print(num_top_spans)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        if is_pronoun is not None:
            top_span_is_pronoun = is_pronoun[selected_idx]
        top_span_speaker = span_speaker[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]                             # [num_top_spans, self.span_emb_size]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None  # [num_top_spans]
        top_span_gold_types = candidate_types[selected_idx] if do_loss else None    # [num_top_spans]
        top_span_pred_types = candidate_mention_type_ids[selected_idx]
        top_span_mention_scores = candidate_mention_scores[selected_idx]            # [num_top_spans]
        top_span_pronoun = span_pronoun[selected_idx]
        top_span_first_person = (top_span_pronoun == 1) | (top_span_pronoun == 9)
        top_span_second_person = (top_span_pronoun == 2) | (top_span_pronoun == 10)
        top_span_third_person = (top_span_pronoun == 3) | (top_span_pronoun == 11)
        top_span_fourth_person = (top_span_pronoun == 4) | (top_span_pronoun == 12)
        top_span_fifth_person = (top_span_pronoun == 5)
        top_span_sixth_person = (top_span_pronoun == 6)
        top_span_seventh_person = (top_span_pronoun == 7)
        top_span_eighth_person = (top_span_pronoun == 8)
        top_span_ninth_person = (top_span_pronoun == 9)
        top_span_tenth_person = (top_span_pronoun == 10)
        top_span_eleventh_person = (top_span_pronoun == 11)
        top_span_twelveth_person = (top_span_pronoun == 12)

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        end_sent_idx = sentence_map[top_span_ends]
        start_sent_idx = sentence_map[top_span_starts]
        antecedent_sentence_distance = torch.abs(torch.unsqueeze(start_sent_idx, 1) - torch.unsqueeze(end_sent_idx, 0))
        bucketed_antecedent_sentence_distance = util.bucket_distance(antecedent_sentence_distance)
        antecedent_mask = (antecedent_offsets >= 1)

        if self.config['pronoun_prev_utt'] >= 0:
            mask_pronoun_utt_num = antecedent_sentence_distance[top_span_is_pronoun] <= self.config['pronoun_prev_utt']
            antecedent_mask[top_span_is_pronoun] &= mask_pronoun_utt_num

        # span_pronoun
        # this part of pronoun constarints can be found in our paper https://aclanthology.org/2021.codi-sharedtask.2
        if conf['use_pronoun_constraints']:
            same_speaker = (torch.unsqueeze(top_span_speaker, 1) == torch.unsqueeze(top_span_speaker, 0))
            antecedent_mask_pronoun = torch.ones(antecedent_mask.shape, dtype=torch.bool, device=self.device)
            mask_fs = (torch.unsqueeze(top_span_first_person, 1) & torch.unsqueeze(top_span_second_person, 0))
            mask_ft = (torch.unsqueeze(top_span_first_person, 1) & torch.unsqueeze(top_span_third_person, 0))
            mask_f4 = (torch.unsqueeze(top_span_first_person, 1) & torch.unsqueeze(top_span_fourth_person, 0))
            mask_sf = (torch.unsqueeze(top_span_second_person, 1) & torch.unsqueeze(top_span_first_person, 0))
            mask_st = (torch.unsqueeze(top_span_second_person, 1) & torch.unsqueeze(top_span_third_person, 0))
            mask_s4 = (torch.unsqueeze(top_span_second_person, 1) & torch.unsqueeze(top_span_fourth_person, 0))
            mask_tf = (torch.unsqueeze(top_span_third_person, 1) & torch.unsqueeze(top_span_first_person, 0))
            mask_ts = (torch.unsqueeze(top_span_third_person, 1) & torch.unsqueeze(top_span_second_person, 0))
            mask_t4 = (torch.unsqueeze(top_span_third_person, 1) & torch.unsqueeze(top_span_fourth_person, 0))
            mask_4f = (torch.unsqueeze(top_span_fourth_person, 1) & torch.unsqueeze(top_span_first_person, 0))
            mask_4s = (torch.unsqueeze(top_span_fourth_person, 1) & torch.unsqueeze(top_span_second_person, 0))
            mask_4t = (torch.unsqueeze(top_span_fourth_person, 1) & torch.unsqueeze(top_span_third_person, 0))
            antecedent_mask_pronoun[mask_fs] = False
            antecedent_mask_pronoun[mask_ft] = False
            antecedent_mask_pronoun[mask_f4] = False
            antecedent_mask_pronoun[mask_sf] = False
            antecedent_mask_pronoun[mask_st] = False
            antecedent_mask_pronoun[mask_s4] = False
            antecedent_mask_pronoun[mask_tf] = False
            antecedent_mask_pronoun[mask_ts] = False
            antecedent_mask_pronoun[mask_t4] = False
            antecedent_mask_pronoun[mask_4f] = False
            antecedent_mask_pronoun[mask_4s] = False
            antecedent_mask_pronoun[mask_4t] = False
            antecedent_mask[same_speaker] = antecedent_mask[same_speaker] & antecedent_mask_pronoun[same_speaker]

            antecedent_mask_pronoun = torch.ones(antecedent_mask.shape, dtype=torch.bool, device=self.device)
            mask_ff = (torch.unsqueeze(top_span_first_person, 1) & torch.unsqueeze(top_span_first_person, 0))
            mask_ss = (torch.unsqueeze(top_span_second_person, 1) & torch.unsqueeze(top_span_second_person, 0))
            antecedent_mask_pronoun[mask_ff] = False
            antecedent_mask_pronoun[mask_ss] = False
            antecedent_mask[~same_speaker] = antecedent_mask[~same_speaker] & antecedent_mask_pronoun[~same_speaker]

            # Here
            antecedent_mask_pronoun = torch.ones(antecedent_mask.shape, dtype=torch.bool, device=self.device)
            mask_7_5 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_fifth_person, 0))
            mask_7_6 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_sixth_person, 0))
            mask_7_8 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_7_9 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_ninth_person, 0))
            mask_7_10 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_tenth_person, 0))
            mask_7_11 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_eleventh_person, 0))
            mask_7_12 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_twelveth_person, 0))
            mask_12_7 = (torch.unsqueeze(top_span_twelveth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_11_7 = (torch.unsqueeze(top_span_eleventh_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_10_7 = (torch.unsqueeze(top_span_tenth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_9_7 = (torch.unsqueeze(top_span_ninth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_8_7 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_6_7 = (torch.unsqueeze(top_span_sixth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_5_7 = (torch.unsqueeze(top_span_fifth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            antecedent_mask_pronoun[mask_7_5] = False
            antecedent_mask_pronoun[mask_7_6] = False
            antecedent_mask_pronoun[mask_7_8] = False
            antecedent_mask_pronoun[mask_7_9] = False
            antecedent_mask_pronoun[mask_7_10] = False
            antecedent_mask_pronoun[mask_7_11] = False
            antecedent_mask_pronoun[mask_7_12] = False
            antecedent_mask_pronoun[mask_12_7] = False
            antecedent_mask_pronoun[mask_11_7] = False
            antecedent_mask_pronoun[mask_10_7] = False
            antecedent_mask_pronoun[mask_9_7] = False
            antecedent_mask_pronoun[mask_8_7] = False
            antecedent_mask_pronoun[mask_6_7] = False
            antecedent_mask_pronoun[mask_5_7] = False
            # print(torch.sum(top_span_seventh_person))
            # print(torch.sum(~antecedent_mask_pronoun))
            antecedent_mask = antecedent_mask & antecedent_mask_pronoun

            # There
            antecedent_mask_pronoun = torch.ones(antecedent_mask.shape, dtype=torch.bool, device=self.device)
            mask_8_5 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_fifth_person, 0))
            mask_8_6 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_sixth_person, 0))
            mask_8_7 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_seventh_person, 0))
            mask_8_9 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_ninth_person, 0))
            mask_8_10 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_tenth_person, 0))
            mask_8_11 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_eleventh_person, 0))
            mask_8_12 = (torch.unsqueeze(top_span_eighth_person, 1) & torch.unsqueeze(top_span_twelveth_person, 0))
            mask_12_8 = (torch.unsqueeze(top_span_twelveth_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_11_8 = (torch.unsqueeze(top_span_eleventh_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_10_8 = (torch.unsqueeze(top_span_tenth_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_9_8 = (torch.unsqueeze(top_span_ninth_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_7_8 = (torch.unsqueeze(top_span_seventh_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_6_8 = (torch.unsqueeze(top_span_sixth_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            mask_5_8 = (torch.unsqueeze(top_span_fifth_person, 1) & torch.unsqueeze(top_span_eighth_person, 0))
            antecedent_mask_pronoun[mask_8_5] = False
            antecedent_mask_pronoun[mask_8_6] = False
            antecedent_mask_pronoun[mask_8_7] = False
            antecedent_mask_pronoun[mask_8_9] = False
            antecedent_mask_pronoun[mask_8_10] = False
            antecedent_mask_pronoun[mask_8_11] = False
            antecedent_mask_pronoun[mask_8_12] = False
            antecedent_mask_pronoun[mask_12_8] = False
            antecedent_mask_pronoun[mask_11_8] = False
            antecedent_mask_pronoun[mask_10_8] = False
            antecedent_mask_pronoun[mask_9_8] = False
            antecedent_mask_pronoun[mask_7_8] = False
            antecedent_mask_pronoun[mask_6_8] = False
            antecedent_mask_pronoun[mask_5_8] = False
            # print(torch.sum(top_span_eighth_person))
            # print(torch.sum(~antecedent_mask_pronoun))
            antecedent_mask = antecedent_mask & antecedent_mask_pronoun

        if conf['type_prediction_constraint']:
            selected_idx_is_entity_mention = (candidate_mention_type_ids[selected_idx] == conf['num_entity_types'] - 1).squeeze(1)
            antecedent_mask[selected_idx_is_entity_mention,:] = False
            antecedent_mask[:,selected_idx_is_entity_mention] = False

        # import ipdb
        # ipdb.set_trace()

        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)          # [num_top_spans, max_top_antecedents]

        if len(antecedent_mask) <= 0:
            import ipdb
            ipdb.set_trace()

        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx, device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents, 1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0, self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores, self.emb_cluster_size, self.cluster_score_ffnn, None, self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'], easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx, top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores, self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)  # [num top spans, max top antecedents + 1]
            return candidate_starts, candidate_ends, span_speaker, candidate_mention_scores, top_span_starts, top_span_ends, top_span_speaker, top_antecedent_idx, top_antecedent_scores, top_span_pred_types, top_span_gold_types

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        if conf['loss_type'] == 'marginalized':
            log_marginalized_antecedent_scores = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif conf['loss_type'] == 'hinge':
            top_antecedent_mask = torch.cat([torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # Add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * conf['mention_loss_coef']
            loss += loss_mention

        # Add type loss
        if conf['type_loss_coef']:
            gold_mention_type = top_span_gold_types
            pred_mention_scores = candidate_mention_type_raw_scores[selected_idx]

            # logger.info('Max label pred: {}'.format(torch.max(top_span_pred_types)))
            # logger.info('Max label gold: {}'.format(torch.max(top_span_gold_types)))

            loss_mention_type = nn.CrossEntropyLoss()(pred_mention_scores, gold_mention_type)
            loss += loss_mention_type * conf['type_loss_coef']

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm
        self.update_steps += 1

        return [candidate_starts, candidate_ends, span_speaker, candidate_mention_scores, top_span_starts, top_span_ends, top_span_speaker, top_antecedent_idx, top_antecedent_scores, top_span_pred_types, top_span_gold_types], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx, key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, top_span_pred_types=None, top_span_gold_types=None):
        """ CPU list input """
        # Get predicted antecedents
        # Check speaker info
        # for idx, l in enumerate(antecedent_idx):
        #     mask = (span_speaker[l] != span_speaker[idx])
        #     mask = np.insert(mask, 0, False, axis=0)
        #     antecedent_scores[idx][mask] = -np.inf

        if top_span_pred_types is not None:
            top_span_pred_types = np.squeeze(top_span_pred_types.cpu().numpy(), 1)
        antecedent_scores = antecedent_scores.tolist()
        # non-referring and non-mentions
        # for i, _ in enumerate(antecedent_idx):
        #     for __, j in enumerate(_):
        #         if top_span_pred_types[j] == 0:
        #             antecedent_scores[i][__] = antecedent_scores[i][0] - 1

        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            # if top_span_pred_types[i] == 0: # non-referring and non-mentions
            #     continue
            if predicted_idx < 0:
                continue
            try:
                assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            except:
                continue
            # Check speaker info
            # if span_speaker[i] != span_speaker[predicted_idx]:
            #     continue

            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        # Add singletons
        if top_span_pred_types is not None:
            for i, predicted_idx in enumerate(span_starts):
                if self.config['type_prediction_constraint']:
                    assert top_span_pred_types[i] == self.config['num_entity_types'] - 1
                # Check if it exists in a cluster
                # if top_span_pred_types[i] == 0: # not an entity
                #     continue
                # if top_span_pred_types[i] == 1: # is a non-refering entity
                #     # if top_span_gold_types is not None:
                #     #     if not top_span_gold_types[i] == 2:
                #     #         continue
                #     # else:
                #     continue
                if top_span_pred_types[i] == 0: # non-referring and non-mentions
                    continue
                mention = (int(span_starts[i]), int(span_ends[i]))
                mention_cluster_id = mention_to_cluster_id.get(mention, -1)
                if mention_cluster_id == -1:
                    mention_cluster_id = len(predicted_clusters)
                    predicted_clusters.append([mention])
                    mention_to_cluster_id[mention] = mention_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]

        if self.debug:
            import ipdb
            ipdb.set_trace()
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, gold_clusters, evaluator, span_pred_types):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, span_pred_types)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple((m[0], m[1])) for m in cluster) for cluster in gold_clusters] # modified
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        # logger.info(predicted_clusters)
        # logger.info(gold_clusters)
        return predicted_clusters