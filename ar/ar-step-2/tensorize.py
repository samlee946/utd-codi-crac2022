import util
import numpy as np
import random
import os
from os.path import join
import json
import pickle
import logging
import torch

logger = logging.getLogger(__name__)


class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = language

        self.max_seg_len = config['max_segment_len']
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']
        self.train_dir = config['train_dir']  # MODIFIED
        self.dev_dir = config['dev_dir']  # MODIFIED
        # self.test_dir = config['test_dir'] # MODIFIED

        self.tokenizer = util.get_tokenizer(config['bert_tokenizer_name'])
        self.tensor_samples, self.stored_info = None, None  # For dataset samples; lazy loading

    def get_tensor_examples_from_custom_input(self, samples):
        """ For interactive samples; no caching """
        tensorizer = Tensorizer(self.config, self.tokenizer)
        tensor_samples = []
        for sample in samples:
            for _ in tensorizer.tensorize_example(sample, 2):
                tensor_samples.append(_)
        tensor_samples = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
        return tensor_samples, tensorizer.stored_info

    def get_tensor_examples(self):
        """ For dataset samples """
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info(f'Loaded tensorized examples from cache file {cache_path}')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config, self.tokenizer)
            paths = {
                # 'trn': join(self.data_dir, f'train.{self.language}.{self.max_seg_len}.jsonlines'),
                # 'dev': join(self.data_dir, f'dev.{self.language}.{self.max_seg_len}.jsonlines'),
                # 'tst': join(self.data_dir, f'test.{self.language}.{self.max_seg_len}.jsonlines')
                'trn': join(self.data_dir, f'{self.train_dir}')
                # 'tst': join(self.data_dir, f'{self.test_dir}')
            }
            self.tensor_samples['dev'] = None
            if not self.config['is_pretraining'] and self.dev_dir != 'none':
                paths['dev'] = join(self.data_dir, f'{self.dev_dir}')
            for split, path in paths.items():
                logger.info('Tensorizing examples from %s; results will be cached)' % path)
                training_flag = 0 if (split == 'trn') else 1 if (split == 'dev') else 0
                with open(path, 'r') as f:
                    samples = [json.loads(line) for line in f.readlines()]

                tensor_samples = []
                for sample in samples:
                    for _ in tensorizer.tensorize_example(sample, training_flag):
                        tensor_samples.append(_)

                self.tensor_samples[split] = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor
                                              in tensor_samples]
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], None  # self.tensor_samples['tst']

    def get_stored_info(self):
        return self.stored_info

    @classmethod
    def convert_to_torch_tensor(cls, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_pronoun,
                                is_training, candidate_starts, candidate_ends, span_speaker, span_pronoun,
                                gold_starts, gold_ends, gold_types, gold_mention_cluster_map):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        genre = torch.tensor(genre, dtype=torch.long)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long)
        is_pronoun = torch.tensor(is_pronoun, dtype=torch.bool) if is_pronoun is not None else None
        is_training = torch.tensor(is_training, dtype=torch.long)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_types = torch.tensor(gold_types, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        candidate_starts = torch.tensor(candidate_starts, dtype=torch.long)
        candidate_ends = torch.tensor(candidate_ends, dtype=torch.long)
        span_speaker = torch.tensor(span_speaker, dtype=torch.long)
        span_pronoun = torch.tensor(span_pronoun, dtype=torch.long)
        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_pronoun, \
               is_training, candidate_starts, candidate_ends, span_speaker, span_pronoun, \
               gold_starts, gold_ends, gold_types, gold_mention_cluster_map,

    def get_cache_path(self):
        # cache_path = join(self.data_dir, f'cached.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
        cache_path = join(self.data_dir,
                          f'cached.tensors.{self.train_dir}.{self.dev_dir}.ErrMen{self.config["error_men_in_second_step"]}.{bool(self.config["is_pretraining"])}.utt{bool(self.config["pronoun_prev_utt"]>0)}.{self.config["max_span_width"]}.{self.max_seg_len}.{self.max_training_seg}.bin')  # MODIFIED
        return cache_path


class Tensorizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Will be used in evaluation
        self.stored_info = {}
        self.stored_info['tokens'] = {}  # {doc_key: ...}
        self.stored_info['subtoken_maps'] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info['gold'] = {}  # {doc_key: ...}
        self.stored_info['genre_dict'] = {genre: idx for idx, genre in enumerate(config['genres'])}
        with open(config['entity_type_list'], 'r') as f:
            self.stored_info['entity_type_dict'] = {entity_type.strip(): idx for idx, entity_type in
                                                    enumerate(f.readlines())}
        print(self.stored_info['entity_type_dict'])

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_spans_default_entity(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends), np.array([label_dict['entity'] for _ in starts])

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, training_flag):
        # Mentions and clusters
        clusters = example['clusters']
        gold_mentions = sorted(tuple(mention) for mention in util.flatten(clusters))
        # print(gold_mentions)
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1  # modified

        # Speakers
        speakers = example['speakers']
        speaker_dict = self._get_speaker_dict(util.flatten(speakers))

        # Sentences/segments
        sentences = example['sentences']  # Segments
        sentences_flattened = util.flatten(sentences)
        sentence_map = example['sentence_map']
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.config['max_segment_len']
        sentence_len = np.array([len(s) for s in sentences])

        # Bert input
        input_ids, input_mask, speaker_ids = [], [], []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['gold'][doc_key] = example['clusters']
        # self.stored_info['tokens'][doc_key] = example['tokens']

        # Construct example
        genre = self.stored_info['genre_dict'].get(doc_key[:2], 0)
        # gold_starts, gold_ends = self._tensorize_spans(gold_mentions)

        if self.config['pronoun_prev_utt'] >= 0:
            pronoun_starts, pronoun_ends = self._tensorize_spans(example['pronouns'])
        else:
            pronoun_starts, pronoun_ends = None, None
        if len(gold_mentions) and len(gold_mentions[0]) == 2:
            gold_starts, gold_ends, gold_types = self._tensorize_spans_default_entity(gold_mentions, self.stored_info[
                'entity_type_dict'])
        else:
            gold_starts, gold_ends, gold_types = self._tensorize_span_w_labels(gold_mentions,
                                                                               self.stored_info['entity_type_dict'])

        predicted_starts, predicted_ends = [], []
        if 'predicted_clusters' in example:
            predicted_mentions = sorted(tuple(mention) for mention in util.flatten(example['predicted_clusters']))
            predicted_starts, predicted_ends = self._tensorize_spans(predicted_mentions)

        example_tensor = (
            input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, pronoun_starts, pronoun_ends, predicted_starts, predicted_ends,
            training_flag,
            gold_starts, gold_ends, gold_types, gold_mention_cluster_map)

        if training_flag == 0 and len(sentences) > self.config['max_training_sentences']:
            example_tensors = self.truncate_example(*example_tensor)
        # elif training_flag == 1 and len(sentences) > self.config['max_evaluating_sentences']:
        #     example_tensor = self.truncate_example(*example_tensor, max_sentences=self.config['max_evaluating_sentences'])
        else:
            example_tensors = [example_tensor]

        # print(training_flag)

        # Get candidate span
        ret_example_tensors = []
        for example_tensor in example_tensors:
            (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, pronoun_starts, pronoun_ends, predicted_starts, predicted_ends,
             training_flag,
             gold_starts, gold_ends, gold_types, gold_mention_cluster_map) = example_tensor
            num_words = len(sentence_map)
            if not self.config['is_second_step']:
                candidate_starts = np.repeat(np.expand_dims(np.arange(0, num_words), 1), self.config['max_span_width'],
                                             axis=1)
                candidate_ends = candidate_starts + np.arange(0, self.config['max_span_width'])
                # candidate_starts = np.repeat(np.expand_dims(np.arange(0, num_words), 1), 10, axis=1) # for AMI only
                # candidate_ends = candidate_starts + np.arange(0, 10)
                candidate_mask = (candidate_ends < num_words)
                speaker_ids_flattened = util.flatten(speaker_ids)
                candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[
                    candidate_mask]  # [num valid candidates]
                # print(len(speaker_ids_flattened), max(candidate_ends))
            else:
                if self.config['error_men_in_second_step'] or training_flag == 2:
                    assert 'predicted_clusters' in example
                    candidate_starts = predicted_starts
                    candidate_ends = predicted_ends
                else:
                    candidate_starts = gold_starts
                    candidate_ends = gold_ends

                valid_candidate_mask = (candidate_starts >= 0) & (candidate_ends < num_words) & ((candidate_ends - candidate_starts) < self.config['max_span_width'])
                candidate_starts = candidate_starts[valid_candidate_mask]
                candidate_ends = candidate_ends[valid_candidate_mask]
                speaker_ids_flattened = util.flatten(speaker_ids)
                ## sanity check passed on 2022/07/02
                # all_predicted_mentions = set([(st, ed) for st, ed in zip(candidate_starts, candidate_ends)])
                # all_gold_mentions = set([(st, ed) for st, ed in zip(gold_starts, gold_ends) if ed - st < self.config['max_span_width']])
                # for m in all_gold_mentions:
                #     try:
                #         assert m in all_predicted_mentions
                #     except:
                #         import ipdb
                #         ipdb.set_trace()

            candidate_mask = np.ones(candidate_starts.shape, dtype=np.bool)
            span_speaker = np.zeros(candidate_starts.shape, dtype=np.int)
            span_pronoun = np.zeros(candidate_starts.shape, dtype=np.int)

            # if example['doc_key'] == 'light_dev/episode_8109':
            #     import ipdb
            #     ipdb.set_trace()

            for idx, (st, ed) in enumerate(zip(candidate_starts, candidate_ends)):
                span_speakers = set(speaker_ids_flattened[st:ed + 1]) - {speaker_dict['[SPL]']}
                if len(span_speakers) > 1 and self.config['use_speaker_constraints']:
                    candidate_mask[idx] = False
                elif len(span_speakers) == 1:
                    # span_speaker[idx] = speaker_dict[list(span_speakers)[0]]
                    span_speaker[idx] = list(span_speakers)[0]
                    tok = example['tokens'][self.stored_info['subtoken_maps'][doc_key][st]].lower()
                    if ed - st < 5:
                        if tok in ['i', 'me', 'my', 'mine']:
                            span_pronoun[idx] = 1
                        elif tok in ['you', 'your', 'yours']:
                            span_pronoun[idx] = 2
                        elif tok in ['he', 'him', 'his']:
                            span_pronoun[idx] = 3
                        elif tok in ['she', 'her']:
                            span_pronoun[idx] = 4
                    if st == ed:
                        if tok == 'here':
                            span_pronoun[idx] = 7
                        elif tok == 'there':
                            span_pronoun[idx] = 8
                        elif tok in ['my']:
                            span_pronoun[idx] = 9
                        elif tok in ['your']:
                            span_pronoun[idx] = 10
                        elif tok in ['his']:
                            span_pronoun[idx] = 11
                        elif tok in ['her']:
                            span_pronoun[idx] = 12
                        elif tok in ['their']:
                            span_pronoun[idx] = 5
                        elif tok in ['it', 'its']:
                            span_pronoun[idx] = 6

            candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[
                candidate_mask]  # [num valid candidates]
            span_speaker, span_pronoun = span_speaker[candidate_mask], span_pronoun[candidate_mask]
            # print(len(candidate_starts))

            if self.config['pronoun_prev_utt'] >= 0:
                same_start = (np.expand_dims(candidate_starts, 1) == np.expand_dims(pronoun_starts, 0))
                same_end = (np.expand_dims(candidate_ends, 1) == np.expand_dims(pronoun_ends, 0))
                same_span = (same_start & same_end)
                is_pronoun = (same_span.sum(1) > 0)
            else:
                is_pronoun = None

            if (len(gold_starts) and len(candidate_starts)) or training_flag == 2:
                example_tensor = (
                    input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_pronoun, training_flag,
                    candidate_starts, candidate_ends, span_speaker, span_pronoun,
                    gold_starts, gold_ends, gold_types, gold_mention_cluster_map)
                ret_example_tensors.append((doc_key, example_tensor))

        return ret_example_tensors

    def truncate_example(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, pronoun_starts,
                         pronoun_ends, predicted_starts, predicted_ends, is_training,
                         gold_starts, gold_ends, gold_types, gold_mention_cluster_map, max_sentences=None,
                         sentence_offset=None):
        if max_sentences is None:
            max_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences
        truncated_examples = []
        for sent_offset_st in range(0, num_sentences, max_sentences):
            sent_offset_ed = min(sent_offset_st + max_sentences, num_sentences)
            sent_offset_st = sentence_len[:sent_offset_st].sum()
            word_offset = sentence_len[:sent_offset_st].sum()
            num_words = sentence_len[sent_offset_st:sent_offset_ed].sum()

            truncated_input_ids = input_ids[sent_offset_st:sent_offset_ed, :]
            truncated_input_mask = input_mask[sent_offset_st:sent_offset_ed, :]
            truncated_speaker_ids = speaker_ids[sent_offset_st:sent_offset_ed, :]
            truncated_sentence_len = sentence_len[sent_offset_st:sent_offset_ed]

            truncated_sentence_map = sentence_map[word_offset: word_offset + num_words]
            truncated_gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
            truncated_gold_starts = gold_starts[truncated_gold_spans] - word_offset
            truncated_gold_ends = gold_ends[truncated_gold_spans] - word_offset
            truncated_gold_types = gold_types[truncated_gold_spans]
            truncated_gold_mention_cluster_map = gold_mention_cluster_map[truncated_gold_spans]

            if len(predicted_starts):
                truncated_predicted_spans = (predicted_starts < word_offset + num_words) & (predicted_ends >= word_offset)
                truncated_predicted_starts = predicted_starts[truncated_predicted_spans] - word_offset
                truncated_predicted_ends = predicted_ends[truncated_predicted_spans] - word_offset
            else:
                truncated_predicted_starts, truncated_predicted_ends = [], []

            truncated_pronoun_starts, truncated_pronoun_ends = None, None
            if pronoun_starts is not None:
                truncated_pronoun_spans = (pronoun_starts < word_offset + num_words) & (pronoun_ends >= word_offset)
                truncated_pronoun_starts = pronoun_starts[truncated_pronoun_spans] - word_offset
                truncated_pronoun_ends = pronoun_ends[truncated_pronoun_spans] - word_offset

            truncated_examples.append(
                (truncated_input_ids, truncated_input_mask, truncated_speaker_ids, truncated_sentence_len,
                 genre, truncated_sentence_map, truncated_pronoun_starts, truncated_pronoun_ends, truncated_predicted_starts, truncated_predicted_ends,
                 is_training, truncated_gold_starts, truncated_gold_ends, truncated_gold_types,
                 truncated_gold_mention_cluster_map)
            )

        # sent_offset = sentence_offset
        # if sent_offset is None:
        #     sent_offset = random.randint(0, num_sentences - max_sentences)
        #
        # input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        # input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        # speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        # sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]
        #
        # sentence_map = sentence_map[word_offset: word_offset + num_words]
        # gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        # # gold_spans = (gold_starts >= word_offset) & (gold_ends < word_offset + num_words)
        # gold_starts = gold_starts[gold_spans] - word_offset
        # gold_ends = gold_ends[gold_spans] - word_offset
        # gold_types = gold_types[gold_spans]
        # gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]
        #
        # if pronoun_starts is not None:
        #     pronoun_spans = (pronoun_starts < word_offset + num_words) & (pronoun_ends >= word_offset)
        #     pronoun_starts = pronoun_starts[pronoun_spans] - word_offset
        #     pronoun_ends = pronoun_ends[pronoun_spans] - word_offset
        #
        # return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, pronoun_starts, pronoun_ends, \
        #        is_training, gold_starts, gold_ends, gold_types, gold_mention_cluster_map

        return truncated_examples
