import json
import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam

import helper
from tensorize import CorefDataProcessor
import util
import time
import os
from os.path import join
from metrics import CorefEvaluator, MentionDetectionEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from model import CorefModel
import conll
import sys

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None, best_model_suffix=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)
        self.best_model_suffix = best_model_suffix

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config)

    def set_test_file(self, test_file, gold_conll=None):
        self.config['test_dir'] = test_file
        if gold_conll:
            self.config['gold_conll'] = gold_conll

    def initialize_model(self, saved_suffix=None):
        model = CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        if not self.config['prediction_only'] and self.config['pretrained_model'] != 'none' and saved_suffix is None:
            self.load_pretrained_checkpoint(model, self.config['pretrained_model'])
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()
        eval_frequency = len(examples_train)

        # Set up seed after tensorizing data
        util.set_seed(self.seed)

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) if d is not None else None for d in example]
                _, loss = model(*example_gpu)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))


                    # Evaluate
                    if conf['dev_dir'] != 'none' and not conf['is_pretraining'] and len(loss_history) > 0 and len(loss_history) % eval_frequency == 0:
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                        if f1 > max_f1:
                            max_f1 = f1
                            self.save_model_checkpoint(model, len(loss_history))
                            self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
                        logger.info('Eval max f1: %.2f' % max_f1)
                        start_time = time.time()


                    if conf['is_pretraining']:
                        if len(loss_history) in [10000, 20000, 30000]:
                            self.save_model_checkpoint(model, f'pretrained_{len(loss_history)}', remove=False)
                        elif len(loss_history) % len(examples_train) == 0:
                            if (not torch.isinf(loss)) and (not torch.isnan(loss)): self.save_model_checkpoint(model, 'pretrained', remove=False)
                    elif len(loss_history) % len(examples_train) == 0 and (len(loss_history) // len(examples_train)) % 5 == 0:
                        if (not torch.isinf(loss)) and (not torch.isnan(loss)): self.save_model_checkpoint(model, len(loss_history) / len(examples_train), remove=False)

                    if (torch.isinf(loss)) or ( torch.isnan(loss)):
                        logger.info(f'Stopping the training early because the loss is not correct. loss={loss}')
                        break

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Evaluate
        if not conf['is_pretraining']:
            if conf['dev_dir'] != 'none':
                f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                if f1 > max_f1:
                    max_f1 = f1
                    self.save_model_checkpoint(model, len(loss_history))
                    self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
                logger.info('Eval max f1: %.2f' % max_f1)
            else:
                if (not torch.isinf(loss)) and (not torch.isnan(loss)): self.save_model_checkpoint(model, len(loss_history) / len(examples_train), remove=False)
        else:
            if (not torch.isinf(loss)) and (not torch.isnan(loss)): self.save_model_checkpoint(model, 'pretrained', remove=False)
        logger.info('Training finished')

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        mention_evaluator = MentionDetectionEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:12]  # Strip out gold
            example_gpu = [d.to(self.device) if d is not None else None for d in tensor_example]
            with torch.no_grad():
                _, _, _, _, span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, span_pred_types, span_gold_types = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            span_speaker = span_speaker.to('cpu').numpy()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.to('cpu').numpy()
            predicted_clusters = model.update_evaluator(span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, gold_clusters, evaluator, span_pred_types)
            _ = model.update_evaluator(span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, gold_clusters, mention_evaluator, span_pred_types)
            doc_to_prediction[doc_key] = predicted_clusters

        p, r, f1 = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f1 * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        (p_men, r_men, f1_men), (p_sing, r_sing, f1_sing), (p_non_sing, r_non_sing, f1_non_sing) = mention_evaluator.get_prf()
        metrics = {'Precision_overall': p_men * 100, 'Recall_overall': r_men * 100, 'F1_overall': f1_men * 100,
                   'Precision_singletons': p_sing * 100, 'Recall_singletons': r_sing * 100,
                   'F1_singletons': f1_sing * 100,
                   'Precision_non_singletons': p_non_sing * 100, 'Eval_Avg_Recall_non_singletons': r_non_sing * 100,
                   'Eval_Avg_F1_non_singletons': f1_non_sing * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        # TODO
        output_path = join(self.config['log_dir'], 'devoutput_{}.jsonlines'.format(step))
        jsonlines_path = join(self.config['data_dir'], self.config['dev_dir'])

        # Input from file
        with open(jsonlines_path, 'r') as f:
            lines = f.readlines()
        docs = [json.loads(line) for line in lines]

        with open(output_path, 'w') as f:
            for i, doc in enumerate(docs):
                doc_key = doc['doc_key']
                docs[i]['predicted_clusters'] = doc_to_prediction[doc_key]
                # doc['clusters'] = predicted_clusters[i]  # MODIFIED
                f.write(json.dumps(docs[i]))
                f.write('\n')  # MODIFIED
        print(f'Saved prediction in {output_path}')

        json_path = output_path
        conll_path = json_path[:-len('.jsonlines')] + '.CONLLUA'
        helper.convert_coref_json_to_ua(json_path, conll_path, MODEL="coref-hoi")
        print(f'Converted to CONLLUA in {conll_path}')

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f1 * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            tensor_example = tensor_example#[:11] # Strip out gold
            example_gpu = [d.to(self.device) if d is not None else None for d in tensor_example]
            with torch.no_grad():
                [_, _, _, _, span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, span_pred_types, span_gold_types], _ = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            span_speaker = span_speaker.to('cpu').numpy()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.to('cpu').numpy()
            if span_gold_types is not None:
                span_gold_types = span_gold_types.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, span_speaker, antecedent_idx, antecedent_scores, span_pred_types, span_gold_types)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

            # del example_gpu
            # del spans
            # del antecedents
            # del clusters
            # del span_pred_types
            # del span_gold_types
            # torch.cuda.empty_cache()

            # print(len(mention_to_cluster_id))

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step, remove=True):
        if self.config['debug']:
            return
        if remove:
            _, _, files = next(os.walk(self.config['log_dir']))
            files = [fn for fn in files if 'model_{}'.format(self.name_suffix) in fn and 'bin' in fn]
            files.sort(key = lambda x: int(x[:-4].split('_')[3]), reverse=True)
            for fn in files[self.config['save_top_k'] - 1:]:
                os.remove(join(self.config['log_dir'], fn))
        if 'pretrained' in str(step):
            path_ckpt = join(self.config['log_dir'], f'model_{step}.bin')
        else:
            path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)

    def load_pretrained_checkpoint(self, model, pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % pretrained_path)


class Evaluator:
    def __init__(self, runner):
        self.runner = runner

    def conll_score(self):
        if self.runner.best_model_suffix is not None:
            model_to_evaluate = self.runner.initialize_model(self.runner.best_model_suffix)
        else:
            _, _, files = next(os.walk(self.runner.config['log_dir']))
            files = [fn for fn in files if fn.startswith('model_') and 'bin' in fn]
            # files = [fn for fn in files if 'model_{}'.format(self.runner.name_suffix) in fn and 'bin' in fn]
            files.sort(key = lambda x: int(x[:-4].split('_')[3]), reverse=True)
            fn = files[0]

            model_to_evaluate = self.runner.initialize_model(fn[6:-4])

        data_processor = self.runner.data
        jsonlines_path = join(self.runner.config['data_dir'], self.runner.config['test_dir'])
        output_path = join(self.runner.config['data_dir'],
                           'output_{}_{}_{}.jsonlines'.format(self.runner.name, self.runner.best_model_suffix,
                                                              '_'.join(self.runner.config['test_dir'].split('_')[-6:]).rstrip('.jsonlines')))

        # Input from file
        with open(jsonlines_path, 'r') as f:
            lines = f.readlines()
        docs = [json.loads(line) for line in lines]
        tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input(docs)
        predicted_clusters, _, _ = self.runner.predict(model_to_evaluate, tensor_examples)

        with open(output_path, 'w') as f:
            for i, doc in enumerate(docs):
                all_mentions = set()
                for c in predicted_clusters[i]:
                    for m in c:
                        all_mentions.add(tuple(m))
                for _, c in enumerate(docs[i]['predicted_clusters']):
                    for __, m in enumerate(c):
                        if tuple(m) not in all_mentions:
                            # print(m, c, docs[i]['predicted_clusters'][_])
                            docs[i]['predicted_clusters'][_].remove(m)
                    if len(docs[i]['predicted_clusters'][_]) == 0:
                        docs[i]['predicted_clusters'].remove(docs[i]['predicted_clusters'][_])

                # docs[i]['predicted_clusters'] =
                # doc['clusters'] = predicted_clusters[i]  # MODIFIED
                f.write(json.dumps(docs[i]))
                f.write('\n')  # MODIFIED
        print(f'Saved prediction in {output_path}')

        json_path = output_path
        conll_path = json_path[:-len('.jsonlines')] + '.CONLLUA'
        helper.convert_coref_json_to_ua(json_path, conll_path, MODEL="coref-hoi")
        print(f'Converted to CONLLUA in {conll_path}')

        if not self.runner.config['prediction_only']:

            mde = MentionDetectionEvaluator()
            for doc in docs:
                mention_to_gold = {(men_st, men_ed): cluster for cluster in doc['clusters'] for (men_st, men_ed, men_ty) in cluster}
                mention_to_sys = {(men_st, men_ed): cluster for cluster in doc['predicted_clusters'] for (men_st, men_ed) in cluster}
                mde.update(None, None, mention_to_sys=mention_to_sys, mention_to_gold=mention_to_gold)
            logger.info(vars(mde))
            (p, r, f1), (p_sing, r_sing, f1_sing), (p_non_sing, r_non_sing, f1_non_sing) = mde.get_prf()
            metrics = {'Precision_overall': p * 100, 'Recall_overall': r * 100, 'F1_overall': f1 * 100,
                       'Precision_singletons': p_sing * 100, 'Recall_singletons': r_sing * 100,
                       'F1_singletons': f1_sing * 100,
                       'Precision_non_singletons': p_non_sing * 100, 'Eval_Avg_Recall_non_singletons': r_non_sing * 100,
                       'Eval_Avg_F1_non_singletons': f1_non_sing * 100}
            for name, score in metrics.items():
                logger.info('%s: %.2f' % (name, score))

            gold_conll_path = self.runner.config['gold_conll']
            import subprocess
            scores = subprocess.check_output(['python', 'ua-scorer.py', gold_conll_path, conll_path], universal_newlines=True)
            for sc in scores.split('\n'):
                logger.info(sc)

def batch_evaluate(gpu_id):
    test_files = [
        ('output_codi2022_arpred_second_step_subm_tl0_NoPronounUtt_GoldMen_30_noMenScores_pronCon_Jul06_20-25-42_15.0_output_codi2022_arpred_first_step_subm_tl500_30_nocons_Jun28_16-06-56_10.0_light_test_no_men.2022.ar.pron10_ns3.jsonlines', None),
        ('output_codi2022_arpred_second_step_subm_tl0_NoPronounUtt_GoldMen_30_noMenScores_pronCon_Jul06_20-25-42_15.0_output_codi2022_arpred_first_step_subm_tl500_30_nocons_Jun28_16-06-56_20.0_AMI_test_no_men.2022.split.ar.pron10.split_ns1.jsonlines', None),
        ('output_codi2022_arpred_second_step_subm_tl0_NoPronounUtt_GoldMen_30_noMenScores_pronCon_Jul06_20-25-42_25.0_output_codi2022_arpred_first_step_subm_tl500_30_nocons_Jun28_16-06-56_20.0_Persuasion_test_no_men.2022.ar.pron10_ns1.jsonlines', None),
        ('output_codi2022_arpred_second_step_subm_tl0_NoPronounUtt_GoldMen_30_noMenScores_pronCon_Jul06_20-25-42_25.0_output_codi2022_arpred_first_step_subm_tl500_30_nocons_Jun28_16-06-56_10.0_Switchboard_test_Phase_1.2022.ar.pron10_ns1.jsonlines', None),
    ]
    configs = [
        # subm
        ('codi2022ARPred_third_step_subm_tl500_GoldMen_30_nocons', 'model_Jul06_01-37-25_10.0.bin', 11),
        ('codi2022ARPred_third_step_subm_tl500_GoldMen_30_nocons', 'model_Jul06_01-37-25_15.0.bin', 11),
        ('codi2022ARPred_third_step_subm_tl500_GoldMen_30_nocons', 'model_Jul06_01-37-25_20.0.bin', 11),
        ('codi2022ARPred_third_step_subm_tl500_GoldMen_30_nocons', 'model_Jul06_01-37-25_25.0.bin', 11),
        ('codi2022ARPred_third_step_subm_tl500_GoldMen_30_nocons', 'model_Jul06_01-37-25_30.0.bin', 11),
        ('codi2022ARPred_third_step_subm_tl500_GoldMen_30_nocons', 'model_Jul06_01-37-25_5.0.bin', 11),
    ]
    for config_name, model_to_evaluate, seed_to_eval in configs:
        runner = Runner(config_name.rstrip('/'), gpu_id, seed=seed_to_eval,
                        best_model_suffix=model_to_evaluate.lstrip('model_').rstrip('.bin/'))
        for test_fn, gold_conll in test_files:
            runner.config['prediction_only'] = True
            runner.set_test_file(test_fn, gold_conll)
            evaluator = Evaluator(runner)
            evaluator.conll_score()
            del evaluator
            torch.cuda.empty_cache()
        del runner
        torch.cuda.empty_cache()



if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    else:
        seed = 11
    if config_name == 'batch':
        batch_evaluate(gpu_id)
    else:
        runner = Runner(config_name, gpu_id, seed)
        model = runner.initialize_model()
        runner.train(model)

        # # Evaluation
        # if not runner.config['is_pretraining']:
        #     del model
        #     evaluator = Evaluator(runner)
        #     evaluator.conll_score()