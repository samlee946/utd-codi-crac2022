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


# logger.addHandler(logging.FileHandler(join('/users/sxl180006/research/codi2021/datadir_dd', 'log_batch_non_refer.txt'), 'a'))

class Runner:
    def __init__(self, config_name, gpu_id=0, best_model_suffix=None, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed
        self.best_model_suffix = best_model_suffix

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(self.seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config)

        # Set up seed
        if seed:
            util.set_seed(self.seed)

    def set_test_file(self, test_file):
        self.config['test_dir'] = test_file

    def initialize_model(self, saved_suffix=None):
        model = CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def initialize_model_from_state_dict(self, state_dict):
        model = CorefModel(self.config, self.device)
        model.load_state_dict(state_dict, strict=False)
        logger.info('Loaded model from combine')
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
        model.stored_info = stored_info

        random.seed(self.seed)
        training_samples_ids = [random.sample(range(len(examples_train)), k=len(examples_train)) for _ in range(epochs)]
        # print(training_samples_ids[:5])

        # Set up seed
        if self.seed:
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
        max_f1_test_set = 0
        max_f1_test_set_at_epoch = -1
        max_anaphor_f1 = 0
        max_f1_at_epoch = -1
        max_anaphor_f1_at_epoch = -1
        start_time = time.time()
        model.zero_grad()

        stats_dir = join(conf['log_dir'], 'stats.txt')
        with open(stats_dir, 'w') as f_stats:
            model.config['stats_dir'] = stats_dir

        for epo in range(epochs):
            # random.shuffle(examples_train)  # Shuffle training set
            # for doc_key, example in examples_train:
            for _idx in training_samples_ids[epo]:
                doc_key, example = examples_train[_idx]
                # print(doc_key)
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) if d is not None else None for d in example]
                _, loss = model(*example_gpu)

                # print stats
                model.config['print_stats'] = False
                if len(loss_history) % 100 == 0:
                    model.config['print_stats'] = True
                    with open(stats_dir, 'a') as f_stats:
                        f_stats.write(f'Stats for training step #{len(loss_history)}, doc_key = {doc_key}\n')


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
                    if examples_dev and len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        save_this_checkpoint = False
                        self.save_model_checkpoint(model, len(loss_history))
                        f1, _, anaphor_r, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history),
                                                            official=False, conll_path=self.config['conll_eval_path'],
                                                            tb_writer=tb_writer)

                        # f1_test, _, _, _ = self.evaluate(model, examples_test, stored_info, len(loss_history),
                        #                                  official=False, conll_path=self.config['conll_eval_path'],
                        #                                  tb_writer=tb_writer)

                        if f1 > max_f1:
                            max_f1 = f1
                            max_f1_at_epoch = len(loss_history)
                            save_this_checkpoint = True
                            self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
                        # if f1_test > max_f1_test_set:
                        #     max_f1_test_set = f1_test
                        #     max_f1_test_set_at_epoch = len(loss_history)
                        #     save_this_checkpoint = True
                        if anaphor_r > max_anaphor_f1:
                            max_anaphor_f1_at_epoch = len(loss_history)
                            max_anaphor_f1 = anaphor_r
                            # save_this_checkpoint = True
                            # self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'

                        # if save_this_checkpoint:
                        #     self.save_model_checkpoint(model, len(loss_history))

                        logger.info('Eval max f1: %.2f' % max_f1)
                        logger.info('Eval max f1 on test set: %.2f' % max_f1_test_set)
                        logger.info('Eval max anaphor f1: %.2f' % max_anaphor_f1)
                        start_time = time.time()

                    if len(loss_history) % len(examples_train) == 0 and (len(loss_history) // len(examples_train)) % 5 == 0:
                        self.save_model_checkpoint(model, len(loss_history) / len(examples_train))

                if 'stop_at_epoch' in self.config and len(loss_history) > self.config['stop_at_epoch']:
                    break

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        if examples_dev:
            # Evaluate
            save_this_checkpoint = False
            self.save_model_checkpoint(model, len(loss_history))
            f1, _, anaphor_r, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False,
                                                conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)

            # f1_test, _, _, _ = self.evaluate(model, examples_test, stored_info, len(loss_history),
            #                                  official=False, conll_path=self.config['conll_eval_path'],
            #                                  tb_writer=tb_writer)
            if f1 > max_f1:
                max_f1 = f1
                max_f1_at_epoch = len(loss_history)
                save_this_checkpoint = True
                self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
            # if f1_test > max_f1_test_set:
            #     max_f1_test_set = f1_test
            #     max_f1_test_set_at_epoch = len(loss_history)
            #     save_this_checkpoint = True
            if anaphor_r > max_anaphor_f1:
                max_anaphor_f1_at_epoch = len(loss_history)
                max_anaphor_f1 = anaphor_r
                # save_this_checkpoint = True
                # self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
            # if save_this_checkpoint:
            #     self.save_model_checkpoint(model, len(loss_history))
            logger.info('Final Eval max f1 (epoch %d): %.2f' % (max_f1_at_epoch, max_f1))
            logger.info('Final Eval max f1 on test set (epoch %d): %.2f' % (max_f1_test_set_at_epoch, max_f1_test_set))
            logger.info('Final Eval max anaphor f1 (epoch %d): %.2f' % (max_anaphor_f1_at_epoch, max_anaphor_f1))

        if len(loss_history) % len(examples_train) == 0 and (len(loss_history) / len(examples_train)) % 5 == 0:
            self.save_model_checkpoint(model, len(loss_history) / len(examples_train))
        logger.info('Finished training')

        del model

        # _, _, files = next(os.walk(self.config['log_dir']))
        # files = [fn for fn in files if 'model_{}'.format(self.name_suffix) in fn and 'bin' in fn]
        # path_ckpt = join(self.config['log_dir'], files[0])
        # state_dict = torch.load(path_ckpt, map_location=torch.device(self.device))
        # for fn in files[1:]:
        #     path_ckpt = join(self.config['log_dir'], fn)
        #     new_state_dict = torch.load(path_ckpt, map_location=torch.device(self.device))
        #     for layer in state_dict:
        #         state_dict[layer] += new_state_dict[layer]
        # for layer in state_dict:
        #     state_dict[layer] /= len(files)
        # model_to_evaluate = self.initialize_model_from_state_dict(state_dict)
        # self.save_combined_model(model_to_evaluate, max_f1_at_epoch, remove=True)

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
            tensor_example = tensor_example[:-4]  # Strip out gold
            example_gpu = [d.to(self.device) if d is not None else None for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, candidate_anaphor_mask, top_anaphor_pred_types = model(
                    *example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                        gold_clusters, evaluator, candidate_anaphor_mask,
                                                        top_anaphor_pred_types)
            _ = model.update_evaluator_anaphor_detection(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                         gold_clusters, mention_evaluator, candidate_anaphor_mask,
                                                         top_anaphor_pred_types)
            doc_to_prediction[doc_key] = predicted_clusters

            # print('eva', len(predicted_clusters))

            # import ipdb
            # ipdb.set_trace()

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        (p_men, r_men, f1_men), (p_sing, r_sing, f1_sing), (
            p_non_sing, r_non_sing, f1_non_sing) = mention_evaluator.get_prf()
        metrics_men = {'Precision_overall': p_men * 100, 'Recall_overall': r_men * 100, 'F1_overall': f1_men * 100,
                       'Precision_singletons': p_sing * 100, 'Recall_singletons': r_sing * 100,
                       'F1_singletons': f1_sing * 100,
                       'Precision_non_singletons': p_non_sing * 100, 'Recall_non_singletons': r_non_sing * 100,
                       'F1_non_singletons': f1_non_sing * 100}
        for name, score in metrics_men.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        if self.config['constraint_antecedent_utter'] in ['constraint_antecedent_utter', 'use_predicted_type_as_constraint', 'no_type_constraint']:
            return f * 100, metrics, f1_men * 100, metrics_men
        else:
            raise NotImplementedError

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            print(doc_key)
            tensor_example = tensor_example[:-4]  # strip out gold
            example_gpu = [d.to(self.device) if d is not None else None for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, candidate_anaphor_mask, top_anaphor_pred_types = model(
                    *example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends,
                                                                                        antecedent_idx,
                                                                                        antecedent_scores,
                                                                                        candidate_anaphor_mask,
                                                                                        top_anaphor_pred_types)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

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
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'],
                 weight_decay=0)
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

    def save_model_checkpoint(self, model, step, remove_prev=False):
        if self.config['debug']:
            return
        if remove_prev:
            _, _, files = next(os.walk(self.config['log_dir']))
            files = [fn for fn in files if 'model_{}'.format(self.name_suffix) in fn and 'bin' in fn]
            files.sort(key = lambda x: int(x[:-4].split('_')[3]), reverse=True)
            for fn in files:
                epoch = int(fn[:-4].split('_')[-1])
                os.remove(join(self.config['log_dir'], fn))
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def save_combined_model(self, model, best_model_on_dev, remove=False):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_combine.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)
        if remove:
            _, _, files = next(os.walk(self.config['log_dir']))
            files = [fn for fn in files if f'model_{self.name_suffix}' in fn and 'bin' in fn]
            # files.sort(key=lambda x: int(x[:-4].split('_')[3]), reverse=True)
            for fn in files:
                if 'combine' not in fn and str(best_model_on_dev) not in fn.strip('.bin').split('_'):
                    os.remove(join(self.config['log_dir'], fn))

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


class Evaluator:
    def __init__(self, runner):
        self.runner = runner
        self.custom_name = None

    def conll_score(self):
        if self.runner.best_model_suffix is not None:
            model_to_evaluate = self.runner.initialize_model(self.runner.best_model_suffix)
        else:
            _, _, files = next(os.walk(self.runner.config['log_dir']))
            files = [fn for fn in files if fn.startswith('model_') and 'bin' in fn]
            # files = [fn for fn in files if 'model_{}'.format(self.runner.name_suffix) in fn and 'bin' in fn]
            files.sort(key=lambda x: int(x[:-4].split('_')[3]), reverse=True)
            fn = files[0]

            model_to_evaluate = self.runner.initialize_model(fn[6:-4])

        data_processor = self.runner.data
        jsonlines_path = join(self.runner.config['data_dir'], self.runner.config['test_dir'])
        model_epoch = str(self.runner.best_model_suffix).split('_')[-1]
        test_fn = ''
        if 'ami' in str(self.runner.config['test_dir']).lower():
            test_fn = 'ami'
        elif 'light' in str(self.runner.config['test_dir']).lower():
            test_fn = 'light'
        elif 'pers' in str(self.runner.config['test_dir']).lower():
            test_fn = 'pers'
        elif 'swbd' in str(self.runner.config['test_dir']).lower() or 'switchboard' in str(
                self.runner.config['test_dir']).lower():
            test_fn = 'swbd'
        else:
            print(self.runner.config['test_dir'])
            raise ValueError
        if 'dev' in str(self.runner.config['test_dir']).lower():
            test_fn += '_dev'
        elif 'test' in str(self.runner.config['test_dir']).lower():
            test_fn += '_test'
        else:
            print(self.runner.config['test_dir'])
            raise ValueError
        runner_name = self.runner.name
        if self.custom_name is not None:
            runner_name = runner_name.replace(self.custom_name[0], self.custom_name[1])

        output_path = join(self.runner.config['data_dir'],
                           'output_{}_{}_{}_{}.jsonlines'.format(runner_name, model_epoch, self.runner.name_suffix,
                                                                 test_fn))

        # Input from file
        with open(jsonlines_path, 'r') as f:
            lines = f.readlines()
        with open(output_path, 'w') as f:
            pass
        docs = [json.loads(line) for line in lines]
        for doc in docs:
            tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
            model_to_evaluate.stored_info = stored_info
            predicted_clusters, _, _ = self.runner.predict(model_to_evaluate, tensor_examples)

            with open(output_path, 'a') as f:
                doc['predicted_clusters_dd'] = predicted_clusters[0]
                # doc['clusters'] = predicted_clusters[i]  # MODIFIED
                f.write(json.dumps(doc))
                f.write('\n')  # MODIFIED
        print(f'Saved prediction in {output_path}')

        json_path = output_path
        conll_path = json_path[:-len('.jsonlines')] + '.CONLLUA'
        helper.convert_coref_json_to_ua(json_path, conll_path, 'anaphor', MODEL="coref-hoi", dd=True)
        print(f'Converted prediction to CONLLUA in {conll_path}')

        if not self.runner.config['prediction_only']:
            gold_conll_path = join(self.runner.config['data_dir'], self.runner.config['gold_conll'])
            import subprocess
            scores = subprocess.check_output(
                ['python', 'ua-scorer.py', gold_conll_path, conll_path, 'evaluate_discourse_deixis'],
                universal_newlines=True)
            for sc in scores.split('\n'):
                logger.info(sc)


def batch_evaluate(gpu_id, seed):
    # test_files = ['dd_light_test_gold_men.jsonlines',
    #               'dd_AMI_test_gold_men.jsonlines',
    #               'dd_Persuasion_test_gold_men.jsonlines',
    #               'dd_swbd_test_gold_men.jsonlines']
    # test_files = ['ar_dd_AMI_test_no_men.jsonlines',
    #                 'ar_dd_Persuasion_test_no_men.jsonlines',
    #                 'ar_dd_light_test_no_men.jsonlines',
    # #                 'ar_dd_swbd_test_no_men.jsonlines']
    # test_files = [  'arddnaacl_TupAna_AMI_test_no_men.jsonlines',
    #                 'arddnaacl_TupAna_Persuasion_test_no_men.jsonlines',
    #                 'arddnaacl_TupAna_light_test_no_men.jsonlines',
    #                 'arddnaacl_TupAna_swbd_test_no_men.jsonlines']
    # test_files = [
    #     'arddnaacl_pred_next_onlylight100_dev.jsonlines',
    #     'arddnaacl_pred_next_onlyami100_dev.jsonlines',
    #     'arddnaacl_pred_next_onlypers100_dev.jsonlines',
    #     'arddnaacl_pred_next_onlyswbd100_dev.jsonlines',
    # ]
    # test_files = [
    #     'arddnaacl_pred_next_onlylight100_dev.jsonlines',
    #     'just_for_stat_purpose_next_light_test.jsonlines',
    #     # 'arddnaacl_pred_next_onlyami100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_AMI_test.jsonlines',
    #     # 'arddnaacl_pred_next_onlypers100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_Persuasion_test.jsonlines',
    #     # 'arddnaacl_pred_next_onlyswbd100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_Switchboard_test.jsonlines',
    # ]
    # test_files = [
    #     # 'arddnaacl_pred_next_dep10_onlylight100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_dep10_light_test.jsonlines',
    #     # 'arddnaacl_pred_next_dep10_onlyami100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_dep10_AMI_test.jsonlines',
    #     # 'arddnaacl_pred_next_dep10_onlypers100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_dep10_Persuasion_test.jsonlines',
    #     # 'arddnaacl_pred_next_dep10_onlyswbd100_dev.jsonlines',
    #     # 'just_for_stat_purpose_next_dep10_Switchboard_test.jsonlines',
    #     # 'arddnaacl_pred_itself_dep10_AMI_dev.jsonlines',
    #     # 'arddnaacl_pred_itself_dep10_AMI_test.jsonlines',
    #     # "arddnaacl_pred_itself_dep10_onlypers100_dev.jsonlines",
    #     # 'arddnaacl_pred_itself_dep10_Persuasion_test.jsonlines',
    #     # 'arddnaacl_pred_itself_dep10_Switchboard_test.jsonlines',
    #     # "arddnaacl_pred_itself_dep10_onlyswbd100_dev.jsonlines",
    #     # 'arddnaacl_pred_itself_dep10_light_test.jsonlines',
    #     # 'arddnaacl_pred_itself_dep10_light_dev.jsonlines',
    # ]

    # 1026 joint
    # configs = [('dd_pipeline_first_model_ml1_d1_tl1000_ts_0_3_predicted_as_constraint', 'model_Oct25_20-22-02_4000.bin'),
    #            ('dd_pipeline_first_model_ml1_d1_tl200_ts_0_2_predicted_as_constraint', 'model_Oct26_01-04-36_4000.bin'),
    #            ('dd_pipeline_first_model_ml1_d1_tl900_ts_0_1_predicted_as_constraint', 'model_Oct25_22-52-31_8000.bin'),
    #            ]
    # configs = [
    #             # features
    #             # mlrgpu09 first batch
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat100_ami', 'model_Jan11_09-35-57_4000.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat110_ami', 'model_Jan11_16-59-10_7500.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat010_ami', 'model_Jan11_20-14-40_6500.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat100_light', 'model_Jan11_11-44-43_8500.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat110_light', 'model_Jan11_18-58-38_11000.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat010_light', 'model_Jan11_21-33-49_8500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat100_pers', 'model_Jan11_11-44-44_2500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat110_pers', 'model_Jan11_16-59-08_10500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat010_pers', 'model_Jan11_20-02-15_8500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat100_swbd', 'model_Jan11_10-28-48_10500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat110_swbd', 'model_Jan11_18-30-08_12840.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat010_swbd', 'model_Jan11_21-32-53_11500.bin', 11),
    #             # vast batch
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat111_light', 'model_Jan12_03-05-23_6000.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat011_light', 'model_Jan12_05-28-08_4500.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat001_light', '', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat111_ami', 'model_Jan12_01-52-25_9500.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat011_ami', 'model_Jan12_04-15-55_3000.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat001_ami', 'model_Jan12_06-36-58_3500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat111_pers', 'model_Jan12_01-52-44_2000.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat011_pers', 'model_Jan12_04-37-27_1500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat111_swbd', 'model_Jan12_03-15-15_11500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat011_swbd', 'model_Jan12_05-58-03_3500.bin', 11),
    #             # mlrgpu09 second batch
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat101_light', 'model_Jan11_23-20-57_2500.bin', 11),
    #            # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat101_ami', 'model_Jan12_01-12-16_2000.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat001_pers', 'model_Jan11_23-20-14_5500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat101_pers', 'model_Jan12_00-15-07_1500.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat001_swbd', 'model_Jan12_00-16-19_8000.bin', 11),
    #            # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat101_swbd', 'model_Jan12_01-11-45_8000.bin', 11),
    #
    #             # dep parsing
    #             # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat100_DepParsing_ami', 'model_Jan12_02-47-26_3000.bin', 11),
    #             # ('baseline_AnaphorNextFiltering_HasNull_10Closest_Feat101_DepParsing_light', 'model_Jan12_02-47-20_8000.bin', 11),
    #             ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat110_DepParsing_pers', 'model_Jan12_03-41-26_6000.bin', 11),
    #             # ('baseline_AnaphorInTrainingSet_HasNull_10Closest_Feat100_DepParsing_swbd', 'model_Jan12_03-41-15_12000.bin', 11),
    # ]

    # configs = [
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_1000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_1500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_2000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_2500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_3000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_3500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_4000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_4500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_5000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_5500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_6000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_6500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_7000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_7500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_8000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_8500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_9000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_9500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_10000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_10500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_11000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_11500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_12000.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_12500.bin'),
    #     ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_12960.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_1000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_1500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_2000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_2500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_3000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_3500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_4000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_4500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_5000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_5500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_6000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_6500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_7000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_7500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_8000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_8500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_9000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_9500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_10000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_10500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_11000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_11500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_12000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_12500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_12960.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_1000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_1500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_2000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_2500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_3000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_3500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_4000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_4500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_5000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_5500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_6000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_6500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_7000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_7500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_8000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_8500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_9000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_9500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_10000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_10500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_11000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_11500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_12000.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_12500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_12600.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_1000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_1500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_2000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_2500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_3000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_3500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_4000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_4500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_5000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_5500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_6000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_6500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_7000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_7500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_8000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_8500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_9000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_9500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_10000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_10500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_11000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_11500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_12000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_12500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_12600.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_1000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_1500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_2000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_2500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_3000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_3500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_4000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_4500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_5000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_5500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_6000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_6500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_7000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_7500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_8000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_8500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_9000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_9500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_10000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_10500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_11000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_11500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_12000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_12500.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_12540.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_1000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_1500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_2000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_2500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_3000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_3500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_4000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_4500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_5000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_5500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_6000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_6500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_7000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_7500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_8000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_8500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_9000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_9500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_10000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_10500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_11000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_11500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_12000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_12500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_12540.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_1000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_1500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_2000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_2500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_3000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_3500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_4000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_4500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_5000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_5500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_6000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_6500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_7000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_7500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_8000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_8500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_9000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_9500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_10000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_10500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_11000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_11500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_12000.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_12500.bin'),
    #     # ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_12840.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_1000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_1500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_2000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_2500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_3000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_3500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_4000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_4500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_5000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_5500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_6000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_6500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_7000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_7500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_8000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_8500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_9000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_9500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_10000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_10500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_11000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_11500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_12000.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_12500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_12840.bin'),
    # ]

    # test_files = [
    #     # 'arddnaacl_gold_stat_itself_dep10_AMI_dev.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_AMI_test.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_Persuasion_dev.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_Persuasion_test.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_Switchboard_dev.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_Switchboard_test.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_light_dev.jsonlines',
    #     # 'arddnaacl_gold_stat_itself_dep10_light_test.jsonlines',
    #     # 'arddnaacl_pred_stat_itself_depLabel_AMI_dev.jsonlines',
    #     'arddnaacl_pred_stat_itself_depLabel_AMI_test.jsonlines',
    #     # 'arddnaacl_pred_stat_itself_depLabel_Persuasion_dev.jsonlines',
    #     'arddnaacl_pred_stat_itself_depLabel_Persuasion_test.jsonlines',
    #     # 'arddnaacl_pred_stat_itself_depLabel_Switchboard_dev.jsonlines',
    #     'arddnaacl_pred_stat_itself_depLabel_Switchboard_test.jsonlines',
    #     # 'arddnaacl_pred_stat_itself_depLabel_light_dev.jsonlines',
    #     'arddnaacl_pred_stat_itself_depLabel_light_test.jsonlines',
    # ]

    # test_files = [
    #     'AMI_test_no_men.2022_content.jsonlines',
    #     'Persuasion_test_no_men.2022_content.jsonlines',
    #     'Switchboard_test_Phase_1.2022_content.jsonlines',
    #     'light_test_no_men.2022_content.jsonlines',
    # ]

    test_files = [
        # 'AMI_test_gold_men.2022_content.jsonlines',
        # 'Persuasion_test_gold_men.2022_content.jsonlines',
        # 'Switchboard_test_gold_men.2022_content.jsonlines',
        # 'light_test_gold_men.2022_content.jsonlines',
        'light_test_gold_ana.2022.removedExcessiveUtt.content.jsonlines',
        'Switchboard_test_gold_ana.2022.removedExcessiveUtt.content.jsonlines',
        'Persuasion_test_gold_ana.2022.removedExcessiveUtt.content.jsonlines',
        'AMI_test_gold_ana.2022.removedExcessiveUtt.content.jsonlines',
    ]

    # config for has null - predicted
    # configs = [
    #     # ('FullModel_pred_HasNull_800_ami', 'model_Jan12_18-57-27_3500.bin'),
    #     # ('FullModel_pred_HasNull_800_light', 'model_Jan12_20-33-04_3000.bin'),
    #     # ('FullModel_pred_HasNull_800_pers', 'model_Jan12_22-05-14_1500.bin'),
    #     # ('FullModel_pred_HasNull_800_swbd', 'model_Jan12_23-37-40_8500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_ami', 'model_Jan12_18-57-26_3500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_light', 'model_Jan12_20-33-56_6500.bin'),
    #     ('FullModel_pred_HasNull_0dot5_pers', 'model_Jan12_22-06-44_3000.bin'),
    #     ('FullModel_pred_HasNull_0dot5_swbd', 'model_Jan12_23-38-20_10500.bin'),
    # ]
    # config for has null - gold
    # configs = [
    #     ('FullModel_gold_HasNull_800_ami', 'model_Jan13_05-43-27_7000.bin'),
    #     ('FullModel_gold_HasNull_0dot5_light', 'model_Jan13_07-16-19_3000.bin'),
    #     ('FullModel_gold_HasNull_800_pers', 'model_Jan13_08-39-08_8500.bin'),
    #     ('FullModel_gold_HasNull_0dot5_swbd', 'model_Jan13_10-00-13_9500.bin'),
    # ]
    # # config for no null - predicted
    # configs = [
    #     ('FullModel_pred_NoNull_0dot5_pers', 'model_Jan13_18-20-40_6500.bin'),
    #     ('FullModel_pred_NoNull_0dot5_ami', 'model_Jan13_19-58-16_4000.bin'),
    #     ('FullModel_pred_NoNull_0dot5_swbd', 'model_Jan13_19-52-36_9000.bin'),
    #     ('FullModel_pred_NoNull_0dot5_light', 'model_Jan13_21-42-35_6000.bin'),
    #     # ('FullModel_pred_NoNull_800_pers', 'model_Jan13_21-29-27_8500.bin'),
    #     # ('FullModel_pred_NoNull_800_swbd', 'model_Jan13_23-01-09_7500.bin'),
    #     # ('FullModel_pred_NoNull_800_ami', 'model_Jan13_23-15-11_6500.bin'),
    #     # ('FullModel_pred_NoNull_800_light', 'model_Jan14_01-01-39_4500.bin'),
    # ]
    # config for no null - gold
    # configs = [
    #     ('FullModel_gold_NoNull_0dot5_ami', 'model_Jan14_02-55-38_3500.bin'),
    #     ('FullModel_gold_NoNull_0dot5_light', 'model_Jan14_04-27-40_7500.bin'),
    #     ('FullModel_gold_NoNull_0dot5_pers', 'model_Jan14_05-50-17_3000.bin'),
    #     ('FullModel_gold_NoNull_0dot5_swbd', 'model_Jan14_06-55-33_7500.bin'),
    #     # ('FullModel_gold_NoNull_800_ami', 'model_Jan14_01-10-54_5500.bin'),
    #     # ('FullModel_gold_NoNull_800_light', 'model_Jan14_02-43-55_6000.bin'),
    #     # ('FullModel_gold_NoNull_800_pers', 'model_Jan14_04-06-29_2500.bin'),
    #     # ('FullModel_gold_NoNull_800_swbd', 'model_Jan14_05-28-33_12000.bin'),
    # ]
    # config for baseline - gold
    # configs = [
    #     ('baselineGold_ami', 'model_Jan12_05-46-23_5500.bin'),
    #     ('baselineGold_pers', 'model_Jan12_05-46-26_7500.bin'),
    #     ('baselineGold_light', 'model_Jan12_07-26-21_3000.bin'),
    #     ('baselineGold_swbd', 'model_Jan12_07-28-14_6000.bin'),
    # ]
    # config for baseline w/ anaphor filtering - gold
    # configs = [
    #     ('baselineGold_AnaphorInTrainingSet_pers', 'model_Jan12_12-32-28_1500.bin'),
    #     ('baselineGold_AnaphorInTrainingSet_ami', 'model_Jan12_12-29-15_10500.bin'),
    #     ('baselineGold_AnaphorInTrainingSet_light', 'model_Jan14_21-16-50_7000.bin'),
    #     ('baselineGold_AnaphorInTrainingSet_swbd', 'model_Jan14_21-16-54_6500.bin'),
    # ]
    # config for ablation
    # configs = [
    #     # ('FullModel_gold_NoNullAblation_0dot5_ami', 'model_Jan14_11-14-05_5000.bin'),
    #     # ('FullModel_gold_NoNullAblation_0dot5_light', 'model_Jan14_12-09-03_11500.bin'),
    #     # ('FullModel_gold_NoNullAblation_0dot5_pers', 'model_Jan14_13-04-02_8000.bin'),
    #     # ('FullModel_gold_NoNullAblation_0dot5_swbd', 'model_Jan14_13-58-51_9000.bin'),
    #     ('FullModel_pred_NoNullAblation_0dot5_ami', 'model_Jan14_11-14-09_10500.bin'),
    #     # ('FullModel_pred_NoNullAblation_0dot5_light', 'model_Jan14_12-10-51_11500.bin'),
    #     # ('FullModel_pred_NoNullAblation_0dot5_pers', 'model_Jan14_13-07-36_5000.bin'),
    #     # ('FullModel_pred_NoNullAblation_0dot5_swbd', 'model_Jan14_14-04-29_6000.bin'),
    # ]
    # configs = [
    #     ('baseline_extend_together', 'model_Mar02_01-06-57_4000.bin', 11),
    #     ('baseline_extend_together', 'model_Mar02_01-06-57_combine.bin', 11),
    # ]
    configs = [
        # ('baseline_extend_Utterance10_ami', 'model_Mar02_11-17-59_combine.bin', 11),
        # ('baseline_extend_Utterance10_light', 'model_Mar02_12-26-39_combine.bin', 11),
        # ('baseline_extend_Utterance10_pers', 'model_Mar02_04-41-54_combine.bin', 11),
        # ('baseline_extend_Utterance10_swbd', 'model_Mar02_05-51-41_combine.bin', 11),
        # ('baseline_extend_Utterance50_ami', 'model_Mar02_04-41-46_combine.bin', 11),
        # ('baseline_extend_Utterance50_light', 'model_Mar02_06-19-39_combine.bin', 11),
        # ('baseline_extend_Utterance50_pers', 'model_Mar02_07-59-03_combine.bin', 11),
        # ('baseline_extend_Utterance50_swbd', 'model_Mar02_09-38-49_combine.bin', 11),
        # ('baseline_extend_UtteranceClosest10_ami', 'model_Mar02_07-01-00_combine.bin', 11),
        # ('baseline_extend_UtteranceClosest10_light', 'model_Mar02_08-09-28_combine.bin', 11),
        # ('baseline_extend_UtteranceClosest10_pers', 'model_Mar02_09-19-07_combine.bin', 11),
        # ('baseline_extend_UtteranceClosest10_swbd', 'model_Mar02_10-29-09_combine.bin', 11),
        # ('baseline_Utterance10_ami', 'model_Mar03_03-46-50_2000.bin', 11),
        # ('baseline_Utterance10_light', 'model_Mar03_03-58-12_2500.bin', 11),
        # ('baseline_Utterance10_pers', 'model_Mar02_22-10-58_6000.bin', 11),
        # ('baseline_Utterance10_swbd', 'model_Mar02_22-44-44_8500.bin', 11),
        # ('baseline_Utterance50_ami', 'model_Mar02_23-12-35_5500.bin', 11),
        # ('baseline_Utterance50_light', 'model_Mar02_23-54-47_6500.bin', 11),
        # ('baseline_Utterance50_pers', 'model_Mar03_00-46-38_11500.bin', 11),
        # ('baseline_Utterance50_swbd', 'model_Mar03_02-18-09_11500.bin', 11),
        # ('baseline_UtteranceClosest10_ami', 'model_Mar02_23-30-42_2000.bin', 11),
        # ('baseline_UtteranceClosest10_light', 'model_Mar02_23-42-01_7000.bin', 11),
        # ('baseline_UtteranceClosest10_pers', 'model_Mar03_00-20-57_8500.bin', 11),
        # ('baseline_UtteranceClosest10_swbd', 'model_Mar03_01-08-23_9500.bin', 11),
        #
        # ('baseline_Utterance10_pers', 'model_Mar03_05-42-47_5000.bin', 11),
        # ('baseline_Utterance10_pers', 'model_Mar03_05-42-47_combine.bin', 11),
        # ('baseline_Utterance10_swbd', 'model_Mar03_06-52-33_7500.bin', 11),
        # ('baseline_Utterance10_swbd', 'model_Mar03_06-52-33_combine.bin', 11),
        # ('baseline_Utterance50_ami', 'model_Mar02_23-12-35_5500.bin', 11),
        # ('baseline_Utterance50_ami', 'model_Mar03_05-42-48_5500.bin', 11),
        # ('baseline_Utterance50_ami', 'model_Mar03_05-42-48_combine.bin', 11),
        # ('baseline_Utterance50_light', 'model_Mar03_07-21-10_2500.bin', 11),
        # ('baseline_Utterance50_light', 'model_Mar03_07-21-10_combine.bin', 11),
        # ('baseline_Utterance50_pers', 'model_Mar03_09-00-54_8500.bin', 11),
        # ('baseline_Utterance50_pers', 'model_Mar03_09-00-54_combine.bin', 11),
        # ('baseline_Utterance50_swbd', 'model_Mar03_10-40-53_6500.bin', 11),
        # ('baseline_Utterance50_swbd', 'model_Mar03_10-40-53_combine.bin', 11),
        # ('baseline_UtteranceClosest10_ami', 'model_Mar03_08-02-20_2500.bin', 11),
        # ('baseline_UtteranceClosest10_ami', 'model_Mar03_08-02-20_combine.bin', 11),
        # ('baseline_UtteranceClosest10_light', 'model_Mar03_09-11-14_2000.bin', 11),
        # ('baseline_UtteranceClosest10_light', 'model_Mar03_09-11-14_combine.bin', 11),
        # ('baseline_UtteranceClosest10_pers', 'model_Mar03_10-21-11_7000.bin', 11),
        # ('baseline_UtteranceClosest10_pers', 'model_Mar03_10-21-11_combine.bin', 11),
        # ('baseline_UtteranceClosest10_swbd', 'model_Mar03_01-08-23_9500.bin', 11),
        # ('baseline_UtteranceClosest10_swbd', 'model_Mar03_11-31-04_9500.bin', 11),
        # ('baseline_UtteranceClosest10_swbd', 'model_Mar03_11-31-04_combine.bin', 11),
        # ('baseline_Utterance10_ami', 'model_Mar03_12-20-22_3000.bin', 11),
        # ('baseline_Utterance10_ami', 'model_Mar03_12-20-22_combine.bin', 11),
        # ('baseline_Utterance10_light', 'model_Mar03_13-29-51_3000.bin', 11),
        # ('baseline_Utterance10_light', 'model_Mar03_13-29-51_combine.bin', 11),
        # ('baseline_Utterance5_ami', 'model_Mar03_14-08-23_6500.bin', 11),
        # ('baseline_Utterance5_ami', 'model_Mar03_14-08-23_combine.bin', 11),
        # ('baseline_UtteranceClosest5_ami', 'model_Mar03_14-40-45_3500.bin', 11),
        # ('baseline_UtteranceClosest5_ami', 'model_Mar03_14-40-45_combine.bin', 11),
        # ('baseline_Utterance5_light', 'model_Mar03_19-46-47_500.bin', 11),
        # ('baseline_Utterance5_light', 'model_Mar03_19-51-18_6000.bin', 11),
        # ('baseline_Utterance5_light', 'model_Mar03_19-51-18_combine.bin', 11),
        # ('baseline_Utterance5_pers', 'model_Mar03_20-57-36_11000.bin', 11),
        # ('baseline_Utterance5_pers', 'model_Mar03_20-57-36_combine.bin', 11),
        # ('baseline_Utterance5_swbd', 'model_Mar03_22-04-16_10500.bin', 11),
        # ('baseline_Utterance5_swbd', 'model_Mar03_22-04-16_combine.bin', 11),
        # ('baseline_UtteranceClosest5_light', 'model_Mar03_19-45-39_4000.bin', 11),
        # ('baseline_UtteranceClosest5_light', 'model_Mar03_19-45-39_combine.bin', 11),
        # ('baseline_UtteranceClosest5_pers', 'model_Mar03_20-52-29_8500.bin', 11),
        # ('baseline_UtteranceClosest5_pers', 'model_Mar03_20-52-29_combine.bin', 11),
        # ('baseline_UtteranceClosest5_swbd', 'model_Mar03_21-59-21_11500.bin', 11),
        # ('baseline_UtteranceClosest5_swbd', 'model_Mar03_21-59-21_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance50_ami', 'model_Mar04_03-28-13_3500.bin', 11),
        # ('baseline_SentDistFeatUtterance50_ami', 'model_Mar04_03-28-13_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance50_light', 'model_Mar04_04-59-39_2000.bin', 11),
        # ('baseline_SentDistFeatUtterance50_light', 'model_Mar04_04-59-39_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance50_pers', 'model_Mar04_03-28-12_10500.bin', 11),
        # ('baseline_SentDistFeatUtterance50_pers', 'model_Mar04_03-28-12_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance50_swbd', 'model_Mar04_05-00-42_7000.bin', 11),
        # ('baseline_SentDistFeatUtterance50_swbd', 'model_Mar04_05-00-42_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance5_ami', 'model_Mar04_09-36-12_11000.bin', 11),
        # ('baseline_SentDistFeatUtterance5_ami', 'model_Mar04_09-36-12_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance5_light', 'model_Mar04_10-43-07_7500.bin', 11),
        # ('baseline_SentDistFeatUtterance5_light', 'model_Mar04_10-43-07_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance5_pers', 'model_Mar04_09-37-41_11000.bin', 11),
        # ('baseline_SentDistFeatUtterance5_pers', 'model_Mar04_09-37-41_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance5_swbd', 'model_Mar04_10-45-34_11500.bin', 11),
        # ('baseline_SentDistFeatUtterance5_swbd', 'model_Mar04_10-45-34_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance10_ami', 'model_Mar04_16-49-57_4000.bin', 11),
        # ('baseline_SentDistFeatUtterance10_ami', 'model_Mar04_16-49-57_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance10_light', 'model_Mar04_17-59-42_4500.bin', 11),
        # ('baseline_SentDistFeatUtterance10_light', 'model_Mar04_17-59-42_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance10_pers', 'model_Mar04_16-49-55_12000.bin', 11),
        # ('baseline_SentDistFeatUtterance10_pers', 'model_Mar04_16-49-55_combine.bin', 11),
        # ('baseline_SentDistFeatUtterance10_swbd', 'model_Mar04_18-00-19_1500.bin', 11),
        # ('baseline_SentDistFeatUtterance10_swbd', 'model_Mar04_18-00-19_combine.bin', 11),
        #
        # ('baseline_SentDistFeatUtteranceClosest10_ami', 'model_Mar04_21-02-14_2000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_ami', 'model_Mar04_21-02-14_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_light', 'model_Mar04_22-10-57_4000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_light', 'model_Mar04_22-10-57_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_pers', 'model_Mar04_21-02-16_4500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_pers', 'model_Mar04_21-02-16_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_swbd', 'model_Mar04_22-12-17_12000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10_swbd', 'model_Mar04_22-12-17_combine.bin', 11),
        #
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_ami', 'model_Mar05_01-45-00_9500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_ami', 'model_Mar05_01-45-00_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_light', 'model_Mar05_02-41-48_5500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_light', 'model_Mar05_02-41-48_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_pers', 'model_Mar05_01-45-00_4000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_pers', 'model_Mar05_01-45-00_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_swbd', 'model_Mar05_02-41-13_6000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_swbd', 'model_Mar05_02-41-13_combine.bin', 11),
        #
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_ami', 'model_Mar05_05-12-09_6000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_ami', 'model_Mar05_05-12-09_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_light', 'model_Mar05_05-12-10_6000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_light', 'model_Mar05_05-12-10_combine.bin',
        #  11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_pers', 'model_Mar05_06-45-09_9500.bin', 11),
        # (
        # 'baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_pers', 'model_Mar05_06-45-09_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_swbd', 'model_Mar05_06-45-10_5500.bin', 11),
        # (
        # 'baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl0dot5_swbd', 'model_Mar05_06-45-10_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_ami', 'model_Mar05_10-55-18_8000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_ami', 'model_Mar05_10-55-18_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_light', 'model_Mar05_11-52-08_3000.bin', 11),
        # (
        # 'baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_light', 'model_Mar05_11-52-08_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_pers', 'model_Mar05_10-56-09_9000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_pers', 'model_Mar05_10-56-09_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_swbd', 'model_Mar05_11-53-54_6500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl1600_swbd', 'model_Mar05_11-53-54_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_ami', 'model_Mar05_09-02-25_4500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_ami', 'model_Mar05_09-02-25_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_light', 'model_Mar05_09-58-58_2500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_light', 'model_Mar05_09-58-58_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_pers', 'model_Mar05_09-02-26_11500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_pers', 'model_Mar05_09-02-26_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_swbd', 'model_Mar05_09-59-11_11000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelection_tl800_swbd', 'model_Mar05_09-59-11_combine.bin', 11),
        #
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_ami',
        #  'model_Mar05_12-48-28_8000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_ami',
        #  'model_Mar06_19-28-56_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_light',
        #  'model_Mar05_18-46-18_2000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_light',
        #  'model_Mar05_18-46-18_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_pers',
        #  'model_Mar05_12-51-43_5500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_pers',
        #  'model_Mar06_19-32-54_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_swbd',
        #  'model_Mar05_21-50-46_12840.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_swbd',
        #  'model_Mar05_21-50-46_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_ami',
        #  'model_Mar06_00-52-11_5000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_ami',
        #  'model_Mar06_00-52-11_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_light',
        #  'model_Mar06_01-52-44_6000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_light',
        #  'model_Mar06_01-52-44_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_pers',
        #  'model_Mar06_02-53-03_8000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_pers',
        #  'model_Mar06_02-53-03_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_swbd',
        #  'model_Mar06_03-53-22_7500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl1600_swbd',
        #  'model_Mar06_03-53-22_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_ami',
        #  'model_Mar05_19-47-25_3500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_ami',
        #  'model_Mar05_19-47-25_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_light',
        #  'model_Mar05_20-48-04_6500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_light',
        #  'model_Mar05_20-48-04_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_pers',
        #  'model_Mar05_22-52-32_6500.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_pers',
        #  'model_Mar05_22-52-32_combine.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_swbd',
        #  'model_Mar05_23-52-14_9000.bin', 11),
        # ('baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl800_swbd',
        #  'model_Mar05_23-52-14_combine.bin', 11),
        #
        # ('baseline_NoNull_tl0dot5_ami', 'model_Mar06_07-49-05_7500.bin', 11),
        # ('baseline_NoNull_tl0dot5_ami', 'model_Mar06_07-49-05_combine.bin', 11),
        # ('baseline_NoNull_tl0dot5_light', 'model_Mar06_08-49-41_8500.bin', 11),
        # ('baseline_NoNull_tl0dot5_light', 'model_Mar06_08-49-41_combine.bin', 11),
        # ('baseline_NoNull_tl0dot5_pers', 'model_Mar06_09-49-29_10000.bin', 11),
        # ('baseline_NoNull_tl0dot5_pers', 'model_Mar06_09-49-29_combine.bin', 11),
        # ('baseline_NoNull_tl0dot5_swbd', 'model_Mar06_10-49-42_4500.bin', 11),
        # ('baseline_NoNull_tl0dot5_swbd', 'model_Mar06_10-49-42_combine.bin', 11),
        # ('baseline_NoNull_tl1600_ami', 'model_Mar06_05-48-40_11000.bin', 11),
        # ('baseline_NoNull_tl1600_ami', 'model_Mar06_05-48-40_combine.bin', 11),
        # ('baseline_NoNull_tl1600_light', 'model_Mar06_06-48-51_11000.bin', 11),
        # ('baseline_NoNull_tl1600_light', 'model_Mar06_06-48-51_combine.bin', 11),
        # ('baseline_NoNull_tl1600_pers', 'model_Mar06_07-48-34_4000.bin', 11),
        # ('baseline_NoNull_tl1600_pers', 'model_Mar06_07-48-34_combine.bin', 11),
        # ('baseline_NoNull_tl1600_swbd', 'model_Mar06_08-48-04_11500.bin', 11),
        # ('baseline_NoNull_tl1600_swbd', 'model_Mar06_08-48-04_combine.bin', 11),
        # ('baseline_NoNull_tl800_ami', 'model_Mar06_09-48-16_7000.bin', 11),
        # ('baseline_NoNull_tl800_ami', 'model_Mar06_09-48-16_combine.bin', 11),
        # ('baseline_NoNull_tl800_light', 'model_Mar06_10-48-29_10000.bin', 11),
        # ('baseline_NoNull_tl800_light', 'model_Mar06_10-48-29_combine.bin', 11),
        # ('baseline_NoNull_tl800_pers', 'model_Mar06_05-48-42_8500.bin', 11),
        # ('baseline_NoNull_tl800_pers', 'model_Mar06_05-48-42_combine.bin', 11),
        # ('baseline_NoNull_tl800_swbd', 'model_Mar06_06-48-44_8000.bin', 11),
        # ('baseline_NoNull_tl800_swbd', 'model_Mar06_06-48-44_combine.bin', 11),
        #
        # ('baseline_typePrediction5subtokens_tl0dot5_ami', 'model_Mar06_22-20-09_3000.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_ami', 'model_Mar06_22-20-09_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_light', 'model_Mar06_23-19-12_7500.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_light', 'model_Mar06_23-19-12_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_pers', 'model_Mar07_00-17-42_6000.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_pers', 'model_Mar07_00-17-42_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_swbd', 'model_Mar07_01-16-12_6500.bin', 11),
        # ('baseline_typePrediction5subtokens_tl0dot5_swbd', 'model_Mar07_01-16-12_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_ami', 'model_Mar06_20-22-31_6500.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_ami', 'model_Mar06_20-22-31_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_light', 'model_Mar06_21-22-01_11500.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_light', 'model_Mar06_21-22-01_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_pers', 'model_Mar06_22-20-47_6500.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_pers', 'model_Mar06_22-20-47_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_swbd', 'model_Mar06_23-19-50_8000.bin', 11),
        # ('baseline_typePrediction5subtokens_tl1600_swbd', 'model_Mar06_23-19-50_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_ami', 'model_Mar07_00-18-45_9000.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_ami', 'model_Mar07_00-18-45_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_light', 'model_Mar07_01-18-03_6500.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_light', 'model_Mar07_01-18-03_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_pers', 'model_Mar06_20-22-30_8000.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_pers', 'model_Mar06_20-22-30_combine.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_swbd', 'model_Mar06_21-21-17_12840.bin', 11),
        # ('baseline_typePrediction5subtokens_tl800_swbd', 'model_Mar06_21-21-17_combine.bin', 11),
        #
        # ('baseline_typePredictionNoNull_tl0dot5_ami', 'model_Mar07_03-12-38_12000.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_ami', 'model_Mar07_03-12-38_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_light', 'model_Mar07_04-10-01_4500.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_light', 'model_Mar07_04-10-01_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_pers', 'model_Mar07_03-12-37_2000.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_pers', 'model_Mar07_03-12-37_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_swbd', 'model_Mar07_04-08-58_7000.bin', 11),
        # ('baseline_typePredictionNoNull_tl0dot5_swbd', 'model_Mar07_04-08-58_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_ami', 'model_Mar07_07-00-39_6500.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_ami', 'model_Mar07_07-00-39_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_light', 'model_Mar07_07-57-53_11500.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_light', 'model_Mar07_07-57-53_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_pers', 'model_Mar07_06-58-22_2500.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_pers', 'model_Mar07_06-58-22_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_swbd', 'model_Mar07_07-54-56_11500.bin', 11),
        # ('baseline_typePredictionNoNull_tl1600_swbd', 'model_Mar07_07-54-56_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_ami', 'model_Mar07_05-06-38_8000.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_ami', 'model_Mar07_05-06-38_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_light', 'model_Mar07_06-04-07_6500.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_light', 'model_Mar07_06-04-07_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_pers', 'model_Mar07_05-05-33_6500.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_pers', 'model_Mar07_05-05-33_combine.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_swbd', 'model_Mar07_06-01-53_3000.bin', 11),
        # ('baseline_typePredictionNoNull_tl800_swbd', 'model_Mar07_06-01-53_combine.bin', 11),
        #
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_ami', 'model_Mar07_16-04-47_4000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_ami', 'model_Mar07_16-04-47_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_light', 'model_Mar07_17-03-46_8500.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_light', 'model_Mar07_17-03-46_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_pers', 'model_Mar07_16-06-00_12500.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_pers', 'model_Mar07_16-06-00_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_swbd', 'model_Mar07_17-04-12_10000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl0dot5_swbd', 'model_Mar07_17-04-12_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_ami', 'model_Mar07_12-15-04_5000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_ami', 'model_Mar07_12-15-04_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_light', 'model_Mar07_13-12-47_4500.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_light', 'model_Mar07_13-12-47_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_pers', 'model_Mar07_12-15-04_11500.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_pers', 'model_Mar07_12-15-04_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_swbd', 'model_Mar07_13-12-47_9000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl1600_swbd', 'model_Mar07_13-12-47_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_ami', 'model_Mar07_14-10-00_3000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_ami', 'model_Mar07_14-10-00_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_light', 'model_Mar07_15-07-30_5000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_light', 'model_Mar07_15-07-30_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_pers', 'model_Mar07_14-10-26_3000.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_pers', 'model_Mar07_14-10-26_combine.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_swbd', 'model_Mar07_15-08-03_10500.bin', 11),
        # ('baseline_typePredictionNoNullAnaphorFeat_tl800_swbd', 'model_Mar07_15-08-03_combine.bin', 11),
        #
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_ami', 'model_Mar07_18-02-42_3000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_ami', 'model_Mar07_18-02-42_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_light', 'model_Mar07_19-38-54_8000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_light', 'model_Mar07_19-38-54_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_pers', 'model_Mar07_18-02-49_12500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_pers', 'model_Mar07_18-02-49_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_swbd', 'model_Mar07_19-27-58_10000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl0dot5_swbd', 'model_Mar07_19-27-58_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_ami', 'model_Mar08_00-07-43_8000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_ami', 'model_Mar08_00-07-43_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_light', 'model_Mar08_01-43-22_11500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_light', 'model_Mar08_01-43-22_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_pers', 'model_Mar07_23-47-04_6500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_pers', 'model_Mar07_23-47-04_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_swbd', 'model_Mar08_01-13-32_11500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1600_swbd', 'model_Mar08_01-13-32_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_ami', 'model_Mar07_21-06-29_7000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_ami', 'model_Mar07_21-06-29_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_light', 'model_Mar07_22-42-04_3500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_light', 'model_Mar07_22-42-04_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_pers', 'model_Mar07_20-53-09_7000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_pers', 'model_Mar07_20-53-09_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_swbd', 'model_Mar07_22-20-17_10500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl800_swbd', 'model_Mar07_22-20-17_combine.bin', 11),
        #
        # ('baseline_typePredictionNoNullAllFeat_tl1200_ami', 'model_Mar08_10-06-03_5000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_ami', 'model_Mar08_10-06-03_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_light', 'model_Mar08_11-42-01_12000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_light', 'model_Mar08_11-42-01_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_pers', 'model_Mar08_04-12-55_8500.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_pers', 'model_Mar08_04-12-55_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_swbd', 'model_Mar08_05-38-59_11000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl1200_swbd', 'model_Mar08_05-38-59_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_ami', 'model_Mar08_04-12-54_3000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_ami', 'model_Mar08_04-12-54_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_light', 'model_Mar08_05-50-03_10000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_light', 'model_Mar08_05-50-03_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_pers', 'model_Mar08_07-15-06_6000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_pers', 'model_Mar08_07-15-06_combine.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_swbd', 'model_Mar08_08-40-16_11000.bin', 11),
        # ('baseline_typePredictionNoNullAllFeat_tl400_swbd', 'model_Mar08_08-40-16_combine.bin', 11),
        #
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_ami', 'model_Mar08_13-09-30_5000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_ami', 'model_Mar08_13-09-30_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_light', 'model_Mar08_14-07-30_7000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_light', 'model_Mar08_14-07-30_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_pers', 'model_Mar08_15-04-47_7000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_pers', 'model_Mar08_15-04-47_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_swbd', 'model_Mar08_16-02-20_9500.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl0dot5_swbd', 'model_Mar08_16-02-20_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_ami', 'model_Mar08_09-03-06_9000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_ami', 'model_Mar08_09-03-06_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_light', 'model_Mar08_10-01-33_5500.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_light', 'model_Mar08_10-01-33_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_pers', 'model_Mar08_10-59-11_6000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_pers', 'model_Mar08_10-59-11_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_swbd', 'model_Mar08_11-57-08_5500.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl1600_swbd', 'model_Mar08_11-57-08_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_ami', 'model_Mar08_16-59-51_5500.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_ami', 'model_Mar08_16-59-51_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_light', 'model_Mar08_17-58-19_3000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_light', 'model_Mar08_17-58-19_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_pers', 'model_Mar08_07-07-00_2000.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_pers', 'model_Mar08_07-07-00_combine.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_swbd', 'model_Mar08_08-04-55_11500.bin', 11),
        # ('baseline_typePredictionHasNullAnaphorFeat_tl800_swbd', 'model_Mar08_08-04-55_combine.bin', 11),
        #
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_ami', 'model_Mar10_17-00-34_6500.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_ami', 'model_Mar10_17-00-34_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_light', 'model_Mar10_18-35-42_3000.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_light', 'model_Mar10_18-35-42_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_pers', 'model_Mar10_17-01-17_9000.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_pers', 'model_Mar10_17-01-17_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_swbd', 'model_Mar10_18-26-35_1500.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl0dot5_swbd', 'model_Mar10_18-26-35_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_ami', 'model_Mar10_20-02-01_9500.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_ami', 'model_Mar10_20-02-01_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_pers', 'model_Mar10_19-52-20_7000.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_pers', 'model_Mar10_19-52-20_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_swbd', 'model_Mar10_21-19-22_9500.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_swbd', 'model_Mar10_21-19-22_combine.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_light', 'model_Mar10_21-36-59_7000.bin', 11),
        # ('marcharr_NoNullFullModelNoCons_tl800_light', 'model_Mar10_21-36-59_combine.bin', 11),
        #
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_ami', 'model_Mar11_03-24-59_4000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_ami', 'model_Mar11_03-24-59_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_light', 'model_Mar11_05-01-03_8500.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_light', 'model_Mar11_05-01-03_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_pers', 'model_Mar11_06-26-59_4500.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_pers', 'model_Mar11_06-26-59_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_swbd', 'model_Mar11_07-52-40_10000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c31_swbd', 'model_Mar11_07-52-40_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_ami', 'model_Mar11_09-24-15_2500.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_ami', 'model_Mar11_09-24-15_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_light', 'model_Mar11_11-01-40_5500.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_light', 'model_Mar11_11-01-40_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_pers', 'model_Mar11_12-27-26_6000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_pers', 'model_Mar11_12-27-26_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_swbd', 'model_Mar11_13-52-54_10000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c21_c35_swbd', 'model_Mar11_13-52-54_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_ami', 'model_Mar11_03-24-59_2000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_ami', 'model_Mar11_03-24-59_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_light', 'model_Mar11_05-02-19_3000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_light', 'model_Mar11_05-02-19_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_pers', 'model_Mar11_06-29-18_1500.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_pers', 'model_Mar11_06-29-18_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_swbd', 'model_Mar11_07-55-44_11000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c11_c25_c31_swbd', 'model_Mar11_07-55-44_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_ami', 'model_Mar11_09-19-26_2000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_ami', 'model_Mar11_09-19-26_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_light', 'model_Mar11_10-55-31_12000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_light', 'model_Mar11_10-55-31_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_pers', 'model_Mar11_12-21-39_2500.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_pers', 'model_Mar11_12-21-39_combine.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_swbd', 'model_Mar11_13-47-39_5000.bin', 11),
        # ('marcharr_HasNullFullModelScoringFunc_tl800_c15_c21_c31_swbd', 'model_Mar11_13-47-39_combine.bin', 11),
        #
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl0dot5_c11_c21_c310_c45_ami',
        #  'model_Mar11_17-44-37_7000.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl0dot5_c11_c21_c310_c45_ami',
        #  'model_Mar11_17-44-37_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl0dot5_c11_c21_c310_c45_pers',
        #  'model_Mar11_19-23-02_11000.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl0dot5_c11_c21_c310_c45_pers',
        #  'model_Mar11_19-23-02_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar11_17-45-38_7500.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar11_17-45-38_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar11_19-22-06_2000.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistHas4FeatScoringFunc_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar11_19-22-06_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar11_21-41-46_9500.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar11_21-41-46_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar11_21-41-47_9500.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar11_21-41-47_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c15_c21_c35_c45_pers',
        #  'model_Mar11_23-38-38_8000.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c15_c21_c35_c45_pers',
        #  'model_Mar11_23-38-38_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c15_c25_c35_c45_pers',
        #  'model_Mar11_23-38-37_2500.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c15_c25_c35_c45_pers',
        #  'model_Mar11_23-38-37_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c10_c20_c310_c40_pers',
        #  'model_Mar12_01-31-13_6000.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c10_c20_c310_c40_pers',
        #  'model_Mar12_01-31-13_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c11_c21_c310_c41_pers',
        #  'model_Mar12_01-31-08_2000.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c11_c21_c310_c41_pers',
        #  'model_Mar12_01-31-08_combine.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c10_c20_c3100_c40_pers',
        #  'model_Mar12_03-30-44_8500.bin', 11),
        # ('marcharr_HasNullFullModelHasSentDistNo4FeatScoringFunc_tl800_c10_c20_c3100_c40_pers',
        #  'model_Mar12_03-30-44_combine.bin', 11),
        # ablation
        # ('marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c310_c45_ami', 'model_Mar12_08-50-16_8500.bin', 11),
        # ('marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c310_c45_ami', 'model_Mar12_08-50-16_combine.bin', 11),
        # ('marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c310_c45_pers', 'model_Mar13_19-06-29_3500.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c310_c45_pers', 'model_Mar13_19-06-29_combine.bin', 11),
        # ('marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c35_c45_light', 'model_Mar13_20-40-30_4000.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c35_c45_light', 'model_Mar13_20-40-30_combine.bin', 11),
        # ('marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c35_c45_swbd', 'model_Mar13_17-33-01_10000.bin', 11),
        # ('marcharr_ScoringFuncAblation50Antecedent_tl800_c11_c21_c35_c45_swbd', 'model_Mar13_17-33-01_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c310_c45_ami', 'model_Mar12_20-27-36_5000.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c310_c45_ami', 'model_Mar12_20-27-36_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c310_c45_pers', 'model_Mar12_22-02-55_2000.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c310_c45_pers', 'model_Mar12_22-02-55_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c35_c45_light', 'model_Mar12_20-29-56_5500.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c35_c45_light', 'model_Mar12_20-29-56_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c35_c45_swbd', 'model_Mar12_21-54-59_9500.bin', 11),
        # ('marcharr_ScoringFuncAblationAnaphorFeat_tl800_c11_c21_c35_c45_swbd', 'model_Mar12_21-54-59_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c310_c45_ami', 'model_Mar12_18-28-48_11500.bin', 11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c310_c45_ami', 'model_Mar12_18-28-48_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c310_c45_pers', 'model_Mar12_19-27-44_1500.bin', 11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c310_c45_pers', 'model_Mar12_19-27-44_combine.bin',
        #  11),
        # (
        # 'marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c35_c45_light', 'model_Mar12_18-27-22_10000.bin', 11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c35_c45_light', 'model_Mar12_18-27-22_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c35_c45_swbd', 'model_Mar12_19-26-54_6500.bin', 11),
        # ('marcharr_ScoringFuncAblationAntecedentFeat_tl800_c11_c21_c35_c45_swbd', 'model_Mar12_19-26-54_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c310_c45_ami', 'model_Mar12_10-08-16_4000.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c310_c45_ami', 'model_Mar12_10-08-16_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c310_c45_pers', 'model_Mar13_13-10-23_8000.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c310_c45_pers', 'model_Mar13_13-10-23_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c35_c45_light', 'model_Mar12_11-44-49_6500.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c35_c45_light', 'model_Mar12_11-44-49_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c35_c45_swbd', 'model_Mar13_11-44-04_6000.bin', 11),
        # ('marcharr_ScoringFuncAblationC1_tl800_c10_c21_c35_c45_swbd', 'model_Mar13_11-44-04_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c310_c45_ami', 'model_Mar12_07-04-54_2000.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c310_c45_ami', 'model_Mar12_07-04-54_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c310_c45_pers', 'model_Mar13_14-37-08_8500.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c310_c45_pers', 'model_Mar13_14-37-08_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c35_c45_light', 'model_Mar12_08-42-38_9000.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c35_c45_light', 'model_Mar12_08-42-38_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c35_c45_swbd', 'model_Mar13_16-03-09_5000.bin', 11),
        # ('marcharr_ScoringFuncAblationC2_tl800_c11_c20_c35_c45_swbd', 'model_Mar13_16-03-09_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_ami', 'model_Mar12_05-09-44_5000.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_ami', 'model_Mar12_05-09-44_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_light', 'model_Mar12_06-06-12_3500.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_light', 'model_Mar12_06-06-12_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_pers', 'model_Mar13_17-29-07_10000.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_pers', 'model_Mar13_17-29-07_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_swbd', 'model_Mar13_18-55-15_9500.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_swbd', 'model_Mar13_18-55-15_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_ami', 'model_Mar13_11-42-06_11500.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_ami', 'model_Mar13_11-42-06_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_light', 'model_Mar13_13-16-37_3500.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_light', 'model_Mar13_13-16-37_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_pers', 'model_Mar13_14-42-04_4000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_pers', 'model_Mar13_14-42-04_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_swbd', 'model_Mar13_16-07-37_11000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC34_tl800_c11_c21_c30_c40_swbd', 'model_Mar13_16-07-37_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar13_08-02-30_5000.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar13_08-02-30_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar13_09-53-10_3500.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar13_09-53-10_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c35_c45_light',
        #  'model_Mar13_08-02-33_11500.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c35_c45_light',
        #  'model_Mar13_08-02-33_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c35_c45_swbd',
        #  'model_Mar13_09-52-28_11000.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAnaphorAntecedentWithin10_tl800_c11_c21_c35_c45_swbd',
        #  'model_Mar13_09-52-28_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c310_c45_ami', 'model_Mar13_05-25-49_7500.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c310_c45_ami', 'model_Mar13_05-25-49_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c310_c45_pers', 'model_Mar13_22-13-48_5500.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c310_c45_pers', 'model_Mar13_22-13-48_combine.bin',
        #  11),
        # (
        # 'marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c35_c45_light', 'model_Mar13_05-25-51_11000.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c35_c45_light', 'model_Mar13_05-25-51_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c35_c45_swbd', 'model_Mar13_06-44-03_10000.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedent_tl800_c11_c21_c35_c45_swbd', 'model_Mar13_06-44-03_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c310_c45_ami', 'model_Mar12_23-35-44_5000.bin', 11),
        # ('marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c310_c45_ami', 'model_Mar12_23-35-44_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c310_c45_pers', 'model_Mar13_01-10-10_1500.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c310_c45_pers', 'model_Mar13_01-10-10_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c35_c45_light', 'model_Mar12_23-35-45_12000.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c35_c45_light', 'model_Mar12_23-35-45_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c35_c45_swbd', 'model_Mar13_01-01-08_9000.bin', 11),
        # ('marcharr_ScoringFuncAblationSentDistFeat_tl800_c11_c21_c35_c45_swbd', 'model_Mar13_01-01-08_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_ami', 'model_Mar12_05-49-27_4000.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_ami', 'model_Mar12_05-49-27_combine.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_light', 'model_Mar12_07-25-48_11500.bin', 11),
        # ('marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_light', 'model_Mar12_07-25-48_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_pers', 'model_Mar13_20-21-26_4000.bin', 11),
        # ('marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_pers', 'model_Mar13_20-21-26_combine.bin',
        #  11),
        # ('marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_swbd', 'model_Mar13_21-47-41_11000.bin', 11),
        # ('marcharr_ScoringFuncAblationTypePrediction_tl800_c11_c21_c30_c40_swbd', 'model_Mar13_21-47-41_combine.bin',
        #  11),
        # new abaltion
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c35_c40_swbd', 'model_Mar14_04-44-31_6500.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c35_c40_swbd', 'model_Mar14_04-44-31_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_ami', 'model_Mar14_04-44-31_7500.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_ami', 'model_Mar14_04-44-31_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c35_c40_swbd', 'model_Mar14_06-23-40_6500.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c35_c40_swbd', 'model_Mar14_06-23-40_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_ami', 'model_Mar14_06-23-38_8500.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_ami', 'model_Mar14_06-23-38_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c35_c40_light', 'model_Mar14_07-49-31_8000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c35_c40_light', 'model_Mar14_07-49-31_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_light', 'model_Mar14_07-59-04_4500.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_light', 'model_Mar14_07-59-04_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_pers', 'model_Mar14_09-24-09_3000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_pers', 'model_Mar14_09-24-09_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c310_c40_ami', 'model_Mar14_09-14-58_2000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c310_c40_ami', 'model_Mar14_09-14-58_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_swbd', 'model_Mar14_10-50-02_12000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC3_tl800_c11_c21_c30_c45_swbd', 'model_Mar14_10-50-02_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c310_c40_pers', 'model_Mar14_10-51-32_7000.bin', 11),
        # ('marcharr_ScoringFuncAblationHardConsC4_tl800_c11_c21_c310_c40_pers', 'model_Mar14_10-51-32_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c35_c40_light', 'model_Mar14_12-37-04_6000.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c35_c40_light', 'model_Mar14_12-37-04_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_light', 'model_Mar14_12-37-03_4000.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_light', 'model_Mar14_12-37-03_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_pers', 'model_Mar14_14-02-43_3000.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_pers', 'model_Mar14_14-02-43_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c310_c40_ami', 'model_Mar14_14-02-11_4000.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c310_c40_ami', 'model_Mar14_14-02-11_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_swbd', 'model_Mar14_15-28-14_9500.bin', 11),
        # ('marcharr_ScoringFuncAblationC3_tl800_c11_c21_c30_c45_swbd', 'model_Mar14_15-28-14_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c310_c40_pers', 'model_Mar14_15-37-46_10000.bin', 11),
        # ('marcharr_ScoringFuncAblationC4_tl800_c11_c21_c310_c40_pers', 'model_Mar14_15-37-46_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_ami', 'model_Mar14_16-54-07_3000.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_ami', 'model_Mar14_16-54-07_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_light', 'model_Mar14_17-04-09_7500.bin', 11),
        # ('marcharr_ScoringFuncAblationC34_tl800_c11_c21_c30_c40_light', 'model_Mar14_17-04-09_combine.bin', 11),
        # ('baseline_rerun_ami', 'model_Mar14_18-30-21_6500.bin', 11),
        # ('baseline_rerun_ami', 'model_Mar14_18-30-21_combine.bin', 11),
        # ('baseline_rerun_pers', 'model_Mar14_18-29-02_8500.bin', 11),
        # ('baseline_rerun_pers', 'model_Mar14_18-29-02_combine.bin', 11),
        # ('baseline_rerun_swbd', 'model_Mar14_20-12-22_9500.bin', 11),
        # ('baseline_rerun_swbd', 'model_Mar14_20-12-22_combine.bin', 11),
        # ('baseline_rerun_light', 'model_Mar14_20-12-07_9500.bin', 11),
        # ('baseline_rerun_light', 'model_Mar14_20-12-07_combine.bin', 11),
        # rerun
        # ('baseline_Utterance50Width45_ami', 'model_Mar15_09-06-56_combine.bin', 11),
        # ('baseline_Utterance50Width45_pers', 'model_Mar15_09-06-53_combine.bin', 11),
        # ('baseline_Utterance50Width45_light', 'model_Mar15_10-48-59_combine.bin', 11),
        # ('baseline_Utterance50Width45_swbd', 'model_Mar15_10-50-23_combine.bin', 11),
        # ('marcharr_ScoringFuncAblation10antecedent_tl800_c11_c21_c35_c45_swbd', 'model_Mar15_12-33-26_combine.bin', 11),
        # ('marcharr_ScoringFuncAblation10antecedent_tl800_c11_c21_c310_c45_ami', 'model_Mar15_12-32-31_combine.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblation10antecedent_tl800_c11_c21_c310_c45_pers', 'model_Mar15_13-59-53_combine.bin', 11),
        # (
        # 'marcharr_ScoringFuncAblation10antecedent_tl800_c11_c21_c35_c45_light', 'model_Mar15_14-08-28_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedentAnaphor45_tl800_c11_c21_c310_c45_ami',
        #  'model_Mar15_19-18-04_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedentAnaphor45_tl800_c11_c21_c310_c45_pers',
        #  'model_Mar15_20-43-02_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedentAnaphor45_tl800_c11_c21_c35_c45_light',
        #  'model_Mar15_19-18-03_combine.bin', 11),
        # ('marcharr_ScoringFuncAblationOnlyAntecedentAnaphor45_tl800_c11_c21_c35_c45_swbd',
        #  'model_Mar15_20-43-55_combine.bin', 11),

        # ('FullModel_pred_NoNull_AllDatasets_0dot5', 'model_Jun25_10-57-16_10.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_0dot5', 'model_Jun25_10-57-16_15.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_0dot5', 'model_Jun25_10-57-16_20.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_0dot5', 'model_Jun25_10-57-16_25.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_0dot5', 'model_Jun25_10-57-16_5.0.bin', 11),
        #
        # ('FullModel_pred_NoNull_AllDatasets_extralight_0dot5', 'model_Jun26_13-28-07_10.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_extralight_0dot5', 'model_Jun26_13-28-07_15.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_extralight_0dot5', 'model_Jun26_13-28-07_20.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_extralight_0dot5', 'model_Jun26_13-28-07_25.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_extralight_0dot5', 'model_Jun26_13-28-07_30.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_extralight_0dot5', 'model_Jun26_13-28-07_5.0.bin', 11),
        #
        # # # gold m setting
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_0dot5', 'model_Jul21_02-34-10_30.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_100', 'model_Jul21_04-24-05_30.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_500', 'model_Jul21_06-11-05_30.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_800', 'model_Jul21_07-57-08_30.0.bin', 11),
        # # #
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_0dot5', 'model_Jul21_02-34-10_5.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_100', 'model_Jul21_04-24-05_5.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_500', 'model_Jul21_06-11-05_5.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_goldm_800', 'model_Jul21_07-57-08_5.0.bin', 11),
        # # #
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_0dot5', 'model_Jul21_02-34-10_10.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_100', 'model_Jul21_04-24-05_10.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_goldm_500', 'model_Jul21_06-11-05_10.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_800', 'model_Jul21_07-57-08_10.0.bin', 11),
        # #
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_0dot5', 'model_Jul21_02-34-10_15.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_100', 'model_Jul21_04-24-05_15.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_goldm_500', 'model_Jul21_06-11-05_15.0.bin', 11),
        # ('FullModel_pred_NoNull_AllDatasets_goldm_800', 'model_Jul21_07-57-08_15.0.bin', 11),
        # # #
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_0dot5', 'model_Jul21_02-34-10_20.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_100', 'model_Jul21_04-24-05_20.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_500', 'model_Jul21_06-11-05_20.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_800', 'model_Jul21_07-57-08_20.0.bin', 11),
        # # #
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_0dot5', 'model_Jul21_02-34-10_25.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_100', 'model_Jul21_04-24-05_25.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_500', 'model_Jul21_06-11-05_25.0.bin', 11),
        # # ('FullModel_pred_NoNull_AllDatasets_goldm_800', 'model_Jul21_07-57-08_25.0.bin', 11),
        # gold anaphor setting
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_1.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_10.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_11.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_12.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_13.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_14.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_16.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_17.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_19.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_2.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_21.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_3.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_4.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_6.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_7.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_8.0.bin', 11),
        # ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_9.0.bin', 11),
        ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_5.0.bin', 11),
        ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_10.0.bin', 11),
        ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_15.0.bin', 11),
        ('FullModel_golda_NoNull_AllDatasets', 'model_Aug01_18-40-56_20.0.bin', 11),

    ]

    for cfg in configs:
        if len(cfg) == 2:
            config_name, model_to_eval = cfg
            seed_to_eval = seed
        else:
            config_name, model_to_eval, seed_to_eval = cfg
        runner = Runner(config_name.rstrip('/'), gpu_id, seed=seed_to_eval,
                        best_model_suffix=model_to_eval.lstrip('model_').rstrip('.bin/'))
        for test_fn in test_files:
            # for _ in range(3, 8):
            # if ('ami' in config_name.lower() and 'ami' in test_fn.lower()) or \
            #         ('light' in config_name.lower() and 'light' in test_fn.lower()) or \
            #         ('swbd' in config_name.lower() and 'swbd' in test_fn.lower()) or \
            #         ('swbd' in config_name.lower() and 'switchboard' in test_fn.lower()) or \
            #         ('pers' in config_name.lower() and 'pers' in test_fn.lower()) or \
            #         ('together' in config_name.lower()):
                # if 'ami' in config_name.lower() and 'ami' in test_fn.lower():
                #     _ = 6
                # elif ('light' in config_name.lower() and 'light' in test_fn.lower()):
                #     _ = 0
                # elif ('swbd' in config_name.lower() and 'swbd' in test_fn.lower()) or \
                #     ('swbd' in config_name.lower() and 'switchboard' in test_fn.lower()):
                #     _ = 5
                # elif ('pers' in config_name.lower() and 'pers' in test_fn.lower()):
                #     _ = 3
            runner.config['prediction_only'] = True
            runner.config['inference_filter_antecedent_raw'] = 7
            # runner.config['use_dummy_antecedent'] = False
            # runner.config['do_type_prediction'] = False
            runner.set_test_file(test_fn)
            evaluator = Evaluator(runner)
            # evaluator.custom_name = ('AllFeat', f'HasNull_SentDist10')
            # evaluator.custom_name = ('AllFeat', f'AllFeatFilteringSentDist{_}')
            # evaluator.custom_name = ('Utterance10', f'Utterance10Seed113')
            # evaluator.custom_name = ('0dot5', f'0dot5_GoldMen_FilterSentenceDist7')
            evaluator.custom_name = ('goldm', f'goldm_FilterSentenceDist7')
            evaluator.conll_score()


def save_combine(gpu_id, seed_to_eval):
    configs = [
        'baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_ami',
        'baseline_SentDistFeatUtteranceClosest10AnaphorSelectionImprovedTypePred_tl0dot5_pers',
    ]

    for cfg in configs:

        runner = Runner(cfg.rstrip('/'), gpu_id, seed=seed_to_eval, best_model_suffix='combine')

        # device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        _, _, files = next(os.walk(runner.config['log_dir']))
        files = [fn for fn in files if 'model_' in fn and 'bin' in fn]
        path_ckpt = join(runner.config['log_dir'], files[0])
        state_dict = torch.load(path_ckpt, map_location=torch.device(runner.device))
        for fn in files[1:]:
            path_ckpt = join(runner.config['log_dir'], fn)
            new_state_dict = torch.load(path_ckpt, map_location=torch.device(runner.device))
            for layer in state_dict:
                state_dict[layer] += new_state_dict[layer]
        for layer in state_dict:
            state_dict[layer] /= len(files)
        model_to_evaluate = runner.initialize_model_from_state_dict(state_dict)
        runner.save_combined_model(model_to_evaluate, -1, remove=False)


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    else:
        seed = 11
    if config_name == 'batch':
        batch_evaluate(gpu_id, seed)
    elif config_name == 'combine_saved_checkpoints':
        save_combine(gpu_id, seed)
    else:
        runner = Runner(config_name, gpu_id, seed=seed)
        model = runner.initialize_model()

        if runner.config['training_phase']:
            runner.train(model)
            # del model

        # Evaluation

        # evaluator = Evaluator(runner)
        # evaluator.conll_score()
