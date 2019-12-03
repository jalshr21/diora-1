import argparse
import datetime
import math
import os
import random
import sys
import json
import types
import uuid

from collections import deque

import torch
import torch.nn as nn

from diora.data.dataset import ConsolidateDatasets, ReconstructDataset, make_batch_iterator

from diora.utils.path import package_path
from diora.logging.configuration import configure_experiment, get_logger
from diora.utils.flags import stringify_flags, init_with_flags_file, save_flags
from diora.utils.checkpoint import save_experiment

from diora.net.experiment_logger import ExperimentLogger

from diora.analysis.cky import ParsePredictor as CKY


data_types_choices = ('nli', 'conll_jsonl', 'txt', 'txt_id', 'synthetic')


def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])


def build_net(options, embeddings, batch_iterator=None):
    from diora.net.trainer import build_net

    trainer = build_net(options, embeddings, batch_iterator, random_seed=options.seed)

    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.net)))

    return trainer


def generate_seeds(n, seed=11):
    random.seed(seed)
    seeds = [random.randint(0, 2**16) for _ in range(n)]
    return seeds


def replace_leaves(tree, leaves):
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree += [newnode]

        return sofar, newtree

    _, newtree = func(tree)

    return newtree


def run_parse(options, train_iterator, trainer, validation_iterator):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)

    # Parse

    diora = trainer.net.diora

    ## Turn off outside pass.
    trainer.net.diora.outside = False

    ## Eval mode.
    trainer.net.eval()

    ## Topk predictor.
    parse_predictor = CKY(net=diora, word2idx=word2idx)

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    logger.info('Beginning to parse.')

    with torch.no_grad():
        for i, batch_map in enumerate(batches):
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Rather than skipping, just log the trees (they are trivially easy to find).
            if length <= 2:
                for i in range(batch_size):
                    example_id = batch_map['example_ids'][i]
                    tokens = sentences[i].tolist()
                    words = [idx2word[idx] for idx in tokens]
                    if length == 2:
                        o = dict(example_id=example_id, tree=(words[0], words[1]))
                    elif length == 1:
                        o = dict(example_id=example_id, tree=words[0])
                    print(json.dumps(o))
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees = parse_predictor.parse_batch(batch_map)

            for ii, tr in enumerate(trees):
                example_id = batch_map['example_ids'][ii]
                s = [idx2word[idx] for idx in sentences[ii].tolist()]
                tr = replace_leaves(tr, s)
                o = dict(example_id=example_id, tree=tr)

                print(json.dumps(o))

# Added now
def str_to_tuple(tr):
    tree = tr.split(" ")
    # print(tree)
    ans = []
    stack = []
    for i, s in enumerate(tree):
        if s != ')':
            stack.append(s)
        elif s == ')':
            inside = deque([])
            while(stack and stack[-1]!='('):
                inside.appendleft(stack.pop())
            if stack:
                stack.pop()
            new_tuple = tuple(inside)
            stack.append(new_tuple)
    return(stack)
# Added now


# Added now
def tree_to_spans(tr):
    # print(type(tr), tr)
    spans = []

    def helper(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1
        size = 0
        for x in tr:
            xsize = helper(x, pos + size)
            size += xsize
        spans.append((pos, size))
        return size
    helper(tr)
    return spans
# Added now


# Added now
def precision_and_recall(gold, not_gold):
    gold_span = set(gold)
    not_gold_span = set(not_gold)
    # assert len(gold_span) == len(not_gold_span)
    prec = 0
    recall = 0
    for span in not_gold_span:
        if span in gold_span:
            prec += 1
    for span in gold_span:
        if span in not_gold_span:
            recall += 1
    return prec, recall, len(gold_span)
# Added now


def run_train(options, train_iterator, trainer, validation_iterator):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    logger.info('Running train.')

    seeds = generate_seeds(options.max_epoch, options.seed)

    step = 0

    # Added now
    idx2word = {v: k for k, v in train_iterator.word2idx.items()}
    parse_predictor = CKY(net=trainer.net.diora, word2idx=train_iterator.word2idx)
    # Added now

    for epoch, seed in zip(range(options.max_epoch), seeds):
        # --- Train--- #

        # Added now
        precision = 0
        recall = 0
        total_len = 0
        count_des = 0
        # Added now

        seed = seeds[epoch]

        logger.info('epoch={} seed={}'.format(epoch, seed))

        def myiterator():
            it = train_iterator.get_iterator(random_seed=seed)

            count = 0

            for batch_map in it:
                # TODO: Skip short examples (optionally).
                if batch_map['length'] <= 2:
                    continue

                yield count, batch_map
                count += 1

        for batch_idx, batch_map in myiterator():
            if options.finetune and step >= options.finetune_after:
                trainer.freeze_diora()

            result = trainer.step(batch_map)

            # Added now
            trainer.net.eval()
            sentences = batch_map['sentences']
            trees = parse_predictor.parse_batch(batch_map)
            o_list = []
            for ii, tr in enumerate(trees):
                example_id = batch_map['example_ids'][ii]
                s = [idx2word[idx] for idx in sentences[ii].tolist()]
                tr = replace_leaves(tr, s)
                o = dict(example_id=example_id, tree=tr)
                o_list.append(o["tree"])
                # print(json.dumps(o))
                # print(o["tree"])
                # print(batch_map["parse_tree"][ii])
                if isinstance(batch_map["parse_tree"][ii], str):
                    parse_tree_tuple = str_to_tuple(batch_map["parse_tree"][ii])
                else:
                    parse_tree_tuple = batch_map["parse_tree"][ii]

                o_spans = tree_to_spans(o["tree"])
                batch_spans = tree_to_spans(parse_tree_tuple[0])


                p, r, t = precision_and_recall(batch_spans, o_spans)
                precision += p
                recall += r
                total_len += t

                # print(precision, recall, total_len)
                # print(precision / total_len, recall / total_len)
                # print((2*precision*recall)/(total_len*(precision+recall)))

            trainer.net.train()
            # Added now

            experiment_logger.record(result)

            if step % options.log_every_batch == 0:
                experiment_logger.log_batch(epoch, step, batch_idx, batch_size=options.batch_size)

            # -- Periodic Checkpoints -- #

            if not options.multigpu or options.local_rank == 0:
                if step % options.save_latest == 0 and step >= options.save_after:
                    logger.info('Saving model (periodic).')
                    trainer.save_model(os.path.join(options.experiment_path, 'model_periodic.pt'))
                    save_experiment(os.path.join(options.experiment_path, 'experiment_periodic.json'), step)

                if step % options.save_distinct == 0 and step >= options.save_after:
                    logger.info('Saving model (distinct).')
                    trainer.save_model(os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)))
                    save_experiment(os.path.join(options.experiment_path, 'experiment.step_{}.json'.format(step)), step)

            del result

            step += 1
        # Added now
        print(precision, recall, total_len)
        print(precision/total_len, recall/total_len)
        print(count_des)
        # Added now
        experiment_logger.log_epoch(epoch, step)

        if options.max_step is not None and step >= options.max_step:
            logger.info('Max-Step={} Quitting.'.format(options.max_step))
            sys.exit()


def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)


def get_train_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=True,
            include_partial=False, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size)


def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
            embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
            data_type=options.validation_data_type)


def get_validation_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=False,
            include_partial=True, filter_length=options.validation_filter_length,
            batch_size=options.validation_batch_size, length_to_size=options.length_to_size)


def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    # Modifies datasets. Unifying word mappings, embeddings, etc.
    ConsolidateDatasets([train_dataset, validation_dataset]).run()

    return train_dataset, validation_dataset


def run(options):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    train_dataset, validation_dataset = get_train_and_validation(options)
    train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    embeddings = train_dataset['embeddings']

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)
    logger.info('Model:')
    for name, p in trainer.net.named_parameters():
        logger.info('{} {}'.format(name, p.shape))

    if options.save_init:
        logger.info('Saving model (init).')
        trainer.save_model(os.path.join(options.experiment_path, 'model_init.pt'))

    if options.parse_only:
        run_parse(options, train_iterator, trainer, validation_iterator)
        sys.exit()

    run_train(options, train_iterator, trainer, validation_iterator)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=1000, type=int)
    parser.add_argument('--save_distinct', default=50000, type=int)
    parser.add_argument('--save_after', default=1000, type=int)
    parser.add_argument('--save_init', action='store_true')

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str)

    # Data (synthetic).
    parser.add_argument('--synthetic-nexamples', default=1000, type=int)
    parser.add_argument('--synthetic-vocabsize', default=1000, type=int)
    parser.add_argument('--synthetic-embeddingsize', default=1024, type=int)
    parser.add_argument('--synthetic-minlen', default=20, type=int)
    parser.add_argument('--synthetic-maxlen', default=21, type=int)
    parser.add_argument('--synthetic-seed', default=11, type=int)
    parser.add_argument('--synthetic-length', default=None, type=int)
    parser.add_argument('--use-synthetic-embeddings', action='store_true')

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=50, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Model.
    parser.add_argument('--arch', default='treelstm', choices=('treelstm', 'mlp', 'mlp-shared'))
    parser.add_argument('--hidden_dim', default=10, type=int)
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='margin', choices=('margin', 'softmax', 'semi'))

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'elmo', 'both'))

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=3, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default=None, type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Parsing.
    parser.add_argument('--parse_only', action='store_true')

    # Training.
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)

    # Optimization.
    parser.add_argument('--lr', default=4e-3, type=float)

    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.experiment_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%s")
        options.experiment_name = '{}-{}'.format(options.train_data_type, timestamp)

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    if options.length_to_size is not None:
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        flags_to_use = []
        flags_to_use += ['arch']
        flags_to_use += ['compress']
        flags_to_use += ['emb']
        flags_to_use += ['hidden_dim']
        flags_to_use += ['normalize']
        flags_to_use += ['reconstruct_mode']

        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        options = init_with_flags_file(options, options.flags)

    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)