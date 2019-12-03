import os
import sys
import traceback
import types

import torch
import torch.nn as nn
import torch.optim as optim

from diora.net.diora import DioraTreeLSTM
from diora.net.diora import DioraMLP
from diora.net.diora import DioraMLPShared

from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY

from diora.data.reading import tree_to_spans


class ReconstructionLoss(nn.Module):
    name = 'reconstruct_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        input_size = self.embeddings.weight.shape[1]
        size = diora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples)

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('ec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        lossfn = nn.MultiMarginLoss(margin=self.margin)
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        self.loss_hook(sentences, neg_samples, inputs)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_loss=loss)

        return loss, ret


class ReconstructionSoftmaxLoss(nn.Module):
    name = 'reconstruct_softmax_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionSoftmaxLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin
        self.input_size = input_size

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        input_size = self.input_size
        size = diora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples.unsqueeze(0))

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('zec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        lossfn = nn.CrossEntropyLoss()
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        self.loss_hook(sentences, neg_samples, inputs)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_softmax_loss=loss)

        return loss, ret


class SemiSupervisedParsingLoss(nn.Module):
    name = 'semi_supervised_parsing_loss'

    def __init__(self, margin=1, cuda=False, word2idx=None):
        super(SemiSupervisedParsingLoss, self).__init__()
        self.margin = margin
        self._cuda = cuda
        self.word2idx = word2idx
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def makeLeftTree(self, spans):
        result = []
        for s in spans:
            tmp = []
            if len(s) > 0:
                for t in s:
                    start = int(t[0])
                    l = t[1]
                    for i in range(2, l+1, 1):
                        tmp.append(tuple((start, i)))
            result.append(tmp[:])
        return result

    def makeRightTree(self, spans):
        result = []
        for s in spans:
            tmp = []
            if len(s) > 0:
                for t in s:
                    start = int(t[0])
                    l = t[1]
                    end = start + l - 1
                    for i in range(1, l, 1):
                        tmp.append(tuple((end - i, i+1)))
            result.append(tmp[:])
        return result

    def removeSentencesWithNoSpans(self, sentences, spans):
        result = []
        for i in range(len(spans)):
            if len(spans[i]) < 1:
                continue
            result.append(i)
        indices = torch.tensor(result)
        a = torch.index_select(sentences, 0, indices)
        return a

    def findParent(self, tree, span):
        children = []
        root_list = []
        if span in tree:
            return list(span), [span]
        else:
            mx = 1000
            root = None
            tmp = []
            for i, j in tree:
                if i<= int(span[0]) and (i+j-1) >= (int(span[0]) + span[1] - 1):
                    tmp.append((i, j))
            for i, j in tmp:
                if j < mx:
                    mx = j
                    root = [i, j]
            for i, j in tree:
                if i >= root[0] and i+j <= root[0]+root[1]:
                    children.append((i, j))
        return root, children

    def findClosestParent(self, max_spans, ner_spans):
        diora_spans = []
        roots = []
        for i in range(len(ner_spans)):
            s = ner_spans[i]
            tmp = []
            tmp_root = []
            if len(s) > 0:
                for t in s:
                    span = (int(t[0]), t[1])
                    root, parent = self.findParent(set(max_spans[i]), span)
                    # if parent == span:
                    #     sub_tree = self.makeLeftTree(span)
                    # else:
                    tmp += parent
                    tmp_root.append(root)
            diora_spans.append(tmp)
            roots.append(tmp_root)
        return roots, diora_spans

    def get_score_for_spans(self, sentences, scalars, spans):
        """
        Returns a list where each element is the score of the given tree.
        """
        batch_size = sentences.shape[0]
        length = sentences.shape[1]
        device = torch.cuda.current_device() if self._cuda else None
        span_sets = [set(span_lst) for span_lst in spans]

        # Chart.
        chart = [torch.full((length-i, batch_size), 1, dtype=torch.float, device=device) for i in range(length)]

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Book-keeping for given span.
                to_choose = [0] * batch_size
                to_choose_assert = [False] * batch_size

                # Assumes that the bottom-left most leaf is in the first constituent.
                spbatch = scalars[level][pos]

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    assert l_level >= 0
                    assert l_pos >= 0
                    assert r_level >= 0
                    assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

                    lp = chart[l_level][l_pos].view(-1, 1)
                    rp = chart[r_level][r_pos].view(-1, 1)
                    sp = spbatch[:, idx].view(-1, 1)

                    lps.append(lp)
                    rps.append(rp)
                    sps.append(sp)

                    pairs.append((l, r))

                    # Identifty the correct span.
                    l_size = l_level + 1
                    r_size = r_level + 1
                    l_span = (l_pos, l_size)
                    r_span = (r_pos, r_size)
                    for batch_idx in range(batch_size):
                        left_in_set = l_size == 1 or l_span in span_sets[batch_idx]
                        right_in_set = r_size == 1 or r_span in span_sets[batch_idx]
                        if left_in_set and right_in_set:
                            to_choose[batch_idx] = idx
                            assert to_choose_assert[batch_idx] is False, "Only one valid tree."
                            to_choose_assert[batch_idx] = True


                lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)

                ps = lps + rps + sps

                # Use the relevant spans.
                argmax1 = ps.argmax(1).long()
                not_to_choose_assert = [not i for i in to_choose_assert]
                argmax1 = argmax1 * torch.tensor(not_to_choose_assert, dtype=torch.long, device=device)
                argmax = torch.tensor(to_choose, dtype=torch.long, device=device)
                argmax2 = argmax1 + argmax

                valmax = ps[range(batch_size), argmax2]

                chart[level][pos, :] = valmax
        #print(chart)
        return chart[-1][0]

    def get_score_for_spans_modified(self, sentences, scalars, spans, root_list):
        """
        Returns a list where each element is the score of the given tree.
        """
        batch_size = sentences.shape[0]
        length = sentences.shape[1]
        device = torch.cuda.current_device() if self._cuda else None
        span_sets = [set(span_lst) for span_lst in spans]

        # Chart.
        chart = [torch.full((length-i, batch_size), 1, dtype=torch.float, device=device) for i in range(length)]

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Book-keeping for given span.
                to_choose = [0] * batch_size
                to_choose_assert = [False] * batch_size

                # Assumes that the bottom-left most leaf is in the first constituent.
                spbatch = scalars[level][pos]

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    assert l_level >= 0
                    assert l_pos >= 0
                    assert r_level >= 0
                    assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

                    lp = chart[l_level][l_pos].view(-1, 1)
                    rp = chart[r_level][r_pos].view(-1, 1)
                    sp = spbatch[:, idx].view(-1, 1)

                    lps.append(lp)
                    rps.append(rp)
                    sps.append(sp)

                    pairs.append((l, r))

                    # Identifty the correct span.
                    l_size = l_level + 1
                    r_size = r_level + 1
                    l_span = (l_pos, l_size)
                    r_span = (r_pos, r_size)
                    for batch_idx in range(batch_size):
                        left_in_set = l_size == 1 or l_span in span_sets[batch_idx]
                        right_in_set = r_size == 1 or r_span in span_sets[batch_idx]
                        if left_in_set and right_in_set:
                            to_choose[batch_idx] = idx
                            assert to_choose_assert[batch_idx] is False, "Only one valid tree."
                            to_choose_assert[batch_idx] = True
                    # print(span_sets)
                    # print(to_choose_assert)
                lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)

                ps = lps + rps + sps
                # Use the relevant spans.
                # argmax1 = ps.argmax(1).long() * ~to_choose_assert

                argmax = torch.tensor(to_choose, dtype=torch.long, device=device)
                # argmax2 = argmax * to_choose_assert

                # argmax2 = argmax2 + argmax1

                valmax = ps[range(batch_size), argmax]

                chart[level][pos, :] = valmax
        # print(chart)
        chart_roots_batch = []
        for bt, roots in enumerate(root_list):
            chart_roots = []
            for root in roots:
                    # print('root', root)
                    # print(len(chart),chart[0].shape, chart[-1].shape)
                    chart_roots.append(chart[int(root[1]-1)][int(root[0])][bt])
            chart_roots_batch.append(chart_roots)
        return chart_roots_batch


    def get_score_for_spans_given(self, sentences, scalars, spans, root_list):
        """
        Returns a list where each element is the score of the given tree.
        """
        batch_size = sentences.shape[0]
        length = sentences.shape[1]
        device = torch.cuda.current_device() if self._cuda else None
        span_sets = [set(span_lst) for span_lst in spans]

        # Chart.
        chart = [torch.full((length-i, batch_size), 1, dtype=torch.float, device=device) for i in range(length)]

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Book-keeping for given span.
                to_choose = [0] * batch_size
                to_choose_assert = [False] * batch_size

                # Assumes that the bottom-left most leaf is in the first constituent.
                spbatch = scalars[level][pos]

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    assert l_level >= 0
                    assert l_pos >= 0
                    assert r_level >= 0
                    assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

                    lp = chart[l_level][l_pos].view(-1, 1)
                    rp = chart[r_level][r_pos].view(-1, 1)
                    sp = spbatch[:, idx].view(-1, 1)

                    lps.append(lp)
                    rps.append(rp)
                    sps.append(sp)

                    pairs.append((l, r))

                    # Identifty the correct span.
                    l_size = l_level + 1
                    r_size = r_level + 1
                    l_span = (l_pos, l_size)
                    r_span = (r_pos, r_size)
                    for batch_idx in range(batch_size):
                        left_in_set = l_size == 1 or l_span in span_sets[batch_idx]
                        right_in_set = r_size == 1 or r_span in span_sets[batch_idx]
                        if left_in_set and right_in_set:
                            to_choose[batch_idx] = idx
                            assert to_choose_assert[batch_idx] is False, "Only one valid tree."
                            to_choose_assert[batch_idx] = True


                lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)

                ps = lps + rps + sps

                # Use the relevant spans.
                # argmax = ps.argmax(1).long()
                argmax = torch.tensor(to_choose, dtype=torch.long, device=device)

                valmax = ps[range(batch_size), argmax]

                chart[level][pos, :] = valmax
        # print(chart)
        chart_roots_batch = []
        for bt, roots in enumerate(root_list):
            chart_roots = []
            for root in roots:
                    # print('root', root)
                    # print(len(chart),chart[0].shape, chart[-1].shape)
                    chart_roots.append(chart[int(root[1]-1)][int(root[0])][bt])
            chart_roots_batch.append(chart_roots)
        return chart_roots_batch


    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        size = diora.outside_h.shape[-1]

        # Get the score for the ground truth tree.
        # gold_spans = self.makeLeftTree(info['spans'])
        gold_spans_r = self.makeRightTree(info['spans'])

        gold_scores = self.get_score_for_spans(sentences, diora.saved_scalars, gold_spans_r)

        # Get the score for maximal tree.
        parse_predictor = CKY(net=diora, word2idx=self.word2idx)
        max_trees = parse_predictor.parse_batch({'sentences': sentences})
        max_spans = [tree_to_spans(x) for x in max_trees]

        max_scores = self.get_score_for_spans(sentences, diora.saved_scalars, max_spans)

        loss = max_scores - gold_scores + self.margin

        loss = loss.sum().view(1) / batch_size

        ret = dict(semi_supervised_parsing_loss=loss)

        return loss, ret


    def forward_(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        size = diora.outside_h.shape[-1]


        # Get the score for the ground truth tree.
        gold_spans = self.makeLeftTree(info['spans'])
        gold_spans_r = self.makeRightTree(info['spans'])
        gold_scores = self.get_score_for_spans_modified(sentences, diora.saved_scalars, gold_spans, info['spans'])
        gold_scores_r = self.get_score_for_spans_modified(sentences, diora.saved_scalars, gold_spans_r, info['spans'])

        # print("gold score", gold_scores)
        # print("right gold score", gold_scores_r)

        #print('info spans', info['spans'])
        #print('ner spans', gold_spans)


        # Get the score for maximal tree.
        parse_predictor = CKY(net=diora, word2idx=self.word2idx)
        max_trees = parse_predictor.parse_batch({'sentences': sentences})
        max_spans = [tree_to_spans(x) for x in max_trees]

        roots, diora_spans = self.findClosestParent(max_spans, info['spans'])
        # print('paresed spans', max_spans)
        # print('closest subtree spans', diora_spans)
        # print('closest roots', roots)

        gold_scores = self.get_score_for_spans_modified(sentences, diora.saved_scalars, gold_spans_r, info['spans'])
        max_scores = self.get_score_for_spans_modified(sentences, diora.saved_scalars, diora_spans, roots)

        total_loss = 0
        # print('gold scores', gold_scores)
        # print('max scores', max_scores)

        for dp in range(len(gold_scores)):
            gold_score_data = gold_scores[dp]
            max_score_data = max_scores[dp]
            # print(gold_score_data, max_score_data)
            loss = 0
            for i in range(len(gold_score_data)):
                if int(info['spans'][dp][i][0])==int(roots[dp][i][0]) and int(info['spans'][dp][i][1])==int(roots[dp][i][1]):
                    # print(info['spans'][dp][i], roots[dp][i])
                    continue
                else:
                    loss += max_score_data[i] - gold_score_data[i] + self.margin
            total_loss += loss
        #loss = max_scores - gold_scores + self.margin

        #loss = loss.sum().view(1) / batch_size
        loss = torch.tensor(total_loss / batch_size, requires_grad=True)
        # print("semi_supervised_parsing_loss", loss)
        ret = dict(semi_supervised_parsing_loss=loss)
        # print("-------------")
        # print('loss', loss)
        return loss, ret


def get_loss_funcs(options, batch_iterator=None, embedding_layer=None):
    input_dim = embedding_layer.weight.shape[1]
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    cuda = options.cuda

    loss_funcs = []

    # Reconstruction Loss
    if options.reconstruct_mode == 'margin':
        reconstruction_loss_fn = ReconstructionLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)
    elif options.reconstruct_mode == 'softmax':
        reconstruction_loss_fn = ReconstructionSoftmaxLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)
    elif options.reconstruct_mode == 'semi':
        reconstruction_loss_fn = SemiSupervisedParsingLoss(margin=margin, cuda=cuda, word2idx=batch_iterator.word2idx)
    loss_funcs.append(reconstruction_loss_fn)

    return loss_funcs


class Embed(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super(Embed, self).__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, x):
        batch_size, length = x.shape
        e = self.embeddings(x.view(-1))
        t = torch.mm(e, self.mat.t()).view(batch_size, length, -1)
        return t


class Net(nn.Module):
    def __init__(self, embed, diora, loss_funcs=[]):
        super(Net, self).__init__()
        size = diora.size

        self.embed = embed
        self.diora = diora
        self.loss_func_names = [m.name for m in loss_funcs]

        for m in loss_funcs:
            setattr(self, m.name, m)

        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def compute_loss(self, batch, neg_samples, info):
        ret, loss = {}, []

        # Loss
        diora = self.diora.get_chart_wrapper()
        for func_name in self.loss_func_names:
            func = getattr(self, func_name)
            subloss, desc = func(batch, neg_samples, diora, info)
            loss.append(subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v

        loss = torch.cat(loss, 1)

        return ret, loss

    def forward(self, batch, neg_samples=None, compute_loss=True, info=None):
        # Embed
        embed = self.embed(batch)

        # Run DIORA
        self.diora(embed)

        # Compute Loss
        if compute_loss:
            ret, loss = self.compute_loss(batch, neg_samples, info=info)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32,
                device=embed.device)

        # Results
        ret['total_loss'] = loss

        return ret


class Trainer(object):
    def __init__(self, net, k_neg=None, ngpus=None, cuda=None):
        super(Trainer, self).__init__()
        self.net = net
        self.optimizer = None
        self.optimizer_cls = None
        self.optimizer_kwargs = None
        self.cuda = cuda
        self.ngpus = ngpus

        self.parallel_model = None

        print("Trainer initialized with {} gpus.".format(ngpus))

    def freeze_diora(self):
        for p in self.net.diora.parameters():
            p.requires_grad = False

    def parameter_norm(self, requires_grad=True, diora=False):
        net = self.net.diora if diora else self.net
        total_norm = 0
        for p in net.parameters():
            if requires_grad and not p.requires_grad:
                continue
            total_norm += p.norm().item()
        return total_norm

    def init_optimizer(self, optimizer_cls, optimizer_kwargs):
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

    @staticmethod
    def get_single_net(net):
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            return net.module
        return net

    def save_model(self, model_file):
        state_dict = self.net.state_dict()

        todelete = []

        for k in state_dict.keys():
            if 'embeddings' in k:
                todelete.append(k)

        for k in todelete:
            del state_dict[k]

        torch.save({
            'state_dict': state_dict,
        }, model_file)

    @staticmethod
    def load_model(net, model_file):
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict_toload = save_dict['state_dict']
        state_dict_net = Trainer.get_single_net(net).state_dict()

        # Bug related to multi-gpu
        keys = list(state_dict_toload.keys())
        prefix = 'module.'
        for k in keys:
            if k.startswith(prefix):
                newk = k[len(prefix):]
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        # Remove extra keys.
        keys = list(state_dict_toload.keys())
        for k in keys:
            if k not in state_dict_net:
                print('deleting {}'.format(k))
                del state_dict_toload[k]

        # Hack to support embeddings.
        for k in state_dict_net.keys():
            if 'embeddings' in k:
                state_dict_toload[k] = state_dict_net[k]

        Trainer.get_single_net(net).load_state_dict(state_dict_toload)

    def run_net(self, batch_map, compute_loss=True, multigpu=False):
        batch = batch_map['sentences']
        neg_samples = batch_map.get('neg_samples', None)
        info = self.prepare_info(batch_map)
        out = self.net(batch, neg_samples=neg_samples, compute_loss=compute_loss, info=info)
        return out

    def gradient_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        params = [p for p in self.net.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        self.optimizer.step()

    def prepare_result(self, batch_map, model_output):
        result = {}
        result['batch_size'] = batch_map['batch_size']
        result['length'] = batch_map['length']
        for k, v in model_output.items():
            if 'loss' in k:
                result[k] = v.mean(dim=0).sum().item()
        return result

    def prepare_info(self, batch_map):
        info = {}
        if 'spans' in batch_map:
            info['spans'] = batch_map['spans']
        return info

    def step(self, *args, **kwargs):
        try:
            return self._step(*args, **kwargs)
        except Exception as err:
            batch_map = args[0]
            print('Failed with shape: {}'.format(batch_map['sentences'].shape))
            if self.ngpus > 1:
                print(traceback.format_exc())
                print('The step failed. Running multigpu cleanup.')
                os.system("ps -elf | grep [p]ython | grep adrozdov | grep " + self.experiment_name + " | tr -s ' ' | cut -f 4 -d ' ' | xargs -I {} kill -9 {}")
                sys.exit()
            else:
                raise err

    def _step(self, batch_map, train=True, compute_loss=True):
        if train:
            self.net.train()
        else:
            self.net.eval()
        multigpu = self.ngpus > 1 and train

        with torch.set_grad_enabled(train):
            model_output = self.run_net(batch_map, compute_loss=compute_loss, multigpu=multigpu)

        # Calculate average loss for multi-gpu and sum for backprop.
        total_loss = model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)

        result = self.prepare_result(batch_map, model_output)

        return result


def build_net(options, embeddings=None, batch_iterator=None, random_seed=None):

    logger = get_logger()

    lr = options.lr
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    normalize = options.normalize
    input_dim = embeddings.shape[1]
    cuda = options.cuda
    rank = options.local_rank
    ngpus = 1

    if cuda and options.multigpu:
        ngpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = options.master_addr
        os.environ['MASTER_PORT'] = options.master_port
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Embed
    embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
    embed = Embed(embedding_layer, input_size=input_dim, size=size)

    # Diora
    if options.arch == 'treelstm':
        diora = DioraTreeLSTM(size, outside=True, normalize=normalize, compress=False)
    elif options.arch == 'mlp':
        diora = DioraMLP(size, outside=True, normalize=normalize, compress=False)
    elif options.arch == 'mlp-shared':
        diora = DioraMLPShared(size, outside=True, normalize=normalize, compress=False)

    # Loss
    loss_funcs = get_loss_funcs(options, batch_iterator, embedding_layer)

    # Net
    net = Net(embed, diora, loss_funcs=loss_funcs)

    # Load model.
    if options.load_model_path is not None:
        logger.info('Loading model: {}'.format(options.load_model_path))
        Trainer.load_model(net, options.load_model_path)

    # CUDA-support
    if cuda:
        if options.multigpu:
            torch.cuda.set_device(options.local_rank)
        net.cuda()
        diora.cuda()

    if cuda and options.multigpu:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[rank], output_device=rank)

    # Trainer
    trainer = Trainer(net, k_neg=k_neg, ngpus=ngpus, cuda=cuda)
    trainer.rank = rank
    trainer.experiment_name = options.experiment_name # for multigpu cleanup
    trainer.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    return trainer