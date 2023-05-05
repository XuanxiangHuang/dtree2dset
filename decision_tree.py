#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Decision Tree Classifier
#
#
################################################################################
import collections
import pickle
from functools import reduce
import pandas as pd
import numpy as np
from queue import Queue
from Orange.classification import TreeLearner
from Orange.tree import NumericNode, DiscreteNode, MappedDiscreteNode
from Orange.data import Table
from sklearn.metrics import accuracy_score
################################################################################


class DecisionTree:
    """
        Using Orange, learning DT classifier from data,
        features could be: binary, categorical or continuous.
    """
    def __init__(self):
        self.features = None
        self.classes = None
        self.learner = None
        self.classifier = None
        self.train_acc = None
        self.test_acc = None
        self.node_feature_dom = None

    def train(self, train_dataset, test_dataset, max_depth=7):
        """
            Training decision tree with given datasets.

            :return: none.
        """
        train_datatable = Table(train_dataset)
        test_datatable = Table(test_dataset)
        self.features = [att.name for att in train_datatable.domain.attributes]
        self.classes = train_datatable.domain.class_var.name
        self.learner = TreeLearner(max_depth=max_depth, binarize=True)
        self.classifier = self.learner(train_datatable)

        self.train_acc = accuracy_score(train_datatable.Y, self.classifier(train_datatable.X))
        self.test_acc = accuracy_score(test_datatable.Y, self.classifier(test_datatable.X))

        return round(self.train_acc, 3), round(self.test_acc, 3)

    def __str__(self):
        res = ''
        res += ('Number of nodes: {0}\n'.format(self.num_nodes()))
        res += ('Depth: {0}\n'.format(self.max_depth()))
        res += self.classifier.print_tree()
        return res

    def num_nodes(self):
        """
        Count the number of nodes in DT.

        :return: number of nodes.
        """
        def _count(node):
            if not node:
                return 0
            return 1 + sum(_count(c) for c in node.children if c)
        return self.classifier.node_count()

    def max_depth(self):
        """
        Get the max depth of DT.
        :return: depth of DT.
        """
        def _depth(node):
            if not node:
                return 0
            return 1 + max((_depth(child) for child in node.children if child), default=0)
        # don't count the leaf
        return _depth(self.classifier.root) - 1

    @staticmethod
    def save_model(dt, filename):
        """
            Save DT to pickle model.

            :param dt: decision tree classifier.
            :param filename: filename storing dt classifier.
            :return: none.
        """
        with open(filename, "wb") as f:
            pickle.dump(dt, f)

    @classmethod
    def from_file(cls, filename):
        """
            Load DT classifier from file.

            :param filename: decision tree classifier in pickle.
            :return: decision tree.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def predict_one(self, in_x):
        """
            Get prediction of given one instance or sample.

            :param in_x: given instance.
            :return: prediction.
        """
        return self.classifier(in_x)

    def bfs(self, root):
        """
            Iterate through nodes in breadth first search (BFS) order.

            :param root: root node of decision tree.
            :return: a set of all tree nodes in BFS order.
        """
        def _bfs(nd, visited):
            queue = collections.deque()
            queue.appendleft(nd)
            while queue:
                node = queue.pop()
                if node and node not in visited:
                    if node.children:
                        for child in node.children:
                            queue.appendleft(child)
                    visited.add(node)
                    yield node

        yield from _bfs(root, set())

    def support_var(self, root):
        """
            Get dependency set of DT, i.e. exclude all don't-cared features.

            :param root: root of DT or sub-DT.
            :return: a set of feature index.
        """
        vars_ = set()
        for nd in self.bfs(root):
            if not len(nd.children):
                continue
            vars_.add(nd.attr_idx)
        return vars_

    def get_all_paths(self):
        """
            Get all decision paths.

            :return: a list decision paths.
        """
        paths = []

        def _get_all_paths(nd, path):
            if len(nd.children):
                assert len(nd.children) == 2
                # add (node, index of 1st child)
                path.append((nd, 0))
                _get_all_paths(nd.children[0], path)
                path.pop()
                # add (node, index of 2nd child)
                path.append((nd, 1))
                _get_all_paths(nd.children[1], path)
                path.pop()
            else:
                probs = nd.value / np.sum(np.array(nd.value))
                pred = np.argmax(probs, axis=-1)
                path.append(pred)
                paths.append(path[:])
                path.pop()

        _get_all_paths(self.classifier.root, [])

        return paths

    def get_path_lits(self, path):
        """
            Get path literals (feature, domain) of a decision path.

            :param path: given decision path.
            :return: a dict of feature to (feature_type, domain).
        """
        feat2dom = dict()
        for ele in path[:-1]:
            if ele[0].attr_idx not in feat2dom:
                # all features are either continuous or discrete
                # for discrete, use a list of sets to store domain along the path,
                # for continuous, use list of ('<=', threshold) or ('>', threshold) to store domain along the path
                if type(ele[0]) == DiscreteNode:
                    feat2dom.update({ele[0].attr_idx: [set(ele[0].attr.values[ele[1]])]})
                elif type(ele[0]) == MappedDiscreteNode:
                    feat2dom.update({ele[0].attr_idx: [set(ele[0].attr.values[i] for i in ele[0].children[ele[1]].condition)]})
                elif type(ele[0]) == NumericNode:
                    if ele[1] == 0:
                        feat2dom.update({ele[0].attr_idx: [('<=', ele[0].threshold)]})
                    else:
                        feat2dom.update({ele[0].attr_idx: [('>', ele[0].threshold)]})
                else:
                    assert False, "Unknown node type."
            else:
                tmp = feat2dom[ele[0].attr_idx]
                if type(ele[0]) == DiscreteNode:
                    tmp.append(set(ele[0].attr.values[ele[1]]))
                elif type(ele[0]) == MappedDiscreteNode:
                    tmp.append(set(ele[0].attr.values[i] for i in ele[0].children[ele[1]].condition))
                elif type(ele[0]) == NumericNode:
                    if ele[1] == 0:
                        tmp.append(('<=', ele[0].threshold))
                    else:
                        tmp.append(('>', ele[0].threshold))
                    tmp.sort(key=lambda x: len(x[0]))
                else:
                    assert False, "Unknown node type."
                feat2dom.update({ele[0].attr_idx: tmp})

        bound = dict()

        for feat in feat2dom:
            dom = feat2dom[feat]
            if type(dom[0]) == set:
                # intersection must be non-empty, otherwise it is dead-end
                ret = reduce(set.intersection, dom)
                assert ret, "Dead-end path."
                bound.update({feat: ret})
            else:
                max_gt = [-np.inf]
                min_le = [np.inf]
                for ele in dom:
                    # (lower-bound, upper-bound]
                    if ele[0] == '>':
                        max_gt.append(ele[1])
                    elif ele[0] == '<=':
                        min_le.append(ele[1])

                lower_b = max(max_gt)
                upper_b = min(min_le)
                assert lower_b <= upper_b
                assert lower_b != -np.inf or upper_b != np.inf
                if upper_b == np.inf:
                    bound.update({feat: pd.Interval(left=lower_b, right=upper_b, closed='neither')})
                else:
                    bound.update({feat: pd.Interval(left=lower_b, right=upper_b, closed='right')})

        return bound

    def path_to_another_class(self, path, univ):
        """
            Check if there is a path leading to another class other than
            the target of the given decision path.

            :param path: given decision path.
            :param univ: given a list of universal features.
            :return: true if there is else false.
        """
        assert self.node_feature_dom is not None

        decision_path_bound = self.get_path_lits(path)
        pred = path[-1]

        q = Queue()
        q.put(self.classifier.root)
        while not q.empty():
            nd = q.get()
            if not nd.children:
                probs = nd.value / np.sum(np.array(nd.value))
                target = np.argmax(probs, axis=-1)
                if pred != target:
                    return True
            else:
                if univ[nd.attr_idx]:
                    q.put(nd.children[0])
                    q.put(nd.children[1])
                else:
                    nd_bound0 = self.node_feature_dom[(nd, 0)]
                    nd_bound1 = self.node_feature_dom[(nd, 1)]
                    if type(nd) == DiscreteNode:
                        assert nd_bound0 != nd_bound1
                        if nd_bound0 == decision_path_bound[nd.attr_idx]:
                            q.put(nd.children[0])
                        if nd_bound1 == decision_path_bound[nd.attr_idx]:
                            q.put(nd.children[1])
                    elif type(nd) == MappedDiscreteNode:
                        assert not nd_bound0.intersection(nd_bound1)
                        if nd_bound0.intersection(decision_path_bound[nd.attr_idx]):
                            q.put(nd.children[0])
                        if nd_bound1.intersection(decision_path_bound[nd.attr_idx]):
                            q.put(nd.children[1])
                    elif type(nd) == NumericNode:
                        assert not nd_bound0.overlaps(nd_bound1)
                        # for 1st child <= threshold: if lower-bound, upper-bound of given path
                        # and lower-bound, upper-bound of nd are not disjoint
                        if nd_bound0.overlaps(decision_path_bound[nd.attr_idx]):
                            q.put(nd.children[0])
                        # for 2nd child > threshold: if lower-bound, upper-bound of given path
                        # and lower-bound, upper-bound of nd are not disjoint
                        if nd_bound1.overlaps(decision_path_bound[nd.attr_idx]):
                            q.put(nd.children[1])
                    else:
                        assert False, "Unknown node type."
        return False

    def node_feature_domain(self):
        """
            Get (feature, domain) for each node.

            :return: node to (feature, domain)
        """
        paths = []

        def _get_all_paths(nd, path):
            if len(nd.children):
                assert len(nd.children) == 2
                # add (node, index of 1st child)
                path.append((nd, 0))
                _get_all_paths(nd.children[0], path)
                path.pop()
                # add (node, index of 2nd child)
                path.append((nd, 1))
                _get_all_paths(nd.children[1], path)
                path.pop()
            else:
                probs = nd.value / np.sum(np.array(nd.value))
                pred = np.argmax(probs, axis=-1)
                path.append(pred)
                paths.append(path[:])
                path.pop()

        _get_all_paths(self.classifier.root, [])

        self.node_feature_dom = dict()
        for p in paths:
            # for each path, and each node in the path
            # get its feature domain
            path_nd_feat2dom = dict()
            for ele in p[:-1]:
                # go through the path, add multiple domain to a feature i,
                # suppose there are multiple nodes (node and its ancestor) having the same feature i
                # and then compute the intersection of domains of this feature i,
                # the intersection is the specific domain of this feature i at a child node
                if ele[0].attr_idx not in path_nd_feat2dom:
                    if type(ele[0]) == DiscreteNode:
                        path_nd_feat2dom.update({ele[0].attr_idx: [set(ele[0].attr.values[ele[1]])]})
                    elif type(ele[0]) == MappedDiscreteNode:
                        path_nd_feat2dom.update(
                            {ele[0].attr_idx: [set(ele[0].attr.values[i] for i in ele[0].children[ele[1]].condition)]})
                    elif type(ele[0]) == NumericNode:
                        if ele[1] == 0:
                            path_nd_feat2dom.update({ele[0].attr_idx: [('<=', ele[0].threshold)]})
                        else:
                            path_nd_feat2dom.update({ele[0].attr_idx: [('>', ele[0].threshold)]})
                    else:
                        assert False, "Unknown node type."
                else:
                    tmp = path_nd_feat2dom[ele[0].attr_idx]
                    if type(ele[0]) == DiscreteNode:
                        tmp.append(set(ele[0].attr.values[ele[1]]))
                    elif type(ele[0]) == MappedDiscreteNode:
                        tmp.append(set(ele[0].attr.values[i] for i in ele[0].children[ele[1]].condition))
                    elif type(ele[0]) == NumericNode:
                        if ele[1] == 0:
                            tmp.append(('<=', ele[0].threshold))
                        else:
                            tmp.append(('>', ele[0].threshold))
                        tmp.sort(key=lambda x: len(x[0]))
                    else:
                        assert False, "Unknown node type."
                    path_nd_feat2dom.update({ele[0].attr_idx: tmp})

                dom = path_nd_feat2dom[ele[0].attr_idx]
                if type(dom[0]) == set:
                    # intersection must be non-empty, otherwise it is dead-end
                    ret = reduce(set.intersection, dom)
                    assert ret, "Dead-end path."
                    self.node_feature_dom.update({(ele[0], ele[1]): ret})
                else:
                    max_gt = [-np.inf]
                    min_le = [np.inf]
                    for d in dom:
                        # (lower-bound, upper-bound]
                        if d[0] == '>':
                            max_gt.append(d[1])
                        elif d[0] == '<=':
                            min_le.append(d[1])

                    lower_b = max(max_gt)
                    upper_b = min(min_le)
                    assert lower_b <= upper_b
                    assert lower_b != -np.inf or upper_b != np.inf
                    if upper_b == np.inf:
                        self.node_feature_dom.update({
                            (ele[0], ele[1]): pd.Interval(left=lower_b, right=upper_b, closed='neither')
                        })
                    else:
                        self.node_feature_dom.update({
                            (ele[0], ele[1]): pd.Interval(left=lower_b, right=upper_b, closed='right')
                        })
        return
