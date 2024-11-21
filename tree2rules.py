#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Convert Decision Tree classifier to Decision Rules classifier
#
#
################################################################################
from decision_tree import DecisionTree
################################################################################


class Tree2Rules:
    """
        Decision Tree to Decision Rules
    """
    def __init__(self, dt: DecisionTree, verb=0):
        self.model = dt
        self.features = dt.features
        self.targets = dt.classes
        self.verbose = verb

    def find_axp(self, path):
        """
            Compute one path abductive explanation (Axp).

            :param path: a decision path.
            :return: one path abductive explanation,
                    each element in the return Axp is a feature index.
        """
        assert self.model.node_feature_dom is not None

        decision_path_bound = self.model.get_path_lits(path)
        fix = [True if i in decision_path_bound else False for i in range(len(self.features))]

        for i in range(len(fix)):
            if fix[i]:
                fix[i] = not fix[i]
                if self.model.path_to_another_class(path, [not ele for ele in fix]):
                    fix[i] = not fix[i]

        axp = [i for i in range(len(fix)) if fix[i]]

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp: {axp}")
            elif self.verbose == 2:
                print(f"Axp: {axp} ({[self.features[i] for i in axp]})")

        return axp

    def check_one_axp(self, axp, path):
        """
            Check if given axp is subset-minimal.

            :param axp: one path abductive explanation.
            :param path: decision path consistent with AXp.
            :return: true if given axp is subset-minimal
                        else false.
        """
        assert self.model.node_feature_dom is not None

        path_feat = [ele[0].attr_idx for ele in path[:-1]]
        for feat in axp:
            assert feat in path_feat

        univ = [True] * len(self.features)
        for i in axp:
            univ[i] = not univ[i]
        for i in range(len(univ)):
            if not univ[i]:
                univ[i] = not univ[i]
                if self.model.path_to_another_class(path, univ):
                    univ[i] = not univ[i]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True

    def redundant_path(self, axp, path):
        """
            Check if given axp is also an axp of the given path.

            :param axp: one path abductive explanation.
            :param path: another decision path.
            :return: true if given axp is also an axp of the given path.
        """
        assert self.model.node_feature_dom is not None

        path_feat = [ele[0].attr_idx for ele in path[:-1]]
        for feat in axp:
            if feat not in path_feat:
                return False

        univ = [True] * len(self.features)
        for i in axp:
            univ[i] = not univ[i]

        if self.model.path_to_another_class(path, univ):
            # it is not the axp of another class
            return False
        # it is subset-minimal
        for i in range(len(univ)):
            if not univ[i]:
                univ[i] = not univ[i]
                if self.model.path_to_another_class(path, univ):
                    univ[i] = not univ[i]
                else:
                    return False

        return True

    def convert_dt_to_ds(self):
        """
            Convert DT to intermediate DS.
            :return: a list of (feature, interval) followed by the class
        """
        paths = self.model.get_all_paths()
        axps = []
        tmp_ds = []
        # get (lower-bound, upper-bound) for each node's feature
        if self.model.node_feature_dom is None:
            self.model.node_feature_domain()

        buckets = dict()
        for p in paths:
            target_class = p[-1]
            if target_class not in buckets:
                buckets[target_class] = []
            buckets[target_class].append(p[:])
        path_len_axp_len = []

        for target_class in buckets:
            ps = buckets[target_class]
            redund = [False] * len(ps)
            for i in range(len(ps)):
                if redund[i]:
                    continue
                # compute an AXp for each path and filter out redundant path
                axp = self.find_axp(ps[i])
                assert self.check_one_axp(axp, ps[i])
                for j in range(i+1, len(ps)):
                    if redund[j]:
                        continue
                    # check if this AXp is also an AXp for path[j]
                    if self.redundant_path(axp, ps[j]):
                        redund[j] = True
                axps.append(axp)
                # when compare path length and axp length, we do not count the class
                path_len_axp_len.append((len(ps[i])-1, len(axp)))
                # build a rule based on computed AXp
                bound = self.model.get_path_lits(ps[i])
                tmp_rule = [(ele, bound[ele]) for ele in axp]
                tmp_rule.sort(key=lambda x: x[0])
                tmp_rule.append(ps[i][-1])
                if tmp_rule not in tmp_ds:
                    tmp_ds.append(tmp_rule)
        tmp_ds.sort(key=lambda x: x[-1])
        assert not self.check_overlap(tmp_ds)

        return paths, axps, tmp_ds, path_len_axp_len

    def check_overlap(self, tmp_ds):
        """
            Check if there are overlap (there are two rules such that
            the conditions are intersected but the output are different)

            :param tmp_ds: a list of (feature, interval) followed by the class
            :return:
        """
        def _conflict_rules(r_i, r_j):
            """
                Two rules (leading to two different classes) contain conflict literals
                :param r_i: rule i
                :param r_j: rule j
                :return:
            """
            assert r_i[-1] != r_j[-1]
            for ele_i in r_i[:-1]:
                for ele_j in r_j[:-1]:
                    if ele_i[0] == ele_j[0]:
                        # there should be at least one feature such that
                        # the associated domains are disjoint
                        if type(ele_i[1]) == set:
                            if not ele_i[1].intersection(ele_j[1]):
                                return True
                        else:
                            if not ele_i[1].overlaps(ele_j[1]):
                                return True
            # if the associated domains of same feature intersected,
            # or if they don't have same feature
            # then rules are overlap
            return False

        for i in range(len(tmp_ds)):
            rule_i = tmp_ds[i]
            for j in range(i+1, len(tmp_ds)):
                rule_j = tmp_ds[j]
                if rule_i[-1] == rule_j[-1]:
                    continue
                if not _conflict_rules(rule_i, rule_j):
                    return True
        return False

    def save_ds(self, tmp_ds, filename):
        """
            Save DS to file
            :param tmp_ds: given a list of (feature, interval) followed by the class.
            :return: None
        """
        output_DS = []
        for tmp_rule in tmp_ds:
            conds = []
            for ele in tmp_rule[:-1]:
                feat_name = self.features[ele[0]]
                if type(ele[1]) == set:
                    conds.append(f"{feat_name} ∈ " + "{" + ', '.join(str(x) for x in sorted(list(ele[1]))) + "}")
                else:
                    conds.append(f"{feat_name} ∈ ({ele[1].left}, {ele[1].right}]")
            str_rule = "IF "
            str_rule += ' AND '.join(conds)
            str_rule += f" THEN {self.targets} == {tmp_rule[-1]}"
            output_DS.append(str_rule)

            with open(filename, "w") as f:
                for rule in output_DS:
                    f.write(f"{rule}\n")
