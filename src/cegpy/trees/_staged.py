"""cegpy Staged Tree"""
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, chain
import itertools
from operator import add, sub, itemgetter
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
import scipy.special
from IPython.display import Image

from cegpy.trees._event import EventTree
from cegpy.utilities._util import generate_colours


# pylint: disable=too-many-instance-attributes
class StagedTree(EventTree):
    """Representation of a Staged Tree.

    A staged tree is a tree where each node is a situation, and each edge is a
    transition from one situation to another. Each situation is given a 'stage'
    which groups it with other situations which have the same outgoing edges,
    with equivalent probabilities of occurring.

    The class is an extension of EventTree.

    :param dataframe: Required - DataFrame containing variables as column headers,
        with event name strings in each cell. These event names will be used to
        create the edges of the event tree. Counts of each event will
        be extracted and attached to each edge.
    :type dataframe: pandas.DataFrame

    :param sampling_zero_paths: Optional - Paths to sampling
        zeros.

        Format is as follows: [('edge_1',), ('edge_1', 'edge_2'), ...]

        If no paths are specified, default setting is that no sampling zero paths
        are created.
    :type sampling_zero_paths: List[Tuple[str]] or None

    :param var_order: Optional - Specifies the ordering of variables to be adopted
        in the event tree.
        Default var_order is obtained from the order of columns in dataframe.
        String labels in the list should match the column names in dataframe.
    :type var_order: List[str] or None

    :param struct_missing_label: Optional - Label in the dataframe for observations
        which are structurally missing; e.g: Post operative health status is
        irrelevant for a dead patient.
        Label example: "struct".
    :type struct_missing_label: str or None

    :param missing_label: Optional - Label in the dataframe for observations which are
        missing values that are not structurally missing.
        e.g: Missing height for some individuals in the sample.
        Label example: "miss"
        Whatever label is provided will be renamed in the event tree to "missing".
    :type missing_label: str or None

    :param complete_case: Optional - If True, all entries (rows) with non-structural
        missing values are removed. Default setting: False.
    :type complete_case: bool
    """

    _edge_attributes: List = ["count", "prior", "posterior", "probability"]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dataframe,
        sampling_zero_paths=None,
        var_order=None,
        struct_missing_label=None,
        missing_label=None,
        complete_case=False,
    ) -> None:
        # Call event tree init to generate event tree
        super().__init__(
            dataframe=dataframe,
            sampling_zero_paths=sampling_zero_paths,
            var_order=var_order,
            struct_missing_label=struct_missing_label,
            missing_label=missing_label,
            complete_case=complete_case,
        )

        self._mean_posterior_probs = []
        self._merged_situations = []
        self._stage_colours = []
        self._sort_count = 0
        self._colours_for_situations = []
        self._ahc_output = {}

    @property
    def prior(self) -> Dict[Tuple[str], List[Fraction]]:
        """A mapping of priors keyed by edge. Keys are
        3-tuples of the form: (src, dst, edge_label).

        :return: A mapping edge -> priors.
        :rtype: Dict[Tuple[str], List[Fraction]]
        """
        return nx.get_edge_attributes(self, "prior")

    @prior.setter
    def prior(self, prior: List[List[Fraction]]) -> None:
        offset = 0
        for node_idx, node_priors in enumerate(prior):
            node_name = f"s{node_idx + offset}"
            while not self.succ[node_name]:
                offset += 1
                node_name = f"s{node_idx + offset}"
            for edge_prior_idx, succ_key in enumerate(self.succ[node_name].keys()):
                label = list(self.succ[node_name][succ_key])[0]
                self.edges[(node_name, succ_key, label)]["prior"] = Fraction(
                    node_priors[edge_prior_idx]
                )

    @property
    def prior_list(self) -> List[List[Fraction]]:
        """
        :return: Priors in the form of a list of lists.
        :rtype: List[List[Fraction]]
        """

        prior_list = []
        prev_node = list(self.prior)[0][0]
        succ_list = []

        for edge, prior in self.prior.items():
            node = edge[0]
            if node != prev_node:
                prior_list.append(succ_list)
                succ_list = []
            succ_list.append(prior)
            prev_node = node

        if succ_list:
            prior_list.append(succ_list)
        return prior_list

    @property
    def posterior(self) -> Dict[Tuple[str], List[Fraction]]:
        """Posteriors along each edge, calculated by adding edge count to the prior for
        each edge.
        Keys are 3-tuples of the form: (src, dst, edge_label).

        :return: Mapping of edge -> edge_count + prior
        :rtype: Dict[Tuple[str], List[Fraction]]
        """
        try:
            posterior = nx.get_edge_attributes(self, "posterior")
            if posterior == {}:
                raise AttributeError("Posterior not yet set.")
            return posterior
        except AttributeError:
            for edge in self.edges:
                edge_dict = self.edges[edge]
                posterior = edge_dict["prior"] + edge_dict["count"]
                edge_dict["posterior"] = posterior
            return nx.get_edge_attributes(self, "posterior")

    @property
    def posterior_list(self) -> List[List[Fraction]]:
        """
        :return: Posteriors in the form of a list of lists.
        :rtype: List[List[Fraction]]
        """
        posterior_list = []
        prev_node = list(self.posterior)[0][0]
        succ_list = []

        for edge, posterior in self.posterior.items():
            node = edge[0]
            if node != prev_node:
                posterior_list.append(succ_list)
                succ_list = []
            succ_list.append(posterior)
            prev_node = node

        if succ_list:
            posterior_list.append(succ_list)
        return posterior_list

    @property
    def alpha(self) -> float:
        """
        The equivalent sample size set for the root node which is then uniformly
        propagated through the tree.

        :return: The value of Alpha.
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def hyperstage(self) -> List[List[str]]:
        """
        Indication of which nodes are allowed to be in the same stage. Each
        list is a list of node names e.g. "s0".

        :return: The List of all hyperstages.
        :rtype: List[List[str]]
        """
        return self._hyperstage

    @hyperstage.setter
    def hyperstage(self, value):
        self._hyperstage = value

    @property
    def initial_staging(self) -> List[List[str]]:
        """
        Specifies groups of nodes which are forced to be in the same stage before
        the AHC selection algorithm is run.

        :return: The List of all the non-singleton initial stages
        :rtype: List[List[str]]"""
        return self._initial_staging

    @initial_staging.setter
    def initial_staging(self, value):
        self._initial_staging = value

    @property
    def edge_countset(self) -> List[List]:
        """
        Indexed the same as situations.
        :return: Edge counts emination from each node of the tree.
        :rtype: List[List]
        """
        return self._create_edge_countset()

    @property
    def ahc_output(self) -> Dict:
        """
        Contains a List of Lists containing all the situations that were merged,
        and the log likelihood.

        :return: The output from the AHC algorithm.
        :rtype: Dict
        """
        return self._ahc_output

    def _check_hyperstages(self, hyperstage) -> None:
        hyper_situations = chain(*hyperstage)
        hyper_situations_set = set(hyper_situations)
        situations_set = set(self.situations)
        # Check if all situations are present
        missing = situations_set.difference(hyper_situations_set)
        if missing:
            raise ValueError(
                f"Situation(s) {missing} are missing from the list of hyperstages."
            )
        # Check if all situations provided exist
        extra = hyper_situations_set.difference(situations_set)
        if extra:
            raise ValueError(f"Situation(s) {extra} are not present in the tree.")
        # Check if all situations in a stage have the same number of edges
        for stage in hyperstage:
            n_edges = self.out_degree[stage[0]]
            for node in stage:
                if self.out_degree[node] != n_edges:
                    raise ValueError(
                        "Situations in the same hyperstage "
                        "must have the same number of outgoing edges."
                    )

    def _check_prior(self, node_priors_list) -> None:
        if len(node_priors_list) != len(self.edge_countset):
            raise ValueError(
                "Number of sub-lists in the list of priors "
                "must agree with the number of situations."
            )

        for node_idx, node_priors in enumerate(node_priors_list):
            if len(node_priors) != len(self.edge_countset[node_idx]):
                raise ValueError(
                    "The length of each sub-list in the list of priors "
                    "must agree with the number of edges emanating from "
                    "its corresponding situation."
                )
            for node_prior in node_priors:
                if node_prior < 0:
                    raise ValueError("All priors must be non-negative.")

    def _store_params(self, prior, alpha, hyperstage, initial_staging) -> None:
        """User has passed in AHC params, this function processes them,
        and generates any default AHC params if required."""
        if prior:
            if alpha:
                self.alpha = None
            self._check_prior(prior)
            self.prior = prior
        else:
            if alpha is None:
                self.alpha = self._calculate_default_alpha()
            else:
                self.alpha = alpha

            # No matter what alpha is, generate default prior
            self.prior = self._create_default_prior(self.alpha)

        if hyperstage is None:
            self.hyperstage = self._create_default_hyperstage()
        else:
            self._check_hyperstages(hyperstage)
            self.hyperstage = hyperstage

        if hyperstage and initial_staging:
            if not self._validate_hyperstage(initial_staging, hyperstage):
                raise ValueError(
                    "initial_staging is not contained within the hyperstage."
                )
        if initial_staging:
            if not self._validate_initial_staging_mutually_exclusive(initial_staging):
                raise ValueError(
                    "All groups of nodes in initial_staging must be mutually exclusive."
                )
        self.initial_staging = initial_staging

    def _calculate_default_alpha(self) -> int:
        """If no alpha is given, a default value is calculated.
        The value is calculated by determining the maximum number
        of categories that any one variable has"""
        max_count = max(list(self.categories_per_variable.values()))
        return max_count

    def _create_default_prior(self, alpha) -> list:
        """default prior set for the AHC method using the mass conservation property.
        That is, the alpha param is the phantom sample starting at
        the root, and it is spread equally across all edges along the tree.
        (see chapter 5 of Collazo, Gorgen & Smith 'Chain Event Graphs', 2018)
        The prior is a list of lists. Each list gives the prior along the
        edges of a specific situation.
        Indexed same as self.situations & self.egde_countset"""

        default_prior = [0] * len(self.situations)
        sample_size_at_node = {}

        # Root node is assigned phantom sample (alpha)
        if isinstance(alpha, float):
            alpha = Fraction.from_float(alpha)
        elif isinstance(alpha, (int, str)):
            alpha = Fraction.from_float(float(alpha))
        else:
            raise TypeError(
                "Prior generator param alpha is in a strange format.\
                            .."
            )

        sample_size_at_node[self.root] = alpha

        for node_idx, node in enumerate(self.situations):
            # How many nodes emanate from the current node?
            number_of_emanating_nodes = self.out_degree[node]

            # Divide the sample size from the current node equally among
            # emanating nodes
            equal_distribution_of_sample = (
                sample_size_at_node[node] / number_of_emanating_nodes
            )
            default_prior[node_idx] = [
                equal_distribution_of_sample
            ] * number_of_emanating_nodes

            relevant_terminating_nodes = [
                edge[1] for edge in list(self.edges) if edge[0] == node
            ]

            for terminating_node in relevant_terminating_nodes:
                sample_size_at_node[terminating_node] = equal_distribution_of_sample

        return default_prior

    def _create_default_hyperstage(self) -> list:
        """Generates default hyperstage for the AHC method.
        A hyperstage is a list of lists such that two situaions can be in the
        same stage only if there are elements of the same list for some list
        in the hyperstage.
        The default is to allow all situations with the same number of
        outgoing edges and the same edge labels to be in a common list."""
        hyperstage = []
        info_of_edges = []

        for node in self.situations:
            labels = [edge[2] for edge in self.edges if edge[0] == node]
            labels.sort()

            info_of_edges.append([self.out_degree[node], labels])

        sorted_info = []
        for info in info_of_edges:
            if info not in sorted_info:
                sorted_info.append(info)

        for info in sorted_info:
            situations_with_value_edges = []
            for idx, situation in enumerate(self.situations):
                if info_of_edges[idx] == info:
                    situations_with_value_edges.append(situation)
            hyperstage = hyperstage + [situations_with_value_edges]

        return hyperstage

    def _create_edge_countset(self) -> list:
        """Each element of list contains a list with counts along edges emanating from
        a specific situation. Indexed same as self.situations"""
        edge_countset = []

        for node in self.situations:
            edge_countset.append(
                [count for edge, count in self.edge_counts.items() if edge[0] == node]
            )
        return edge_countset

    @staticmethod
    def _calculate_lg_of_sum(array) -> float:
        """function to calculate log gamma of the sum of an array"""
        array = [float(x) for x in array]
        return scipy.special.gammaln(sum(array))

    @staticmethod
    def _calculate_sum_of_lg(array) -> float:
        """function to calculate log gamma of each element of an array"""
        return sum([scipy.special.gammaln(float(x)) for x in array])

    def _calculate_initial_loglikelihood(self, prior, posterior) -> float:
        """calculating log likelihood given a prior and posterior"""
        # Calculate prior contribution

        pri_lg_of_sum = [self._calculate_lg_of_sum(elem) for elem in prior]
        pri_sum_of_lg = [self._calculate_sum_of_lg(elem) for elem in prior]
        pri_contribution = list(map(sub, pri_lg_of_sum, pri_sum_of_lg))

        # Calculate posterior contribution
        post_lg_of_sum = [self._calculate_lg_of_sum(elem) for elem in posterior]
        post_sum_of_lg = [self._calculate_sum_of_lg(elem) for elem in posterior]
        post_contribution = list(map(sub, post_sum_of_lg, post_lg_of_sum))

        sum_pri_contribution = sum(pri_contribution)
        sum_post_contribution = sum(post_contribution)
        return sum_pri_contribution + sum_post_contribution

    def _calculate_bayes_factor(self, prior1, posterior1, prior2, posterior2) -> float:
        """calculates the bayes factor comparing two models which differ in
        only one stage"""
        if all(prior1) == 0 or all(prior2) == 0:
            return -1.00
        else:
            new_prior = list(map(add, prior1, prior2))
            new_posterior = list(map(add, posterior1, posterior2))
            return (
                self._calculate_lg_of_sum(new_prior)
                - self._calculate_lg_of_sum(prior1)
                - self._calculate_lg_of_sum(prior2)
                - self._calculate_lg_of_sum(new_posterior)
                + self._calculate_lg_of_sum(posterior1)
                + self._calculate_lg_of_sum(posterior2)
                + self._calculate_sum_of_lg(new_posterior)
                - self._calculate_sum_of_lg(posterior1)
                - self._calculate_sum_of_lg(posterior2)
                - self._calculate_sum_of_lg(new_prior)
                + self._calculate_sum_of_lg(prior1)
                + self._calculate_sum_of_lg(prior2)
            )

    def _sort_list(self, list_of_tuples) -> list:
        """function to sort a list of lists to remove repetitions"""

        for idx_1, tup_1 in enumerate(list_of_tuples):
            for idx_2, tup_2 in enumerate(list_of_tuples):
                tups_intersect = set(tup_1) & set(tup_2)
                if tups_intersect:
                    union = tuple(set(tup_1) | set(tup_2))
                    list_of_tuples[idx_1] = []
                    list_of_tuples[idx_2] = union

        new_list_of_tuples = [elem for elem in list_of_tuples if elem != []]

        if new_list_of_tuples == list_of_tuples:
            return new_list_of_tuples

        return self._sort_list(new_list_of_tuples)

    def _sort_merged_sit_tuples(self, merged_sit_list: List[Tuple[str]]) -> None:
        """Sorts the tuples in the merged situations."""
        prefix = self.root[0]
        for idx, merged in enumerate(merged_sit_list):
            if len(merged) > 1:
                node_nums = [int(node.split(prefix)[1]) for node in merged]
                node_nums.sort()
                merged = tuple(node_nums)
                merged_sit_list[idx] = tuple(
                    [f"{prefix}{node_num}" for node_num in node_nums]
                )

    def _calculate_mean_posterior_probs(
        self,
        merged_situations: List,
        posteriors: List,
    ) -> List:
        """Iterates through array of lists, calculates mean
        posterior probabilities"""
        mean_posterior_probs = []

        for sit in self.situations:
            if sit not in list(chain(*merged_situations)):
                merged_situations.append((sit,))
        for stage in merged_situations:
            for sit in stage:
                sit_idx = self.situations.index(sit)
                if all(posteriors[sit_idx]) != 0:
                    stage_probs = posteriors[sit_idx]
                    break
                stage_probs = []

            total = sum(stage_probs)
            mean_posterior_probs.append(
                [round(elem / total, 3) for elem in stage_probs]
            )

        return mean_posterior_probs

    @staticmethod
    def _independent_hyperstage_generator(hyperstages: List[List]) -> List[List[List]]:
        """Spit out the next hyperstage that can be dealt with
        independently."""
        new_hyperstages = [[hyperstages[0]]]

        for sublist in hyperstages[1:]:
            hs_to_add = [sublist]

            for hyperstage in new_hyperstages.copy():
                for other_sublist in hyperstage:
                    if not set(other_sublist).isdisjoint(set(sublist)):
                        hs_to_add.extend(hyperstage)
                        new_hyperstages.remove(hyperstage)

            new_hyperstages.append(hs_to_add)

        return new_hyperstages

    def _execute_ahc(
        self, hyperstage: Optional[List[List[str]]] = None, initial_staging: Optional[List[List[str]]] = None
    ) -> Tuple[List, float, List]:
        """finds all subsets and scores them"""
        if hyperstage is None:
            hyperstage = deepcopy(self.hyperstage)

        priors = deepcopy(self.prior_list)
        posteriors = deepcopy(self.posterior_list)

        loglikelihood = self._calculate_initial_loglikelihood(priors, posteriors)
        merged_situation_list = []

        # Which list in hyperstage have only 1 edge coming out of them
        # For that list, add the list to the merged situation list, and
        # add together their priors/posteriors, and remove it from the
        # hyperstage
        for sub_hyper in deepcopy(hyperstage):
            if self.out_degree[sub_hyper[0]] == 1:
                merged_situation_list.append(tuple(sub_hyper))
                indexes = [self.situations.index(situ) for situ in sub_hyper]
                if len(indexes) == 1:
                    sub_hyper_priors = itemgetter(*indexes)(priors)
                    sub_hyper_posteriors = itemgetter(*indexes)(posteriors)
                else:
                    sub_hyper_priors = list(chain(*itemgetter(*indexes)(priors)))
                    sub_hyper_posteriors = list(
                        chain(*itemgetter(*indexes)(posteriors))
                    )
                for loop_idx, p_index in enumerate(indexes):
                    if loop_idx == 0:
                        priors[p_index] = [sum(sub_hyper_priors)]
                        posteriors[p_index] = [sum(sub_hyper_posteriors)]
                    else:
                        priors[p_index] = [0]
                        posteriors[p_index] = [0]

                hyperstage.remove(sub_hyper)

        hyperstage_combinations = [
            item for sub_hyper in hyperstage for item in combinations(sub_hyper, 2)
        ]

        if initial_staging:
            initial_staging_combinations = [
                (sub_initial[0], node)
                for sub_initial in initial_staging
                for node in sub_initial[1:]
            ]

            for situ_pair in initial_staging_combinations:
                score = self._calculate_bayes_factor(
                    priors[self.situations.index(situ_pair[0])],
                    posteriors[self.situations.index(situ_pair[0])],
                    priors[self.situations.index(situ_pair[1])],
                    posteriors[self.situations.index(situ_pair[1])],
                )
                pair_indexes = (
                    self.situations.index(situ_pair[0]),
                    self.situations.index(situ_pair[1]),
                )
                merged_situation_list.append(situ_pair)

                priors[pair_indexes[0]] = list(
                    map(add, priors[pair_indexes[0]], priors[pair_indexes[1]])
                )
                posteriors[pair_indexes[0]] = list(
                    map(add, posteriors[pair_indexes[0]], posteriors[pair_indexes[1]])
                )
                priors[pair_indexes[1]] = [0] * len(priors[pair_indexes[0]])
                posteriors[pair_indexes[1]] = [0] * len(posteriors[pair_indexes[0]])

                loglikelihood += score

                for comb in deepcopy(hyperstage_combinations):
                    if situ_pair[1] in comb:
                        hyperstage_combinations.remove(comb)

        while True:
            newscores_list = [
                self._calculate_bayes_factor(
                    priors[self.situations.index(sub_hyper[0])],
                    posteriors[self.situations.index(sub_hyper[0])],
                    priors[self.situations.index(sub_hyper[1])],
                    posteriors[self.situations.index(sub_hyper[1])],
                )
                for sub_hyper in hyperstage_combinations
            ]

            local_score = max(newscores_list)

            if local_score > 0:
                local_merged = hyperstage_combinations[
                    newscores_list.index(local_score)
                ]
                merge_situ_1, merge_situ_2 = local_merged
                merge_situ_1_idx = self.situations.index(merge_situ_1)
                merge_situ_2_idx = self.situations.index(merge_situ_2)
                merged_situation_list.append(local_merged)

                priors[merge_situ_1_idx] = list(
                    map(add, priors[merge_situ_1_idx], priors[merge_situ_2_idx])
                )
                posteriors[merge_situ_1_idx] = list(
                    map(add, posteriors[merge_situ_1_idx], posteriors[merge_situ_2_idx])
                )
                priors[merge_situ_2_idx] = [0] * len(priors[merge_situ_1_idx])
                posteriors[merge_situ_2_idx] = [0] * len(posteriors[merge_situ_1_idx])

                loglikelihood += local_score
            else:
                break

        merged_situation_list = self._sort_list(merged_situation_list)
        self._sort_merged_sit_tuples(merged_situation_list)
        for sit in self.situations:
            if sit not in list(chain(*merged_situation_list)):
                merged_situation_list.append((sit,))

        self._apply_mean_posterior_probs(
            merged_situations=merged_situation_list,
            mean_posterior_probs=_calculate_mean_posterior_probs(
                self.situations, merged_situation_list, posteriors
            ),
        )

        return loglikelihood, merged_situation_list

    def _mark_nodes_with_stage_number(self, merged_situations):
        """AHC algorithm creates a list of indexes to the situations list.
        This function takes those indexes and creates a new list which is
        in a string representation of nodes."""
        self._sort_count = 0
        stage_count = 0
        for stage in merged_situations:
            if len(stage) > 1:
                for node in stage:
                    self.nodes[node]["stage"] = stage_count
                stage_count += 1

    def _generate_colours_for_situations(self, merged_situations, colour_list):
        """Colours each stage of the tree with an individual colour"""
        num_colours = len([m for m in merged_situations if len(m) > 1])
        if colour_list is None:
            stage_colours = generate_colours(num_colours)
        else:
            stage_colours = colour_list
            if len(colour_list) < num_colours:
                raise IndexError(
                    f"The number of colours provided ({len(colour_list)}) is "
                    "less than the number of distinct colours required "
                    f"({num_colours})."
                )
        self._stage_colours = stage_colours
        for node in self.nodes:
            if node in self.leaves:
                self.nodes[node]["colour"] = "lightgrey"
            else:
                try:
                    stage = self.nodes[node]["stage"]
                    self.nodes[node]["colour"] = stage_colours[stage]
                except KeyError:
                    self.nodes[node]["colour"] = "white"

    @staticmethod
    def _validate_hyperstage(initial_staging: List, hyperstage) -> bool:
        for init_stage in initial_staging:
            found = 0
            # init_stage is a subset of only one stage in hyperstages
            for h_stage in hyperstage:
                if set(init_stage).issubset(h_stage):
                    found += 1
            if found == 0:
                return False
        return True

    @staticmethod
    def _validate_initial_staging_mutually_exclusive(initial_staging: List) -> bool:
        if len(set(chain(*initial_staging))) != len(list(chain(*initial_staging))):
            return False
        return True

    def calculate_AHC_transitions(
        self,
        prior=None,
        alpha=None,
        hyperstage=None,
        colour_list=None,
        initial_staging=None,
    ) -> Dict:
        """Bayesian Agglommerative Hierarchical Clustering algorithm
        implementation. It returns a list of lists of the situations which
        have been merged together, the likelihood of the final model.

        :param prior: Optional - For each node in the event tree, ordered by node index,
            a list of prior parameters for each outgoing edge, in alphabetical order by
            edge label.
        :type prior: List[List[Fraction]]

        :param alpha: Optional - The equivalent sample size set for the root node
            which is then uniformly propagated through the tree.
        :type alpha: float

        :param hyperstage: Optional - Indication of which nodes are allowed to be
            in the same stage. Each list is a list of node names e.g. "s0".
        :type hyperstage: List[List[str]]

        :param colour_list: Optional - a list of hex colours to be used for stages.
            Otherwise, colours evenly spaced around the colour spectrum are used.
        :type colour_list: List[str]

        :return: The output from the AHC algorithm, specified above.
        :rtype: Dict
        """

        self._store_params(prior, alpha, hyperstage, initial_staging)

        loglikelihood, merged_situations = self._execute_ahc(
            initial_staging=initial_staging
        )

        self._mark_nodes_with_stage_number(merged_situations)

        self._generate_colours_for_situations(merged_situations, colour_list)

        self._ahc_output = {
            "Merged Situations": merged_situations,
            "Log Likelihood": loglikelihood,
        }
        return self.ahc_output

    def dot_graph(self, edge_info: str = "count", staged: bool = True):
        """Returns Dot graph representation of the staged tree.

        :param edge_info: Optional - Chooses which summary measure to be displayed on edges. Defaults to "count". Options: ["count", "prior", "posterior", "probability", "none"]
        :type edge_info: str

        :param staged: if True, returns the coloured staged tree,
            if False, returns the underlying event tree.
        :type staged: bool

        :return: A graphviz Dot representation of the graph.
        :rtype: pydotplus.Dot
        """
        if staged:
            return self._generate_dot_graph(edge_info=edge_info)
        else:
            return super()._generate_dot_graph(
                fill_colour="lightgrey", edge_info=edge_info
            )

    def create_figure(
        self,
        filename: Optional[str] = None,
        edge_info: str = "count",
        staged: bool = True,
    ) -> Union[Image, None]:
        """Draws the coloured staged tree or the underlying event tree
        for the process described by the dataset.

        :param filename: Optional - When provided, file is saved to the filename,
            local to the current working directory.
            e.g. if filename = "output/event_tree.svg", the file will be saved to:
            cwd/output/staged_tree.svg
            Otherwise, if function is called inside an interactive notebook, image
            will be displayed in the notebook, even if filename is omitted.

            Supports any filetype that graphviz supports. e.g: "staged_tree.png" or
            "staged_tree.svg" etc.

        :type filename: str

        :param edge_info: Optional - Chooses which summary measure to be displayed on edges. Defaults to "count". Options: ["count", "prior", "posterior", "probability", "none"]
        :type edge_info: str

        :param staged: if True, returns the coloured staged tree,
            if False, returns the underlying event tree.
        :type staged: bool

        :return: The event tree Image object.
        :rtype: IPython.display.Image or None
        """
        if staged:
            try:
                if not self._ahc_output:
                    raise AttributeError

                graph = self.dot_graph(edge_info)
                graph_image = super()._create_figure(graph, filename)

            except AttributeError as err:
                raise RuntimeError(
                    "----- PLEASE RUN AHC ALGORITHM before trying to"
                    " export a staged tree graph -----"
                ) from err

            return graph_image

        return super().create_figure(filename, edge_info=edge_info)

    def _apply_mean_posterior_probs(
        self, merged_situations: List, mean_posterior_probs: List
    ) -> None:
        """Apply the mean posterior probabilities to each edge."""
        for stage_idx, stage in enumerate(merged_situations):
            for sit in stage:
                dst_nodes = list(chain(self.succ[sit]))
                edge_labels = list(chain(*self.succ[sit].values()))
                for edge_idx, label in enumerate(edge_labels):
                    self.edges[(sit, dst_nodes[edge_idx], label)][
                        "probability"
                    ] = mean_posterior_probs[stage_idx][edge_idx]


def _calculate_mean_posterior_probs(
    all_situations: List, merged_situations: List, posteriors: List
) -> List:
    """Given a staged tree, calculate the mean posterior probs."""
    # Add all situations that are not in a stage.
    mean_posterior_probs = []
    for stage in merged_situations:
        for sit in stage:
            sit_idx = all_situations.index(sit)
            if all(posteriors[sit_idx]) != 0:
                stage_posteriors = posteriors[sit_idx]
                break

        total = sum(stage_posteriors)
        mean_posterior_probs.append(
            [round(posterior / total, 3) for posterior in stage_posteriors]
        )

    return mean_posterior_probs
