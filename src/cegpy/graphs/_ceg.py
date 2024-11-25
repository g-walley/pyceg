"""Chain Event Graph"""

from collections import defaultdict
from copy import deepcopy
import itertools as it
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)
import pydotplus as pdp
import networkx as nx
from IPython.display import Image
from IPython import get_ipython

from cegpy.utilities._util import generate_filename_and_mkdir
from cegpy.trees._staged import StagedTree


class CegAlreadyGenerated(Exception):
    """Raised when a CEG is generated twice."""


class ChainEventGraph(nx.MultiDiGraph):
    """
    Representation of a Chain Event Graph.

    A Chain Event Graph reduces a staged tree.

    The class is an extension of NetworkX MultiDiGraph.

    :param staged_tree: A staged tree object where the stages have been calculated.
    :type staged_tree: StagedTree

    :param node_prefix: The prefix that is used for the nodes in the Chain Event Graph.
        Default = "w"
    :type node_prefix: str

    :param generate: Automatically generate the Chain Event Graph upon creation of the
        object. Default = True.
    :type generate: bool
    """

    _edge_attributes: List = ["count", "prior", "posterior", "probability"]

    sink_suffix: str = "&infin;"
    node_prefix: str
    generated: bool = False

    def __init__(
        self,
        staged_tree: Optional[StagedTree] = None,
        node_prefix: str = "w",
        generate: bool = True,
    ):
        self.ahc_output = deepcopy(getattr(staged_tree, "ahc_output", {}))
        super().__init__(staged_tree, attr={})
        self.node_prefix = node_prefix
        self._stages = {}
        self.staged_root = staged_tree.root if staged_tree is not None else None

        if generate and staged_tree is not None:
            self.generate()

    @property
    def sink(self) -> str:
        """
        :return: Sink node name
        :rtype: str
        """
        return f"{self.node_prefix}_infinity"

    @property
    def root(self) -> str:
        """
        :return: Root node name
        :rtype: str
        """
        return f"{self.node_prefix}0"

    @property
    def stages(self) -> Mapping[str, Set[str]]:
        """
        :return: Mapping of stages to constituent nodes.
        :rtype: Mapping[str, Set[str]]
        """
        node_stages = dict(self.nodes(data="stage", default=None))
        stages = defaultdict(set)
        for node, stage in node_stages.items():
            stages[stage].add(node)

        return stages

    @property
    def path_list(self) -> List[List[Tuple[str]]]:
        """
        :return: All the paths through the CEG, as a list of lists of edge tuples.
        :rtype: List[List[Tuple[str]]]"""
        path_list: List[Tuple[str]] = list(
            nx.all_simple_edge_paths(self, self.root, self.sink)
        )
        return path_list

    def generate(self) -> None:
        """
        Identifies the positions i.e. the nodes of the CEG and the edges of the CEG
        along with their edge labels and edge counts. Here we use the
        algorithm from our paper with the optimal stopping time.
        """
        if self.generated:
            raise CegAlreadyGenerated("CEG has already been generated.")

        if self.ahc_output is None or self.ahc_output == {}:
            raise ValueError(
                "There is no AHC output in your StagedTree. "
                "Run StagedTree.calculate_AHC_transitions() first."
            )

        # rename root node:
        nx.relabel_nodes(self, {self.staged_root: self.root}, copy=False)
        self._trim_leaves_from_graph()
        self._update_distances_to_sink()
        self._backwards_construction(self._gen_nodes_with_increasing_distance(start=1))
        self._relabel_nodes()
        self.generated = True

    def _backwards_construction(self, node_generator: Iterable[str]) -> None:
        """Working backwards from the sink, the algorithm constructs the CEG."""
        next_set_of_nodes: List = next(node_generator)

        while next_set_of_nodes != [self.root]:
            nodes_to_merge = []
            while len(next_set_of_nodes) > 1:
                node_1 = next_set_of_nodes.pop(0)
                for node_2 in next_set_of_nodes:
                    mergeable = self._check_nodes_can_be_merged(node_1, node_2)
                    if mergeable:
                        nodes_to_merge.append((node_1, node_2))

            if nodes_to_merge:
                self._merge_nodes(nodes_to_merge)

            try:
                next_set_of_nodes: List = next(node_generator)
            except StopIteration:
                break

    def _merge_nodes(self, nodes_to_merge: List[Tuple[str]]):
        """nodes to merge should be a set of 2 element tuples"""
        temp_1 = "temp_1"
        temp_2 = "temp_2"
        while nodes_to_merge:
            nodes = nodes_to_merge.pop(0)
            new_node = nodes[0]
            # Copy nodes to temp nodes
            node_map = {nodes[0]: temp_1, nodes[1]: temp_2}
            nx.relabel_nodes(self, node_map, copy=False)
            self.add_node(new_node)

            edges_to_remove = self._merge_and_add_edges(
                new_node,
                temp_1,
                temp_2,
            )
            self.remove_edges_from(edges_to_remove)
            nx.relabel_nodes(
                G=self, mapping={temp_1: new_node, temp_2: new_node}, copy=False
            )

            # Some nodes have been removed, we need to update the
            # mergeable list to point to new nodes if required
            temp_list = nodes_to_merge.copy()
            for pair in temp_list:
                if nodes[1] in pair:
                    new_pair = (
                        # the new node it will be merged to
                        new_node,
                        # the other node of the pair
                        pair[pair.index(nodes[1]) - 1],
                    )
                    nodes_to_merge.remove(pair)
                    if new_pair[0] != new_pair[1]:
                        nodes_to_merge.append(new_pair)

    def dot_graph(self, edge_info: str = "probability") -> pdp.Dot:
        """Returns Dot graph representation of the CEG.
        :param edge_info: Optional - Chooses which summary measure to be displayed on edges. Defaults to "count". Options: ["count", "prior", "posterior", "probability", "none"]
        :type edge_info: str
        
        :return: A graphviz Dot representation of the graph.
        :rtype: pydotplus.Dot"""
        return self._generate_dot_graph(edge_info=edge_info)

    def _generate_dot_graph(self, edge_info="probability"):
        graph = pdp.Dot(graph_type="digraph", rankdir="LR")
        if edge_info in self._edge_attributes:
            edge_info_dict = nx.get_edge_attributes(self, edge_info)
        else:
            edge_info_dict = nx.get_edge_attributes(self, "probability")

        for (src, dst, label), attribute in edge_info_dict.items():
            if edge_info == "count":
                edge_details = str(label) + "\n" + str(attribute)
            elif edge_info == "none":
                edge_details = str(label)
            else:
                edge_details = f"{label}\n{float(attribute):.2f}"

            if dst[1:] == "_infinity":
                dst = f"{self.node_prefix}&infin;"

            graph.add_edge(
                pdp.Edge(
                    src,
                    dst,
                    label=edge_details,
                    labelfontcolor="#009933",
                    fontsize="10.0",
                    color="black",
                )
            )
        nodes = list(nx.topological_sort(self))
        for node in nodes:
            try:
                fill_colour = self.nodes[node]["colour"]
            except KeyError:
                fill_colour = "white"
            if node[1:] == "_infinity":
                node = f"{self.node_prefix}&infin;"
            label = "<" + node[0] + "<SUB>" + node[1:] + "</SUB>" + ">"
            graph.add_node(
                pdp.Node(name=node, label=label, style="filled", fillcolor=fill_colour)
            )
        return graph

    def create_figure(
        self,
        filename=None,
        edge_info: str = "probability",
    ) -> Union[Image, None]:
        """Draws the coloured chain event graph for the staged_tree.

        :param filename: Optional - When provided, file is saved to the filename,
            local to the current working directory.
            e.g. if filename = "output/ceg.svg", the file will be saved to:
            cwd/output/ceg.svg
            Otherwise, if function is called inside an interactive notebook, image
            will be displayed in the notebook, even if filename is omitted.

            Supports any filetype that graphviz supports. e.g: "ceg.png" or
            "ceg.svg" etc.

        :type filename: str

        :param edge_info: Optional - Chooses which summary measure to be displayed on edges. Defaults to "count". Options: ["count", "prior", "posterior", "probability", "none"]
        :type edge_info: str

        :return: The event tree Image object.
        :rtype: IPython.display.Image or None
        """
        graph = self.dot_graph(edge_info=edge_info)

        if get_ipython() is not None:
            graph_image = Image(graph.create_png())  # pylint: disable=no-member
            return graph_image
        elif filename:
            filename, filetype = generate_filename_and_mkdir(filename)
            graph.write(str(filename), format=filetype)
        else:
            raise RuntimeError(
                "Cannot display graph in notebook. "
                "Please provide a filename to save the graph to."
            )

        return None

    def _trim_leaves_from_graph(self):
        """Trims all the leaves from the graph, and points each incoming
        edge to the sink node."""
        # Create new CEG sink node
        self.add_node(self.sink, colour="lightgrey")
        outgoing_edges = deepcopy(self.succ).items()
        # Check to see if any nodes have no outgoing edges.
        mapping = {}
        for node, out_edges in outgoing_edges:
            if not out_edges and node != self.sink:
                mapping[node] = self.sink

        nx.relabel_nodes(self, mapping, copy=False)

    def _update_distances_to_sink(self) -> None:
        """
        Iterates through the graph until it finds the root node.
        For each node, it determines the maximum number of edges
        from that node to the sink node.
        """
        max_dist = "max_dist_to_sink"
        self.nodes[self.sink][max_dist] = 0
        node_queue = [self.sink]

        while node_queue != [self.root]:
            node = node_queue.pop(0)
            for pred in self.predecessors(node):
                max_dist_to_sink = set()
                for succ in self.successors(pred):
                    try:
                        max_dist_to_sink.add(self.nodes[succ][max_dist])
                        self.nodes[pred][max_dist] = max(max_dist_to_sink) + 1
                    except KeyError:
                        break

                if pred not in node_queue:
                    node_queue.append(pred)

    def _gen_nodes_with_increasing_distance(self, start=0) -> list:
        """Generates nodes that are either the same or further
        from the sink node than the last node generated."""
        max_dists = nx.get_node_attributes(self, "max_dist_to_sink")
        distance_dict: Mapping[int, Iterable[str]] = {}
        for node, distance in max_dists.items():
            dist_list: List = distance_dict.setdefault(distance, [])
            dist_list.append(node)

        for dist in range(0, max(distance_dict) + 1):
            nodes = distance_dict.get(dist)
            if dist >= start and nodes is not None:
                yield nodes

    def _relabel_nodes(self):
        """Relabels nodes whilst maintaining ordering."""

        all_nodes: List[str] = list(self.nodes)
        all_nodes.remove(self.root)
        all_nodes.remove(self.sink)

        nodes_num = [int(node.split(self.staged_root[0])[1]) for node in all_nodes]
        nodes_num.sort()
        nodes_to_rename = [f"{self.staged_root[0]}{num}" for num in nodes_num]
        renamed_nodes = [
            f"{self.node_prefix}{i + 1}" for i in range(len(nodes_to_rename))
        ]
        node_mapping = dict(zip(nodes_to_rename, renamed_nodes))
        nx.relabel_nodes(self, node_mapping, copy=False)

    def _merge_and_add_edges(
        self,
        new_node: str,
        node_1: str,
        node_2: str,
    ) -> List[Tuple]:
        """Merges outgoing edges of two nodes so that the two nodes can be
        merged."""
        old_edges_to_remove = []
        for succ, t1_edge_dict in self.succ[node_1].items():
            edge_labels = list(t1_edge_dict.keys())
            while edge_labels:
                label = edge_labels.pop(0)
                n1_edge_data = t1_edge_dict[label]
                n2_edge_data = self.succ[node_2][succ][label]

                new_edge_data = _merge_edge_data(
                    edge_1=n1_edge_data,
                    edge_2=n2_edge_data,
                )
                self.add_edge(
                    u_for_edge=new_node,
                    v_for_edge=succ,
                    key=label,
                    **new_edge_data,
                )
                old_edges_to_remove.extend(
                    [(node_1, succ, label), (node_2, succ, label)]
                )

        return old_edges_to_remove

    def _check_nodes_can_be_merged(self, node_1, node_2) -> bool:
        """Determine if the two nodes are able to be merged."""
        have_same_successor_nodes = set(self.adj[node_1].keys()) == set(
            self.adj[node_2].keys()
        )

        if have_same_successor_nodes:
            have_same_outgoing_edges = True
            n1_adj = self.succ[node_1]
            for succ_node in list(n1_adj.keys()):
                n1_edges = self.succ[node_1][succ_node]
                n2_edges = self.succ[node_2][succ_node]

                n2_edge_labels = list(n2_edges.keys())

                for label in n1_edges.keys():
                    if label not in n2_edge_labels:
                        have_same_outgoing_edges &= False
                        break
                    have_same_outgoing_edges &= True
        else:
            have_same_outgoing_edges = False

        try:
            in_same_stage = self.nodes[node_1]["stage"] == self.nodes[node_2]["stage"]
        except KeyError:
            in_same_stage = False

        return in_same_stage and (
            have_same_successor_nodes and have_same_outgoing_edges
        )


def _merge_edge_data(
    edge_1: Dict[str, Any],
    edge_2: Dict[str, Any],
) -> Dict[str, Any]:
    """Merges the counts, priors, and posteriors of two edges."""
    new_edge_data = {}
    edge = edge_1 if len(edge_1) > len(edge_2) else edge_2
    for key in edge:
        if key == "probability":
            new_edge_data[key] = edge_1.get(key, 1)
        else:
            new_edge_data[key] = edge_1.get(key, 0) + edge_2.get(key, 0)
    return new_edge_data
