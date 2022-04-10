import itertools
import re
from typing import Dict, Mapping
import unittest
import networkx as nx
import pandas as pd
import pytest_mock
from src.cegpy import StagedTree, ChainEventGraph
from src.cegpy.graphs.ceg import (
    _merge_edge_data,
    _relabel_nodes,
    _merge_and_add_edges,
    _trim_leaves_from_graph,
    _update_distances_to_sink,
    _gen_nodes_with_increasing_distance,
    _check_nodes_can_be_merged,
)
from pathlib import Path


class TestMockedCEGMethods():
    def setup(self):
        self.node_prefix = 'w'
        self.sink_suffix = '&infin;'
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.st = StagedTree(
            dataframe=pd.read_excel(df_path),
            name="medical_staged"
        )
        self.st.calculate_AHC_transitions()

    def test_generate_argument(self, mocker: pytest_mock.mocker):
        """When ChainEventGraph called with generate, the .generate()
        method is called."""
        mocker.patch('src.cegpy.graphs.ceg.ChainEventGraph.generate')
        ceg = ChainEventGraph(self.st, generate=True)
        ceg.generate.assert_called_once()


class TestUnitCEG(unittest.TestCase):
    def setUp(self):
        self.node_prefix = 'w'
        self.sink_suffix = '&infin;'
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.st = StagedTree(
            dataframe=pd.read_excel(df_path),
            name="medical_staged"
        )
        self.st.calculate_AHC_transitions()

    def test_stages_property(self):
        """Stages is a mapping of stage names to lists of nodes"""
        ceg = ChainEventGraph(self.st)
        node_stage_mapping: Mapping = (
            dict(ceg.nodes(data='stage', default=None))
        )
        stages = ceg.stages
        self.assertEqual(
            len(ceg.nodes),
            sum(len(nodes) for nodes in stages.values())
        )
        for node, stage in node_stage_mapping.items():
            self.assertIn(node, stages[stage])


class TestCEGHelpersTestCases(unittest.TestCase):
    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = [
            'w0', 'w1', 'w2', 'w3', 'w4', 'w&infin;'
        ]
        self.init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w3', 'c'),
            ('w2', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.ceg = ChainEventGraph(self.graph)

    def test_merge_edge_data(self):
        """Edges are merged"""
        edge_1 = dict(
            zip(
                ['count', 'prior', 'posterior', 'probability'],
                [250, 0.5, 250, 0.8],
            )
        )
        edge_2 = dict(
            zip(
                ['count', 'prior', 'posterior', 'probability'],
                [550, 25, 0.4, 0.9],
            )
        )

        new_edge_dict = _merge_edge_data(edge_1=edge_1, edge_2=edge_2)
        self.assert_edges_merged(new_edge_dict, edge_1, edge_2)

    def test_merge_edge_data_with_missing(self):
        """Edges are merged even when some data is missing in one edge."""
        edge_1 = dict(
            zip(
                ['count', 'prior'],
                [250, 0.5],
            )
        )
        edge_2 = dict(
            zip(
                ['count', 'prior', 'posterior'],
                [550, 25, 0.4],
            )
        )

        new_edge_dict = _merge_edge_data(edge_1=edge_1, edge_2=edge_2)
        self.assert_edges_merged(new_edge_dict, edge_1, edge_2)

    def assert_edges_merged(self, new_edge: Dict, edge_1: Dict, edge_2: Dict):
        """Edges were merged successfully."""

        assert (
            set(new_edge.keys()) == set(
                edge_1.keys()
            ).union(set(edge_2.keys()))
        ), "Edges do not have the same keys."

        for key, value in new_edge.items():
            if key == "probability":
                assert (
                    value == edge_1.get(key)
                ), "Probability shouldn't be summed."
            else:
                assert (
                    value == edge_1.get(key, 0) + edge_2.get(key, 0)
                ), f"{key} not merged. Merged value: {value}"

    def test_relabel_nodes(self):
        """Relabel nodes successfully renames all the nodes."""
        _relabel_nodes(self.ceg)
        node_pattern = r"^w([0-9]+)|w(&infin;)$"
        prog = re.compile(node_pattern)
        for node in self.ceg.nodes:
            result = prog.match(node)
            assert result is not None, "Node does not match expected format."

    def test_merge_outgoing_edges(self):
        """Outgoing edges are merged."""
        edges_to_add = [
            {
                "src": "s1", "dst": "s3", "key": "event_1",
                "counts": 5, "priors": 0.2, "posteriors": 1,
                "probability": 0.8,
            },
            {
                "src": "s1", "dst": "s4", "key": "event_2",
                "counts": 10, "priors": 0.3, "posteriors": 5,
                "probability": 0.2,
            },
            {
                "src": "s2", "dst": "s3", "key": "event_1",
                "counts": 11, "priors": 0.5, "posteriors": 2,
                "probability": 0.8,
            },
            {
                "src": "s2", "dst": "s4", "key": "event_2",
                "counts": 6, "priors": 0.9, "posteriors": 3,
                "probability": 0.2,
            },
        ]
        for edge in edges_to_add:
            self.ceg.add_edge(
                u_for_edge=edge["src"],
                v_for_edge=edge["dst"],
                key=edge["key"],
                counts=edge["counts"],
                priors=edge["priors"],
                posteriors=edge["posteriors"],
            )
        self.ceg.remove_edges_from(
            [
                ("s1", "s3", "c"),
                ("s1", "s4", "d"),
                ("s2", "s3", "c"),
                ("s2", "s4", "d"),
            ]
        )

        expected = [
            (edge["src"], edge["dst"], edge["key"]) for edge in edges_to_add
        ]
        actual = (
            _merge_and_add_edges(self.ceg, "s99", "s1", "s2")
        )
        for edge in expected:
            assert (
                edge in actual
            ), f"Expected {edge} in return value, but it was not found."
        assert (
            len(expected) == len(actual)
        ), "Actual number of edges does not match expected number of edges."


class TestNodesCanBeMerged(unittest.TestCase):
    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = [
            'w0', 'w1', 'w2', 'w3', 'w4', 'w&infin;'
        ]
        self.graph.add_nodes_from(self.init_nodes)

    def test_check_nodes_can_be_merged(self):
        """Nodes can be merged."""
        init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w3', 'c'),
            ('w2', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        assert (
            _check_nodes_can_be_merged(
                ceg, "w1", "w2"
            )
        ), "Nodes should be mergeable."

    def test_nodes_not_in_same_stage_cannot_be_merged(self):
        """Nodes not in same stage cannot be merged"""
        init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w3', 'c'),
            ('w2', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph)
        ceg.nodes["w1"]["stage"] = 1
        ceg.nodes["w2"]["stage"] = 2
        assert (
            not _check_nodes_can_be_merged(
                ceg, "w1", "w2"
            )
        ), "Nodes should not be mergeable."

    def test_nodes_with_different_successor_nodes(self):
        """Nodes with different successor nodes won't be merged."""
        init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w&infin;', 'c'),
            ('w2', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        assert (
            not _check_nodes_can_be_merged(
                ceg, "w1", "w2"
            )
        ), "Nodes should not be mergeable."

    def test_nodes_with_different_outgoing_edges(self):
        """Nodes with different outgoing edges won't be merged."""
        init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w3', 'g'),
            ('w2', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        assert (
            not _check_nodes_can_be_merged(
                ceg, "w1", "w2"
            )
        ), "Nodes should not be mergeable."

    def test_merging_of_nodes(self):
        """The nodes are merged, and all edges are merged too."""
        init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w3', 'c'),
            ('w2', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph)
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        ceg._merge_nodes({("w1", "w2")})
        expected_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w1', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w3', 'w&infin;', 'e'),
            ('w4', 'w&infin;', 'f'),
        ]
        for edge in expected_edges:
            self.assertIn(edge, list(ceg.edges))

    def test_merging_of_three_nodes(self):
        """The three nodes are merged, and all edges are merged too."""
        self.graph.add_node("w5")
        init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w0', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w1', 'w5', 'e'),
            ('w2', 'w4', 'd'),
            ('w2', 'w5', 'e'),
            ('w3', 'w4', 'd'),
            ('w3', 'w5', 'e'),
            ('w4', 'w&infin;', 'f'),
            ('w5', 'w&infin;', 'g'),
        ]
        self.graph.add_edges_from(init_edges)
        ceg = ChainEventGraph(self.graph)

        nodes_to_merge = {"w1", "w2", "w3"}
        ceg.nodes["w1"]["stage"] = 2
        ceg.nodes["w2"]["stage"] = 2
        ceg.nodes["w3"]["stage"] = 2
        ceg._merge_nodes({("w1", "w2"), ("w2", "w3"), ("w1", "w3")})
        nodes_post_merge = set(ceg.nodes)
        merged_node = nodes_post_merge.intersection(nodes_to_merge).pop()
        expected_edges = [
            ('w0', merged_node, 'a'),
            ('w0', merged_node, 'b'),
            ('w0', merged_node, 'c'),
            (merged_node, 'w4', 'd'),
            (merged_node, 'w5', 'e'),
            ('w4', 'w&infin;', 'f'),
            ('w5', 'w&infin;', 'g'),
        ]

        for edge in expected_edges:
            self.assertIn(edge, list(ceg.edges))

        self.assertEqual(len(list(ceg.edges)), len(expected_edges))


class TestTrimLeavesFromGraph:
    def setup(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = [
            'w0', 'w1', 'w2', 'w3', 'w4'
        ]
        self.init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'c'),
            ('w1', 'w4', 'd'),
            ('w2', 'w3', 'c'),
            ('w2', 'w4', 'd'),
        ]
        self.leaves = ["s3", "s4"]
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.ceg = ChainEventGraph(self.graph)

    def test_leaves_trimmed_from_graph(self) -> None:
        """Leaves are trimmed from the graph."""
        _trim_leaves_from_graph(self.ceg)
        for leaf in self.leaves:
            try:
                self.ceg.nodes[leaf]
                leaf_removed = False
            except KeyError:
                leaf_removed = True

            assert leaf_removed, "Leaf was not removed."

            for edge_list_key in self.ceg.edges.keys():
                assert (
                    edge_list_key[1] != leaf
                ), f"Edge still pointing to leaf: {leaf}"


class TestDistanceToSink:
    def setup(self):
        self.graph = nx.MultiDiGraph()
        self.init_nodes = [
            'w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w&infin;'
        ]
        self.init_edges = [
            ('w0', 'w1', 'a'),
            ('w0', 'w2', 'b'),
            ('w1', 'w3', 'e'),
            ('w1', 'w4', 'e'),
            ('w2', 'w&infin;', 'c'),
            ('w3', 'w&infin;', 'd'),
            ('w4', 'w5', 'c'),
            ('w5', 'w&infin;', 'd'),
        ]
        self.graph.add_nodes_from(self.init_nodes)
        self.graph.add_edges_from(self.init_edges)
        self.ceg = ChainEventGraph(self.graph)

    def test_update_distances_to_sink(self) -> None:
        """Distance to sink is always max length of paths to sink."""
        def check_distances():
            actual_node_dists = (
                nx.get_node_attributes(self.ceg, 'max_dist_to_sink')
            )
            for node, distance in actual_node_dists.items():
                assert distance == expected_node_dists[node]

        expected_node_dists = {
            "w0": 4,
            "w1": 3,
            "w2": 1,
            "w3": 1,
            "w4": 2,
            "w5": 1,
            "w&infin;": 0
        }
        _update_distances_to_sink(self.ceg)
        check_distances()

        # Add another edge to the dictionary, to show that the path is max,
        # and not min distance to sink
        self.ceg.add_edge("w1", self.ceg.sink_node)
        self.ceg.add_edge("w4", self.ceg.sink_node)
        _update_distances_to_sink(self.ceg)
        check_distances()

    def test_gen_nodes_with_increasing_distance(self) -> None:
        expected_nodes = {
            0: [self.ceg.sink_node],
            1: ["w2", "w3", "w5"],
            2: ["w4"],
            3: ["w1"],
            4: [self.ceg.root_node]
        }
        for dist, nodes in expected_nodes.items():
            for node in nodes:
                self.ceg.nodes[node]["max_dist_to_sink"] = dist

        nodes_gen = _gen_nodes_with_increasing_distance(self.ceg, start=0)

        for nodes in range(len(expected_nodes)):
            expected_node_list = expected_nodes[nodes]
            actual_node_list = next(nodes_gen)
            assert actual_node_list.sort() == expected_node_list.sort()
