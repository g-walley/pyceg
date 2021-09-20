from ..src.cegpy.trees.event import EventTree
from collections import defaultdict
import pandas as pd
# from ceg_util import CegUtil as util
from pathlib import Path
import re


class TestEventTree():
    def setup(self):
        df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')

        self.df = pd.read_excel(df_path)
        self.et = EventTree(dataframe=self.df)
        self.node_format = re.compile('^s\\d\\d*$')

    def test_check_sampling_zero_paths_param(self) -> None:
        """Tests the function that is checking the sampling zero paths param"""
        szp = [('Medium',), ('Medium', 'High')]
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) == szp
        szp = [1, 2, 3, 4]
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) is None

        szp = [('path', 'to'), (123, 'something'), 'path/to']
        assert self.et._EventTree__check_sampling_zero_paths_param(szp) is None

    def test_check_sampling_zero_get_and_set(self) -> None:
        """Tests the functions that set and get the sampling zeros"""
        assert self.et.sampling_zeros is None

        szp = [('Medium',), ('Medium', 'High')]
        self.et.sampling_zeros = szp
        assert self.et.sampling_zeros == szp

    def test_create_node_list_from_paths(self) -> None:
        paths = defaultdict(int)
        paths[('path',)] += 1
        paths[('path', 'to')] += 1
        paths[('path', 'away')] += 1
        paths[('road',)] += 1
        paths[('road', 'to')] += 1
        paths[('road', 'away')] += 1

        # code being tested:
        node_list = self.et._EventTree__create_node_list_from_paths(paths)

        print(node_list)
        assert len(list(paths.keys())) + 1 == len(node_list)
        assert node_list[0] == 's0'
        assert node_list[-1] == 's%d' % (len(node_list) - 1)

    def test_construct_event_tree(self) -> None:
        """Tests the construction of an event tree from a set of paths,
        nodes, and """
        EXPECTED_NODE_COUNT = 45
        assert len(self.et) == EXPECTED_NODE_COUNT
        assert len(self.et.edges) == EXPECTED_NODE_COUNT - 1
        edge_counts = self.et.edge_counts

        assert len(edge_counts) == EXPECTED_NODE_COUNT - 1
        for _, count in edge_counts.items():
            assert isinstance(count, int)

    def test_get_functions_producing_expected_data(self) -> None:
        edges = list(self.et.edges)
        assert isinstance(edges, list)
        for edge in edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 3
            assert isinstance(edge[0], str)
            assert isinstance(edge[1], str)
            assert isinstance(edge[2], str)

        check_list_contains_strings(list(self.et))
        check_list_contains_strings(self.et.situations)
        check_list_contains_strings(self.et.leaves)

        edge_counts = self.et.edge_counts
        print(edge_counts)
        assert isinstance(edge_counts, dict)
        for edge, count in edge_counts.items():
            assert isinstance(edge, tuple)
            for node in edge:
                assert isinstance(node, str)
            assert isinstance(count, int)


class TestIntegration():
    def setup(self):
        # stratified dataset
        med_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/medical_dm_modified.xlsx')
        self.med_s_z_paths = None
        self.med_df = pd.read_excel(med_df_path)
        self.med_et = EventTree(
            dataframe=self.med_df,
            sampling_zero_paths=self.med_s_z_paths
        )

        # non-stratified dataset
        fall_df_path = Path(__file__).resolve(
            ).parent.parent.joinpath(
            'data/Falls_Data.xlsx')
        self.fall_s_z_paths = None
        self.fall_df = pd.read_excel(fall_df_path)
        self.fall_et = EventTree(
            dataframe=self.fall_df,
            sampling_zero_path=self.fall_s_z_paths
        )

    def test_categories_per_variable(self) -> None:
        expected_med_cats_per_var = {
            "Classification": 2,
            "Group": 3,
            "Difficulty": 2,
            "Response": 2,
        }
        actual_med_cats_per_var = self.med_et.categories_per_variable
        assert expected_med_cats_per_var == actual_med_cats_per_var

        expected_fall_cats_per_var = {
            "HousingAssessment": 4,
            "Risk": 2,
            "Treatment": 3,
            "Fall": 2,
        }
        actual_fall_cats_per_var = self.fall_et.categories_per_variable
        assert expected_fall_cats_per_var == actual_fall_cats_per_var


def check_list_contains_strings(str_list) -> bool:
    assert isinstance(str_list, list)
    for elem in str_list:
        assert isinstance(elem, str)
