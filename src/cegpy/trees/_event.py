"""cegpy event tree."""

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Tuple, Union
import textwrap
import numpy as np
import pydotplus as pdp
from IPython.display import Image
from IPython import get_ipython
import pandas as pd
import networkx as nx
from cegpy.utilities._util import (
    generate_filename_and_mkdir,
    create_sampling_zeros,
)


class EventTree(nx.MultiDiGraph):
    """
    This class extends the NetworkX MultiDiGraph class to allow the creation
    of event tree representations of data.

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

    _sampling_zero_paths: Optional[List[Tuple]] = None
    _sorted_paths: Mapping[Tuple[str], int]
    _edge_attributes: List = ["count"]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sampling_zero_paths=None,
        var_order=None,
        struct_missing_label=None,
        missing_label=None,
        complete_case=False,
    ) -> None:
        # Checking argument inputs are sensible
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("The dataframe parameter must be a pandas.DataFrame")
        # Initialise Networkx DiGraph class
        incoming_graph_data = None
        attr = dict()
        super().__init__(incoming_graph_data, **attr)

        if struct_missing_label is not None and not isinstance(
            struct_missing_label, str
        ):
            raise ValueError("struct_missing_label should be a string")

        if missing_label is not None and not isinstance(missing_label, str):
            raise ValueError("missing_label should be a string")

        if not isinstance(complete_case, bool):
            raise ValueError("complete_case should be a boolean")

        self.sampling_zeros = sampling_zero_paths

        # Dealing with structural and non-structural...
        # ... missing value labels
        if struct_missing_label is not None:
            dataframe.replace(
                struct_missing_label,
                "",
                inplace=True,
            )

        if missing_label is not None:
            if complete_case is True:
                dataframe.replace(
                    missing_label,
                    "missing",
                    inplace=True,
                )
                rows_to_delete = np.where(dataframe == "missing")[0].tolist()
                dataframe.drop(
                    rows_to_delete,
                    axis=0,
                    inplace=True,
                )
                dataframe.reset_index(drop=True, inplace=True)
            else:
                dataframe.replace(
                    missing_label,
                    "missing",
                    inplace=True,
                )

        # Paths sorted alphabetically in order of length
        self._sorted_paths = defaultdict(int)

        # pandas dataframe passed via parameters
        self.dataframe = (
            dataframe[var_order].astype(str)
            if var_order is not None
            else dataframe.astype(str)
        )

        self._construct_event_tree()

    @property
    def root(self) -> str:
        """
        :return: The name of the root node of the event tree, currently hard coded to 's0'.
        :rtype: str
        """
        return "s0"

    @property
    def variables(self) -> List:
        """
        :return: The column headers of the dataset.
        :rtype: List[str]
        """
        variables = list(self.dataframe.columns)
        return variables

    @property
    def sampling_zeros(self) -> Union[List[Tuple[str]], None]:
        """
        Setting this property will apply sampling zero paths to the tree.
        If different to previous value, the event tree will be regenerated.

        :return: Sampling zero paths provided by the user.

        :rtype: List[Tuple[str]] or None
        """
        return self._sampling_zero_paths

    @sampling_zeros.setter
    def sampling_zeros(self, sz_paths):
        if sz_paths is None:
            self._sampling_zero_paths = None
        else:
            # checkes if the user has inputted sz paths correctly
            sz_paths = self._check_sampling_zero_paths_param(sz_paths)

            if sz_paths:
                self._sampling_zero_paths = sz_paths
            else:
                raise ValueError(
                    "Parameter 'sampling_zero_paths' not in expected format. "
                    "Should be a list of tuples like so:\n"
                    "[('edge_1',), ('edge_1', 'edge_2'), ...]"
                )

    @property
    def situations(self) -> List[str]:
        """
        :return: The situations of the tree (non-leaf nodes).
        :rtype: List[str]
        """
        return [node for node, out_degree in self.out_degree if out_degree != 0]

    @property
    def leaves(self) -> List[str]:
        """
        :return: The leaves of the tree.
        :rtype: List[str]
        """
        return [node for node, out_degree in self.out_degree if out_degree == 0]

    @property
    def edge_counts(self) -> Dict:
        """
        The counts along edges all edges in the tree, where edges are a
        Tuple like so: ("source_node", "destination_node", "edge_label").

        :return: A mapping of edges to their counts.

        :rtype: Dict[Tuple[str], Int]
        """
        return nx.get_edge_attributes(self, "count")

    @property
    def categories_per_variable(self) -> Dict:
        """
        The number of unique categories/levels for each variable
        (column headings in dataframe).

        :return: A mapping of variables to the number of unique categories/levels.
        :rtype: Dict[str, Int]
        """

        categories_to_ignore = {"N/A", "NA", "n/a", "na", "NAN", "nan"}
        catagories_per_variable = {}

        for var in self.variables:
            categories = set(self.dataframe[var].unique().tolist())
            # remove nan with pandas
            pd_filtered_categories = {x for x in categories if pd.notna(x)}

            # remove any string nans that might have made it in.
            filtered_cats = pd_filtered_categories - categories_to_ignore

            catagories_per_variable[var] = len(filtered_cats)

        return catagories_per_variable

    def dot_graph(self, edge_info: str = "count") -> pdp.Dot:
        """Returns Dot graph representation of the event tree.
        :param edge_info: Optional - Chooses which summary measure to be displayed on edges. Defaults to "count". Options: ["count", "none"]
        :type edge_info: str

        :type edge_info: str
        :return: A graphviz Dot representation of the graph.
        :rtype: pydotplus.Dot
        """
        return self._generate_dot_graph(fill_colour="lightgrey", edge_info=edge_info)

    def _generate_dot_graph(self, fill_colour=None, edge_info="count"):
        node_list = list(self)
        graph = pdp.Dot(graph_type="digraph", rankdir="LR")
        if edge_info in self._edge_attributes:
            edge_info_dict = nx.get_edge_attributes(self, edge_info)
        else:
            edge_info_dict = nx.get_edge_attributes(self, "count")

        for edge, attribute in edge_info_dict.items():
            if edge_info == "count":
                edge_details = str(edge[2]) + "\n" + str(attribute)
            elif edge_info == "none":
                edge_details = str(edge[2])
            else:
                edge_details = f"{edge[2]}\n{float(attribute):.2f}"

            graph.add_edge(
                pdp.Edge(
                    edge[0],
                    edge[1],
                    label=edge_details,
                    labelfontcolor="#009933",
                    fontsize="10.0",
                    color="black",
                )
            )
        for node in node_list:
            if fill_colour is None:
                try:
                    fill_node_colour = self.nodes[node]["colour"]
                except KeyError:
                    fill_node_colour = "lightgrey"
            else:
                fill_node_colour = fill_colour
            label = "<" + node[0] + "<SUB>" + node[1:] + "</SUB>" + ">"
            graph.add_node(
                pdp.Node(
                    name=node, label=label, style="filled", fillcolor=fill_node_colour
                )
            )
        return graph

    def _create_figure(self, graph: pdp.Dot, filename: str):
        """Draws the event tree for the process described by the dataset,
        and saves it to "<filename>.filetype". Supports any filetype that
        graphviz supports. e.g: "event_tree.png" or "event_tree.svg" etc.
        """

        if get_ipython() is not None:
            graph_image = Image(graph.create_png())
        elif filename:
            filename, filetype = generate_filename_and_mkdir(filename)
            graph.write(str(filename), format=filetype)
            graph_image = None
        else:
            raise RuntimeError(
                "Cannot display graph in notebook. "
                "Please provide a filename to save the graph to."
            )

        return graph_image

    def create_figure(
        self, filename=None, edge_info: str = "count"
    ) -> Union[Image, None]:
        """Creates event tree from the dataframe.

        :param filename: Optional - When provided, file is saved to the filename,
            local to the current working directory.
            e.g. if filename = "output/event_tree.svg", the file will be saved to:
            cwd/output/event_tree.svg
            Otherwise, if function is called inside an interactive notebook, image
            will be displayed in the notebook, even if filename is omitted.
            Supports any filetype that graphviz supports. e.g: "event_tree.png" or
            "event_tree.svg" etc.

        :type filename: str

        :param edge_info: Optional - Chooses which summary measure to be displayed on edges. Defaults to "count". Options: ["count", "none"]
        :type edge_info: str

        :return: The event tree Image object.
        :rtype: IPython.display.Image or None
        """
        return self._create_figure(self.dot_graph(edge_info=edge_info), filename)

    def _create_unsorted_paths_dict(self) -> defaultdict:
        """Creates and populates a dictionary of all paths provided in the dataframe,
        in the order in which they are given."""
        unsorted_paths = defaultdict(int)

        for variable_number in range(0, len(self.variables)):
            dataframe_upto_variable = self.dataframe.loc[
                :, self.variables[0 : variable_number + 1]
            ]

            for row in dataframe_upto_variable.itertuples():
                row = row[1:]
                new_row = [
                    edge_label
                    for edge_label in row
                    if edge_label != np.nan
                    and str(edge_label) != "NaN"
                    and str(edge_label) != "nan"
                    and edge_label != ""
                ]
                new_row = tuple(new_row)

                # checking if the last edge label in row was nan. That would
                # result in double counting nan must be identified as string
                if (
                    row[-1] != np.nan
                    and str(row[-1]) != "NaN"
                    and str(row[-1]) != "nan"
                    and row[-1] != ""
                ):
                    unsorted_paths[new_row] += 1

        return unsorted_paths

    def _create_path_dict_entries(self):
        """Create path dict entries for each path, including the
        sampling zero paths if any.
        Each path is an ordered sequence of edge labels starting
        from the root.
        The keys in the dict are ordered alphabetically.
        Also calls the method self._sampling_zeros to ensure
        manually added path format is correct.
        Added functionality to remove NaN/null edge labels
        assuming they are structural zeroes"""
        unsorted_paths = self._create_unsorted_paths_dict()

        if self.sampling_zeros is not None:
            unsorted_paths = create_sampling_zeros(self.sampling_zeros, unsorted_paths)

        depth = len(max(list(unsorted_paths.keys()), key=len))
        keys_of_list = list(unsorted_paths.keys())
        sorted_keys = []
        for deep in range(0, depth + 1):
            unsorted_mini_list = [key for key in keys_of_list if len(key) == deep]
            sorted_keys = sorted_keys + sorted(unsorted_mini_list)

        for key in sorted_keys:
            self._sorted_paths[key] = unsorted_paths[key]

        node_list = self._create_node_list_from_paths(self._sorted_paths)
        self.add_nodes_from(node_list)

    def _check_sampling_zero_paths_param(self, sampling_zero_paths) -> List:
        """Check param 'sampling_zero_paths' is in the correct format"""
        coerced_sampling_zero_paths = []
        for tup in sampling_zero_paths:
            if not isinstance(tup, tuple):
                return None
            coerced_sampling_zero_paths.append(tuple([str(elem) for elem in tup]))

        return coerced_sampling_zero_paths

    def _create_node_list_from_paths(self, paths) -> list:
        """Creates list of all nodes: includes root, situations, leaves"""
        node_list = [self.root]

        for node_number, _ in enumerate(list(paths.keys()), start=1):
            node_list.append(f"s{node_number}")

        return node_list

    def _construct_event_tree(self):
        """Constructs event_tree DiGraph.
        Takes the paths, and adds all the nodes and edges to the Graph"""

        self._create_path_dict_entries()
        # Taking a list of a networkx graph object (self) provides a list
        # of all the nodes
        node_list = list(self)

        # Work through the sorted paths list to build the event tree.
        edge_labels_list = ["root"]
        for path, count in list(self._sorted_paths.items()):
            path = list(path)
            edge_labels_list.append(path)
            if path[:-1] in edge_labels_list:
                path_edge_comes_from = edge_labels_list.index(path[:-1])
                self.add_edge(
                    u_for_edge=node_list[path_edge_comes_from],
                    v_for_edge=node_list[edge_labels_list.index(path)],
                    key=path[-1],
                    count=count,
                )
            else:
                self.add_edge(
                    u_for_edge=node_list[0],
                    v_for_edge=node_list[edge_labels_list.index(path)],
                    key=path[-1],
                    count=count,
                )
