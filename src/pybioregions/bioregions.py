from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import geodatasets as gds
import networkx as nx

# import datatable as dt
from scipy.stats import zscore
from sklearn import cluster as skl_cluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics as skl_metrics
from functools import lru_cache
from scipy import stats
from collections.abc import Sequence

from libpysal import weights
import topojson as tp
from topojson import utils as tp_utils
import shapely
from shapely import geometry, ops
import itertools as it

import infomap
import tempfile
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import seaborn as sns
from coloraide import Color

rng = np.random.default_rng(seed=1)


def plogp(p, p_tot=1):
    if len(p) == 0 or p_tot < 1e-7:
        return 0.0
    p = np.asarray(p) / p_tot
    # Note: Need to define output array as values will be uninitialized where condition is not met
    return p * np.log2(p, where=p > 0, out=np.zeros_like(p, dtype=float))


def entropy(p):
    p_tot = np.sum(p)
    return -np.sum(plogp(p, p_tot))


def perplexity(p):
    return 2 ** entropy(p)


@lru_cache()
def get_pmf(k, pmf="uniform", shape=2):
    """
    Get probability mass function for `k` categories from a uniform or
    Zipfian distribution with different shape

    Note:
    Zipfian with shape 0 equals uniform
    """
    if pmf == "uniform":
        return np.array([1 / k] * k)
    if pmf == "zipf":
        return stats.zipfian(a=shape, n=k).pmf(list(range(1, k + 1)))
    raise Exception(f"Probability distribution '{pmf}' not recognized")


def sample_multinomial(n, k, pmf="uniform", shape=2):
    """Add ``n`` coins in ``k`` bins with probability mass function ``pmf``."""
    return rng.multinomial(n, get_pmf(k, pmf=pmf, shape=shape))


def get_first_node(G, **kwargs):
    return next(iter(G.nodes(**kwargs)))


def is_np_array(a):
    return isinstance(a, np.ndarray)


def is_equal(elements):
    if is_np_array(elements):
        return all(elements == elements[0])
    return elements.count(elements[0]) == len(elements)


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def get_N(G):
    """Get number of nodes in ``G``"""
    return nx.number_of_nodes(G)


def get_E(G):
    """Get number of edges in ``G``"""
    return nx.number_of_edges(G)


def is_array_like(a):
    """list or numpy array"""
    return isinstance(a, (Sequence, np.ndarray, pd.Series))


def minmax(a):
    return np.min(a), np.max(a)


def clamp(x, vmax=None, vmin=None, where=True):
    """
    Clamp array. Replaces and simplifies np.clip, np.fmin and np.fmax

    Example:
    x = np.linspace(0,2)
    plt.plot(x, x*(2-x), label="x(2-x)");
    plt.plot(x, clamp(x*(2-x), vmin=1, where=x>1), label="x(2-x) if x <= 1 else 1");
    plt.plot(x, clamp(x*(2-x), vmax=0.75), label="min(x(2-x), 0.75)");
    """
    if not is_array_like(x):
        a = x
        if vmax is not None:
            a = min(a, vmax)
        if vmin is not None:
            a = max(a, vmin)
        return a
    a = np.asarray(x).copy()
    if vmax is not None:
        a = np.fmin(a, vmax, out=a, where=where)
    if vmin is not None:
        a = np.fmax(a, vmin, out=a, where=where)
    return a


def find_communities_simple(
    G: nx.Graph | str,
    silent=True,
    initial_partition=None,
    phys_id="phys_id",
    store_json_output=False,
    print_result=True,
    **infomap_args,
):
    im = infomap.Infomap(silent=silent, **infomap_args)

    have_file_input = isinstance(G, str)
    if have_file_input:
        im.read_file(G)
    else:
        im.add_networkx_graph(G)

    im.run(initial_partition=initial_partition)

    if print_result:
        if im.num_levels > 2:
            print(
                f"Found {im.num_levels} levels with {im.num_top_modules} top modules and codelength {im.codelength}\n"
            )
        else:
            print(
                f"Found {im.num_top_modules} modules with codelength {im.codelength}\n"
            )

    if have_file_input:
        is_state_network = im.have_memory
    else:
        phys_ids = dict(G.nodes.data(phys_id)) if phys_id is not None else dict()
        is_state_network = None not in phys_ids.values()

        if store_json_output:
            # with tempfile.TemporaryDirectory() as tmp:
            #     json_filename = Path(tmp).joinpath('infomap_output.json')
            json_filename = "output/temp/infomap_output.json"
            im.write_json(json_filename)
            with open(json_filename, "r") as fp:
                G.graph["output"] = json.load(fp)

        G.graph["N"] = im.num_nodes
        G.graph["E"] = im.num_links
        G.graph["num_top_modules"] = im.num_top_modules
        G.graph["effective_num_top_modules"] = im.get_effective_num_modules(
            depth_level=1
        )
        G.graph["L"] = im.codelength
        G.graph["L_ind"] = im.index_codelength
        G.graph["L_mod"] = im.module_codelength
        G.graph["L_0"] = im.one_level_codelength
        G.graph["savings"] = im.relative_codelength_savings
        G.graph["max_depth"] = im.max_depth
        G.graph["modules"] = pd.DataFrame(
            im.get_multilevel_modules(states=is_state_network)
        ).T
        G.graph["effective_num_modules_per_level"] = [
            im.get_effective_num_modules(lvl) for lvl in range(1, im.max_depth)
        ]
        # G.graph["effective_num_nodes"] = calc_effective_number_of_nodes(G)
        # G.graph['entropy_rate'] = calc_entropy_rate(G)

    return im.get_dataframe(
        ["state_id", "node_id", "name", "module_id", "path", "flow"]
    ).set_index("state_id")


def find_communities(
    G,
    verbose=False,
    silent=True,
    initial_partition=None,
    tree_output=None,
    depth_level=1,
    return_im=False,
    return_dataframe=False,
    phys_id=None,
    layer_id=None,
    multilayer_inter_intra_format=True,
    store_json_output=False,
    community_attribute="community",
    **infomap_args,
):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and 'flow'.
    """

    if phys_id is None:
        phys_id = "phys_id"
    if layer_id is None:
        layer_id = "layer_id"

    phys_ids = dict(G.nodes.data(phys_id))
    is_state_network = None not in phys_ids.values()
    layer_ids = dict(G.nodes.data(layer_id))
    is_multilayer_network = None not in layer_ids.values()

    im = infomap.Infomap(silent=silent, **infomap_args)

    node_map = im.add_networkx_graph(G)
    # node_map = im.add_networkx_graph(G, phys_id=phys_id, multilayer_inter_intra_format=multilayer_inter_intra_format)
    # node_map = add_networkx_graph(
    #     im,
    #     G,
    #     phys_id=phys_id,
    #     layer_id=layer_id,
    #     multilayer_inter_intra_format=multilayer_inter_intra_format,
    # )
    # TODO: Remap nodes to state ids if multilayer?
    # is_state_network = phys_id is not None
    # add_networkx_graph(im, G, phys_id=phys_id)

    im.run(initial_partition=initial_partition)

    if return_im:
        return im

    # Store result on nodes and graph
    communities = im.get_modules(depth_level=depth_level, states=is_state_network)
    num_modules = np.unique(list(communities.values())).shape[0]

    if store_json_output:
        # with tempfile.TemporaryDirectory() as tmp:
        #     json_filename = Path(tmp).joinpath('infomap_output.json')
        json_filename = "output/temp/infomap_output.json"
        im.write_json(json_filename)
        with open(json_filename, "r") as fp:
            G.graph["output"] = json.load(fp)

    if verbose:
        print(
            f"Found {num_modules} modules in {im.max_depth} levels with codelength {im.codelength} (savings: {im.relative_codelength_savings:.3%})"
        )

    if is_multilayer_network:
        # node_map maps (layer_id, phys_id) -> state_id
        multilayer_node_to_nx_id = {
            (d["layer_id"], d["phys_id"]): n for n, d in G.nodes.data()
        }
        flow = {
            multilayer_node_to_nx_id[(node.layer_id, node.node_id)]: node.data.flow
            for node in im.nodes
        }
        # link_flow = {(source, target): flow for source, target, flow in im.get_links(data="flow")}
        # 2024-05-14, TODO: FIX: below gives KeyError: (1, 11) for StarWars.net
        for _, d in G.nodes.data():
            d["state_id"] = node_map[(d["layer_id"], d["phys_id"])]
    else:
        flow = {node.state_id: node.data.flow for node in im.nodes}
        link_flow = {
            (source, target): flow for source, target, flow in im.get_links(data="flow")
        }
        nx.set_edge_attributes(G, link_flow, "flow")

    nx.set_node_attributes(G, communities, community_attribute)
    nx.set_node_attributes(G, flow, "flow")
    G.graph["N"] = im.num_nodes
    G.graph["E"] = im.num_links
    G.graph["num_modules"] = num_modules
    G.graph["effective_num_modules"] = im.get_effective_num_modules(
        depth_level=depth_level
    )
    G.graph["L"] = im.codelength
    G.graph["L_ind"] = im.index_codelength
    G.graph["L_mod"] = im.module_codelength
    G.graph["L_0"] = im.one_level_codelength
    G.graph["savings"] = im.relative_codelength_savings
    G.graph["max_depth"] = im.max_depth
    G.graph["modules"] = pd.DataFrame(
        im.get_multilevel_modules(states=is_state_network)
    ).T
    G.graph["modules_old"] = im.get_multilevel_modules(states=is_state_network)
    # G.graph['multimodules'] = get_multilevel_modules(im)
    G.graph["effective_num_modules_per_level"] = [
        im.get_effective_num_modules(lvl) for lvl in range(1, im.max_depth)
    ]
    # G.graph["effective_num_nodes"] = calc_effective_number_of_nodes(G)
    # G.graph['entropy_rate'] = calc_entropy_rate(G)

    data = []
    for node in im.tree:
        if node.depth != 1:
            continue
        data.append(
            [node.path[0], node.data.flow, node.data.exitFlow, node.data.enterFlow]
        )
    G.graph["module_flow"] = pd.DataFrame(
        data, columns=["module_id", "flow", "exit", "enter"]
    ).set_index("module_id")
    G.graph["flow"] = (
        pd.DataFrame({"flow": flow.values(), "node_id": flow.keys()})
        .set_index("node_id")
        .sort_index()
    )

    if tree_output is not None:
        im.write_tree(tree_output)
        print(f"Wrote tree to '{tree_output}'")

    if return_dataframe:
        return im.get_dataframe(
            ["state_id", "node_id", "name", "module_id", "path", "flow"]
        ).set_index("state_id")
    return num_modules, im.codelength


def crop_cmap(
    cmap: LinearSegmentedColormap, fmin: float = 0.0, fmax: float = 1.0
) -> LinearSegmentedColormap:
    """Chops off the beginning `fmin` and ending `fmax` fraction of a colormap.
    Example: `crop_cmap(sns.color_palette("Greys", as_cmap=True), 0.2)`
    https://stackoverflow.com/a/71742009
    """
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[
        int(fmin * len(cmap_as_array)) : int(fmax * len(cmap_as_array))
    ]
    return LinearSegmentedColormap.from_list(
        cmap.name + f"_fmin{fmin}_fmax{fmax}", cmap_as_array
    )


plasma_cropped = crop_cmap(
    sns.color_palette("plasma", as_cmap=True), fmin=0.1, fmax=0.9
)


def progressBar(
    iterable, prefix="", suffix="", decimals=1, length=50, fill="█", printEnd="\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function

    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)

    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


def read_pajek(path: str, directed: bool = False):
    # return nx.read_pajek(path)
    G = nx.DiGraph() if directed else nx.Graph()
    with open(path) as fp:
        is_edges = True
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            if line[0] == "*":
                is_edges = not line[: len("*Vertices")].lower() == "*vertices"
                continue
            if not is_edges:
                # Line syntax: node_id "name" [optional weight]
                # node_id = line.split()[0]
                node_id, *rest = line.split()
                # TODO: Parse name correctly
                name = " ".join(rest)[1:-1]
                G.add_node(int(node_id), name=name)
            else:
                e = line.split()
                n1, n2 = int(e[0]), int(e[1])
                w = float(e[2]) if len(e) == 3 else 1
                G.add_edge(n1, n2, weight=w)
    return G


dt = {}

colors_turbo = [
    "#c63a69",
    "#74c83c",
    "#40afcb",
    "#cd8144",
    "#4a67d1",
    "#d3c04e",
    "#52d690",
    "#d96e57",
    "#695ddc",
    "#c2dd60",
    "#63dfc9",
    "#e18a66",
    "#6aa6e2",
    "#e3b76d",
    "#70e577",
    "#e68073",
    "#9777e7",
    "#bae878",
    "#7ae7e9",
    "#e9a37c",
    "#7da4ea",
    "#eacc7f",
    "#80eb9e",
    "#eb9082",
    "#838aec",
    "#e9ec84",
    "#86edc8",
    "#ed9f87",
    "#88c9ee",
    "#eebf8a",
    "#9eee8b",
    "#ef948c",
    "#b78def",
    "#beef8e",
    "#8ee5ef",
    "#f0b58f",
    "#8fabf0",
    "#f0db90",
    "#90f0b4",
    "#f09f91",
    "#9391f0",
    "#e4f092",
    "#92f0d7",
    "#f1ab92",
    "#93c8f1",
    "#f1c993",
    "#9af194",
    "#f19d94",
    "#a595f1",
    "#d4f195",
    "#95f1e9",
    "#f1b496",
    "#96bdf2",
    "#f2d396",
    "#97f2a7",
    "#f2a197",
    "#97a4f2",
    "#f2ec98",
    "#98f2ca",
    "#f2ab98",
    "#99d9f2",
    "#f2c599",
    "#b4f299",
    "#f2a09a",
    "#d49af3",
    "#c2f39a",
    "#9ae5f3",
    "#f3bf9a",
    "#9ab1f3",
    "#f3e39a",
    "#9bf3c0",
    "#f3a99b",
    "#a09bf3",
    "#e3f39b",
    "#9bf3df",
    "#f3b49b",
    "#9bcaf3",
    "#f3d09b",
    "#9cf39c",
    "#f3a49c",
    "#b09cf3",
    "#d4f39c",
    "#9cf3f0",
    "#f3ba9c",
    "#9cbff3",
    "#f3d89c",
    "#9cf3b0",
    "#f3a79c",
    "#9da6f3",
    "#f3f29d",
    "#9df3d0",
    "#f3b09d",
    "#9dd8f3",
    "#f3c99d",
    "#b2f39d",
    "#f3a49d",
    "#bc9df3",
    "#ccf39d",
    "#9deef3",
    "#f3be9e",
    "#9eb9f4",
    "#f4de9e",
    "#9ef4ba",
    "#f4aa9e",
    "#9ea0f4",
    "#edf49e",
    "#9ef4d9",
    "#f4b39e",
    "#9ed2f4",
    "#f4ce9e",
    "#a9f49e",
    "#f4a69e",
    "#ab9ff4",
    "#ddf49f",
    "#9ff4e8",
    "#f4b99f",
    "#9fc6f4",
    "#f4d59f",
    "#9ff4a9",
    "#f4a89f",
    "#9faff4",
    "#f4eb9f",
    "#9ff4cb",
    "#f4af9f",
    "#9fe0f4",
    "#f4c79f",
    "#bdf49f",
    "#f4a79f",
    "#eea0f4",
    "#c4f4a0",
    "#a0e5f4",
    "#f4c4a0",
    "#a0b3f4",
    "#f4e6a0",
    "#a0f4c5",
    "#f4aea0",
    "#a6a0f4",
    "#e3f4a0",
    "#a0f4e3",
    "#f4b8a0",
    "#a0cbf4",
    "#f4d3a0",
    "#a0f4a2",
    "#f4a8a0",
    "#b6a0f4",
    "#d4f4a0",
    "#a0f4f3",
    "#f4bea0",
    "#a0c0f4",
    "#f4dba0",
    "#a0f4b5",
    "#f4aba0",
    "#a0a7f4",
    "#f4f4a0",
    "#a0f4d4",
    "#f4b3a0",
    "#a0d8f4",
    "#f4cca0",
    "#b2f4a0",
    "#f4a7a0",
    "#c1a0f4",
    "#ccf4a0",
    "#a0edf4",
    "#f4c1a0",
    "#a0baf4",
    "#f4e0a0",
    "#a0f4be",
    "#f4ada0",
    "#a1a1f4",
    "#ecf4a1",
    "#a1f4dc",
    "#f4b6a1",
    "#a1d2f4",
    "#f4d0a1",
    "#a9f4a1",
    "#f4a8a1",
    "#aea1f4",
    "#dcf4a1",
    "#a1f4eb",
    "#f4bba1",
    "#a1c6f4",
    "#f4d7a1",
    "#a1f4ad",
    "#f4aaa1",
    "#a1aef4",
    "#f4eda1",
    "#a1f4ce",
    "#f4b1a1",
    "#a1dff4",
    "#f4c9a1",
    "#bcf4a1",
    "#f4a8a1",
    "#cca1f4",
    "#c8f4a1",
    "#a1e9f4",
    "#f4c4a1",
    "#a1b8f4",
    "#f4e4a1",
    "#a1f4c3",
    "#f4aea1",
    "#a5a1f4",
    "#e8f4a1",
    "#a1f4e0",
    "#f4b8a1",
    "#a1cff4",
    "#f4d2a1",
    "#a5f4a1",
    "#f4aaa1",
    "#b3a1f4",
    "#d8f4a1",
    "#a1f4ef",
    "#f4bda1",
    "#a1c4f4",
    "#f4daa1",
    "#a1f4b2",
    "#f4aca1",
    "#a1acf4",
    "#f4f1a1",
    "#a1f4d1",
    "#f4b3a1",
    "#a1dcf4",
    "#f4cba1",
    "#b8f4a1",
    "#f4a8a1",
    "#bba1f4",
    "#d1f4a1",
    "#a1f1f4",
    "#f4c0a1",
    "#a1bef4",
    "#f4dea2",
    "#a2f4bb",
    "#f4ada2",
    "#a2a5f4",
    "#f0f4a2",
    "#a2f4d9",
    "#f4b6a2",
    "#a2d6f4",
    "#f4cfa2",
    "#aff4a2",
    "#f4a9a2",
    "#aca2f4",
    "#e0f4a2",
    "#a2f4e7",
    "#f4bba2",
    "#a2c9f4",
    "#f4d6a2",
    "#a2f4a9",
    "#f4aaa2",
    "#a2b2f4",
    "#f4eaa2",
    "#a2f4ca",
    "#f4b1a2",
    "#a2e3f4",
    "#f4c8a2",
    "#c1f4a2",
    "#f4a9a2",
]
colors = colors_turbo
cmap = mpl.colors.ListedColormap(colors, name="c3")
if mpl.colormaps.get("c3") is None:
    mpl.colormaps.register(cmap)

projections = {"natural_earth": "+proj=natearth", "web_mercator": "epsg:3857"}

_proj = projections["web_mercator"]

cmap_red_blue = Color.interpolate(["red", "blue"], space="hsl")


def get_red_blue_color(p_hue, p_darkness, lightness_min=0.25, lightness_max=0.75):
    p_lightness = 1 - p_darkness
    lightness = lightness_min + p_lightness * (lightness_max - lightness_min)
    return cmap_red_blue(p_hue).set("lightness", lightness).convert("srgb")[:]


def to_gdf(geom):
    if type(geom) == gpd.GeoSeries:
        return gpd.GeoDataFrame({"geometry": geom})
    return gpd.GeoDataFrame({"geometry": [geom]})


def buffer_gdf(gdf, buffer=0.1):
    return to_gdf(gdf.iloc[0].geometry.buffer(buffer))


class Bioregions:
    def __init__(
        self,
        occurrence_input: str | pd.DataFrame,
        sep=",",
        resolution: float = 1,
        plot_width=10,
        add_worldmap=True,
        encoding="utf-8",
        latlong_cols=["latitude", "longitude"],
        species_col="species",
    ):
        self.occurrence_input = occurrence_input
        self.sep = sep
        self.resolution = resolution
        self.plot_width = plot_width
        self.G = nx.Graph()
        self.df_mesh = None
        self.G_neighbors = None
        self.add_worldmap = add_worldmap
        self.encoding = encoding
        self.latlong_cols = latlong_cols
        self.species_col = species_col
        self.cluster_data = dict()
        self.init()

    def init(self):
        resolution = self.resolution
        df = (
            pd.read_csv(self.occurrence_input, sep=self.sep, encoding=self.encoding)
            if isinstance(self.occurrence_input, str)
            else self.occurrence_input.copy()
        )
        # self.bbox = df.describe().loc[['min', 'max']]
        species_col = self.species_col

        df["count"] = 1
        df_species = df[[species_col, "count"]].groupby(species_col).count()
        num_species = df_species.shape[0]
        print(f"{num_species} species found!")
        lat, long = self.latlong_cols

        df["ilat"] = df[lat] // resolution
        df["ilong"] = df[long] // resolution
        df["ilat_ilong"] = df.apply(lambda row: f"{row['ilat']}_{row['ilong']}", axis=1)

        cell_ids = df["ilat_ilong"].unique()
        cell_ids = pd.Series(range(cell_ids.shape[0]), index=cell_ids)
        df["cell_id"] = df.apply(lambda row: cell_ids.loc[row["ilat_ilong"]], axis=1)

        df_cells = (
            df[["ilat", "ilong", "ilat_ilong"]]
            .groupby("ilat_ilong")
            .agg({"ilat": "first", "ilong": "first"})
        )
        num_cells = df_cells.shape[0]
        df_cells["cell_id"] = cell_ids[df_cells.index]
        df_cells["cell_name"] = df_cells.index
        df_cells = df_cells.set_index("cell_id")
        print(f"{num_cells} cells generated!")

        def create_cell(row):
            lat = row["ilat"] * self.resolution
            long = row["ilong"] * self.resolution
            return shapely.geometry.box(
                long, lat, long + self.resolution, lat + self.resolution
            )

        df_cells["geometry"] = df_cells.apply(create_cell, axis=1)
        df_cells = gpd.GeoDataFrame(df_cells, crs="EPSG:4326")
        df_cells["module"] = 0
        df_cells["bioregion"] = 0

        df_species["id"] = np.arange(num_cells, num_cells + num_species, dtype=int)
        df_species["module"] = 0

        df["species_id"] = df.apply(
            lambda row: df_species.loc[row[species_col], "id"], axis=1
        )

        df_species = df_species.reset_index().set_index("id")

        self.df = df
        self.df_species = df_species  # Index on node id, columns ['species', 'count']
        self.df_cells = df_cells  # Index on node id

        self.generate_graph()
        self.generate_node_data()
        self.init_endemic_count()

        # Fix special case for square grid cells
        # self.generate_neighbor_graph()
        # self.generate_mesh()

    def init_endemic_count(self):
        k1_species = self.df_species.loc[self.df_species.degree == 1].index
        self.df_cells["endemic_count"] = (
            self.df.loc[self.df.species_id.isin(k1_species)][["count", "cell_id"]]
            .groupby("cell_id")
            .sum()
        )

    def generate_neighbor_graph(self):
        print("Creating neighbor graph...")
        centroids = np.column_stack(
            (self.df_cells.centroid.x, self.df_cells.centroid.y)
        )
        G = weights.Queen.from_dataframe(
            self.df_cells, idVariable="cell_name"
        ).to_networkx()
        nx.set_node_attributes(G, dict(zip(G.nodes, centroids)), "pos")
        self.G_neighbors = G

    def generate_mesh(self, mesh_width=0.2):
        data = []
        for u, v in self.G_neighbors.edges:
            name1, geom1 = self.df_cells.loc[u, ["cell_name", "geometry"]]
            name2, geom2 = self.df_cells.loc[v, ["cell_name", "geometry"]]
            try:
                shared_geom = (
                    to_gdf(geom1.buffer(mesh_width / 2))
                    .overlay(to_gdf(geom2.buffer(mesh_width / 2)))
                    .iloc[0, 0]
                )
                data.append([u, v, name1, name2, shared_geom])
            except Exception as e:
                print(e)

        self.df_mesh = gpd.GeoDataFrame(
            data, columns=["n1", "n2", "name1", "name2", "geometry"]
        )

    def generate_graph(self, weighted=False, log=True):
        G = nx.Graph()
        for sp in self.df_species.itertuples():
            G.add_node(sp.Index, name=sp.species, type="species")
        for cell in self.df_cells.itertuples():
            G.add_node(cell.Index, name=cell.cell_name, type="cell")
        #     for sp in np.unique(cell.species_id):
        #         G.add_edge(cell.Index, sp)
        if not weighted:
            for record in self.df.itertuples():
                G.add_edge(record.cell_id, record.species_id)
            self.G = G
            return

        k_species_max = self.df_nodes[self.df_nodes.type == "species"].degree.max()
        for record in self.df.itertuples():
            # k_cell = self.df_nodes.loc[record.cell_id, 'degree']
            k_species = self.df_nodes.loc[record.species_id, "degree"]
            # weight = k_max / max(k_cell, k_species)
            weight = k_species_max / k_species
            if log:
                weight = np.log2(weight + 0.1)
            G.add_edge(record.cell_id, record.species_id, weight=weight)
        self.G = G

    def generate_graph_projected(self, weighted=False):
        if not weighted:
            return nx.projected_graph(self.G, self.cell_nodes)

        Gm = nx.projected_graph(self.G, self.cell_nodes, multigraph=True)
        G = nx.Graph()
        for u, v, w in Gm.edges.data("weight", default=1):
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
        return G

    def calculate_cartography_metrics(self, standardize=True):
        # metrics = ['k_intra', 'k_inter', 'sp_intra', 'sp_inter']
        metrics = ["species_richness", "biota_overlap", "endemicity", "occupancy"]
        self.biodiversity_metrics = metrics
        for metric in metrics:
            self.df_cells[metric] = 0

        for n, d in self.G.nodes.data():
            if d["type"] == "species":
                continue
            cell_module = d["community"]
            k_intra = 0
            k_inter = 0
            mean_sp_intra = 0
            mean_sp_inter = 0
            # Calculate num intra/inter links
            for sp in self.G[n]:
                sp_module = self.G.nodes[sp]["community"]
                if sp_module != cell_module:
                    k_inter += 1
                else:
                    k_intra += 1
                    sp_intra = 0
                    sp_inter = 0
                    # Calculate mean values for the regional species in the cell
                    for cell2 in self.G[sp]:
                        cell2_module = self.G.nodes[cell2]["community"]
                        if cell2_module == sp_module:
                            sp_intra += 1
                        else:
                            sp_inter += 1
                    mean_sp_intra += sp_intra
                    mean_sp_inter += sp_inter / (sp_intra + sp_inter)
            mean_sp_intra /= k_intra
            mean_sp_inter /= k_intra

            self.df_cells.at[n, "species_richness"] = k_intra
            self.df_cells.at[n, "biota_overlap"] = k_inter / (k_intra + k_inter)
            self.df_cells.at[n, "endemicity"] = 1 - mean_sp_inter
            self.df_cells.at[n, "occupancy"] = mean_sp_intra

        if standardize:
            standardized_metrics = ["species_richness", "occupancy"]
            # https://stackoverflow.com/questions/54907933/pandas-groupby-and-calculate-z-score
            self.df_cells[standardized_metrics] = (
                self.df_cells[["bioregion"] + standardized_metrics]
                .groupby("bioregion")
                .transform(zscore)
                .fillna(0)
            )

        for metric in metrics:
            self.df_cells[f"{metric}_raw"] = self.df_cells[metric]

        # if not categorize:
        #     return

        # Bin values to three intervals with equal amount of samples
        values = ["Low", "Medium", "High"]
        for metric in self.biodiversity_metrics:
            s = self.df_cells[metric].sort_values()
            mask_low = np.full((s.size,), False)
            q3 = s.size // 3
            mask_low[:q3] = True
            mask_medium = np.full((s.size,), False)
            mask_medium[q3 : 2 * q3] = True
            mask_medium
            mask_high = ~mask_low & ~mask_medium
            self.df_cells.loc[s.index[mask_low], metric] = "Low"
            self.df_cells.loc[s.index[mask_medium], metric] = "Medium"
            self.df_cells.loc[s.index[mask_high], metric] = "High"
            # self.df_cells[metric] = pd.cut(
            #     s,
            #     [s.min(), s.iloc[s.size // 3], s.iloc[2 * s.size // 3], s.max()],
            #     include_lowest=True,
            #     labels=values,
            # )

        # sort values to avoid PerformanceWarning: indexing past lexsort depth may impact performance.
        values.sort()
        outer_dims = ["biota_overlap", "endemicity"]
        inner_dims = ["species_richness", "occupancy"]

        index = pd.MultiIndex.from_product(
            [values, values], names=[outer_dims[0], inner_dims[0]]
        )
        columns = pd.MultiIndex.from_product(
            [values, values], names=[outer_dims[1], inner_dims[1]]
        )
        df = pd.DataFrame(np.zeros((9, 9), dtype=int), index=index, columns=columns)

        i_col, j_col = outer_dims
        df0 = pd.DataFrame(np.zeros((3, 3), dtype=int), index=values, columns=values)
        df0.index.name = inner_dims[0]
        df0.columns.name = inner_dims[1]
        X = self.df_cells[metrics]

        for i_val in values:
            df_outer = X[X[i_col] == i_val]
            for j_val in values:
                df_inner_incomplete = df_outer[df_outer[j_col] == j_val][inner_dims]
                df_inner = df0.copy()
                df_inner.update(df_inner_incomplete.value_counts().unstack())
                df.loc[(i_val,), (j_val,)] = df_inner.astype(int).values

        self.df_metrics = df

    def plot_metrics_raw(self, cluster=True):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        metrics_raw = [f"{metric}_raw" for metric in self.biodiversity_metrics]

        if cluster:
            X = self.df_cells[metrics_raw].copy()
            db = skl_cluster.DBSCAN(eps=0.3, min_samples=10).fit(X)
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)

            unique_labels = set(labels)
            core_samples_mask = np.zeros_like(labels, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            X["core"] = core_samples_mask
            X["label"] = labels
            X["core_strength"] = X["core"].astype(int) + 1
            num_labels = X["label"].nunique()
            palette = sns.color_palette(n_colors=num_labels)
            palette = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
            palette[-1] = (0, 0, 0)
            for i, (x, y), ax in zip(
                range(6), it.combinations(metrics_raw, 2), axes.flatten()
            ):
                draw_legend = (i + 1) % 3 == 0
                sns.scatterplot(
                    X,
                    x=x,
                    y=y,
                    hue="label",
                    size="core_strength",
                    ax=ax,
                    legend=draw_legend,
                    palette=palette,
                )
                r = self.df_cells[[x, y]].corr().iloc[0, 1]
                ax.set(title=f"r = {r:.2f}", xlabel=x[:-4], ylabel=y[:-4])
                if draw_legend:
                    ax.legend(bbox_to_anchor=(1, 1))
            fig.suptitle(
                f"{n_clusters_} clusters and {n_noise_}/{X.shape[0]} noise points"
            )
        else:
            X = self.df_cells[["bioregion"] + metrics_raw]
            num_bioregions = X["bioregion"].nunique()
            for i, (x, y), ax in zip(
                range(6), it.combinations(metrics_raw, 2), axes.flatten()
            ):
                draw_legend = (i + 1) % 3 == 0
                sns.scatterplot(
                    X,
                    x=x,
                    y=y,
                    hue="bioregion",
                    ax=ax,
                    legend=draw_legend,
                    palette="deep" if num_bioregions < 13 else "flare",
                )
                r = self.df_cells[[x, y]].corr().iloc[0, 1]
                ax.set(title=f"r = {r:.2f}", xlabel=x[:-4], ylabel=y[:-4])
                if draw_legend:
                    ax.legend(bbox_to_anchor=(1, 1))
        fig.tight_layout()

    def plot_metrics_categorical(self):
        fig, axes = plt.subplots(
            nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8)
        )

        effective_number_of_combinations = perplexity(self.df_metrics.values.flatten())
        max_value = self.df_metrics.values.max()
        values = ["Low", "Medium", "High"]
        # outer_dims = ["endemicity", "biota_overlap"]
        # inner_dims = ['species_richness', 'occupancy']
        # i_col, j_col = outer_dims

        for i, i_val in enumerate(values):
            for j, j_val in enumerate(values):
                ax = axes[i][j]
                df = self.df_metrics.loc[(i_val,), (j_val,)].loc[values, values]
                sns.heatmap(
                    df, ax=ax, cbar=j == 2, vmin=0, vmax=max_value, cmap="flare"
                )
                if i != 2:
                    ax.set(xlabel=None)
                    if i == 0:
                        ax.set(title=f"biota_overlap {j_val}")
                if j != 0:
                    ax.set(ylabel=None)
                else:
                    ylabel = ax.yaxis.get_label().get_text()
                    ax.set_ylabel(f"endemicity {i_val}\n{ylabel}")

        random_data = [
            perplexity(sample_multinomial(n=self.df_cells.shape[0], k=81))
            for _ in range(1000)
        ]
        rand_mean, rand_std = np.mean(random_data), np.std(random_data)
        fig.suptitle(
            f"Effective number of combinations: {round(effective_number_of_combinations)} of 81"
        )
        fig.text(
            0.5,
            1,
            f"(M={rand_mean:.1f}, SD={rand_std:.1f} expected by chance)",
            ha="center",
        )
        fig.tight_layout()

    @property
    def num_cells(self):
        return self.df_cells.shape[0]

    @property
    def num_species(self):
        return self.df_species.shape[0]

    @property
    def cell_nodes(self):
        return [n for n, d in self.G.nodes(data=True) if d["type"] == "cell"]

    @property
    def species_nodes(self):
        return [n for n, d in self.G.nodes(data=True) if d["type"] == "species"]

    @property
    def modules(self):
        return nx.get_node_attributes(self.G, "community")

    @property
    def module_counts(self):
        return self.df_nodes.module.value_counts()

    @property
    def cell_modules(self):
        return self.df_cells.module.value_counts()

    @property
    def species_modules(self):
        return self.df_species.module.value_counts()

    @property
    def cell_singletons(self):
        return self.cell_modules[self.cell_modules == 1].index

    @property
    def species_singletons(self):
        return self.species_modules[self.species_modules == 1].index

    @property
    def singletons(self):
        return self.module_counts[self.module_counts == 1].index

    @property
    def num_modules(self):
        return self.G.graph["num_modules"]

    @property
    def num_cell_modules(self):
        return self.cell_modules.size

    num_bioregions = num_cell_modules

    @property
    def num_species_modules(self):
        return self.species_modules.size

    @property
    def num_cell_singletons(self):
        return self.cell_singletons.size

    @property
    def num_species_singletons(self):
        return self.species_singletons.size

    @property
    def num_singletons(self):
        return self.singletons.size

    @property
    def codelength(self):
        return self.G.graph["L"]

    @property
    def savings(self):
        return self.G.graph["savings"]

    @property
    def num_module_levels(self):
        return self.G.graph["max_depth"] - 1

    @property
    def num_modules_per_level(self):
        return [
            self.df_nodes[f"module_{lvl}"].nunique()
            for lvl in range(self.num_module_levels)
        ]

    @property
    def num_bioregions_per_level(self):
        return [
            self.df_cells[f"module_{lvl}"].nunique()
            for lvl in range(self.num_module_levels)
        ]

    @property
    def effective_num_modules_per_level(self):
        return np.array(self.G.graph["effective_num_modules_per_level"])

    @property
    def species_degree(self):
        return pd.Series(dict(self.G.degree(self.species_nodes))).sort_values(
            ascending=False
        )

    @property
    def cell_degree(self):
        return pd.Series(dict(self.G.degree(self.cell_nodes))).sort_values(
            ascending=False
        )

    def generate_node_data(self):
        data = []
        _, d = get_first_node(self.G, data=True)
        for n, d in self.G.nodes(data=True):
            data.append([n, d["name"], d["type"], self.G.degree(n)])
        columns = ["node", "name", "type", "degree"]
        df_nodes = pd.DataFrame(data, columns=columns)
        df_nodes = df_nodes.set_index("node")
        df_nodes["module"] = 0
        self.df_nodes = df_nodes

        self.df_cells["degree"] = self.df_cells.apply(
            lambda row: self.df_nodes.loc[row.name, "degree"], axis=1
        )
        self.df_species["degree"] = self.df_species.apply(
            lambda row: self.df_nodes.loc[row.name, "degree"], axis=1
        )

    def update_modular_level(self, level=0):
        modules = self.df_nodes[f"module_{level}"]
        self.df_nodes["module"] = modules
        self.df_species["module"] = modules
        self.df_cells["module"] = modules

        # Ordered by the number of nodes in each module
        for lvl in range(self.num_module_levels):
            module_col = f"module_{lvl}"
            modules = self.df_nodes[module_col]
            self.df_species[module_col] = modules
            self.df_cells[module_col] = modules
            module_counts = self.df_cells[module_col].value_counts()
            reindex = dict(zip(module_counts.index, range(module_counts.shape[0])))
            self.df_cells[f"bioregion_{lvl}"] = self.df_cells.apply(
                lambda row: reindex[row[module_col]], axis=1
            )  # .astype(int)
        self.df_cells["bioregion"] = self.df_cells[f"bioregion_{level}"]

    def update_interior(self):
        if self.df_mesh is None:
            return

        # e_attr = dict()
        # for u, v in self.G_neighbors.edges:
        #     e_attr[(u, v)] = [
        #         is_equal(self.df_nodes.loc[[u, v], f'module_{lvl}'].values) for lvl in range(self.num_module_levels)
        #     ]
        # nx.set_edge_attributes(self.G_neighbors, e_attr, "interior")

        cols = [f"is_interior_{lvl}" for lvl in range(self.num_module_levels)]

        def add_interior(row):
            return [
                is_equal(self.df_nodes.loc[[row.n1, row.n2], f"module_{lvl}"].values)
                for lvl in range(self.num_module_levels)
            ]

        self.df_mesh[cols] = self.df_mesh.apply(
            add_interior, axis=1, result_type="expand"
        )

    def update_partition(self, modules=None, level=0):
        """
        params
        ------
        modules: dict
            Partition, single or multilevel
        """
        if modules is None:
            modules = self.G.graph["modules"]  # multilevel

        for lvl in modules.columns:
            self.df_nodes[f"module_{lvl}"] = modules[lvl]

        self.update_interior()
        self.update_modular_level(level=level)

    def partition(self, G=None, **infomap_kwargs):
        if G is None:
            G = self.G

        find_communities(G, **infomap_kwargs)

        print(
            f"Found {self.num_module_levels} modular levels with {self.num_modules} top modules, {self.effective_num_modules_per_level.round(1)} effective modules per level and codelength {self.G.graph['L']} with relative codelength savings {self.G.graph['savings']:.1%}"
        )

        self.update_partition(G.graph["modules"])

        print(f"Number of bioregions per level: {self.num_bioregions_per_level}")

    def get_perturbed_graph(self, fraction_species_to_remove=0.95):
        G = self.G
        N, E = get_N(G), get_E(G)
        nodes = [n for n, t in G.nodes.data("type") if t == "species"]
        G.remove_nodes_from(
            rng.choice(
                nodes, size=int(fraction_species_to_remove * len(nodes)), replace=False
            )
        )
        print(
            f"Removed {fraction_species_to_remove:.1%} of nodes. N: {N} -> {get_N(G)}, E: {E} -> {get_E(G)}"
        )
        return G

    def add_significance_clustering(
        self,
        filename="wcvp_bipartite.csv",
        dir="data/data/output/notebooks/bioregions",
        prefix="",
    ):
        filename = Path(dir).joinpath(filename)
        df_s_modules = pd.read_csv(filename, index_col="node_id")
        num_levels = np.array(
            [col.startswith("modules ") for col in df_s_modules.columns]
        ).sum()

        icol_L_begin = df_s_modules.shape[1] - 1
        while is_float(df_s_modules.columns[icol_L_begin]):
            icol_L_begin -= 1
        icol_L_begin += 1
        num_partitions = df_s_modules.shape[1] - icol_L_begin
        self.cluster_data[f"{prefix}_num_partitions"] = num_partitions
        self.cluster_data[f"{prefix}_codelengths"] = [
            float(L) for L in df_s_modules.columns[icol_L_begin:]
        ]
        self.cluster_data[f"{prefix}_num_levels"] = num_levels
        print(
            f"Add {num_levels} levels of significance clustering from {num_partitions} partitions..."
        )

        # Fallback on level 1 scores
        for lvl in range(2, num_levels + 1):
            df_s_modules.loc[df_s_modules[f"level {lvl}"].isna(), f"level {lvl}"] = (
                df_s_modules.loc[
                    df_s_modules[f"level {lvl}"].isna(), f"level {lvl - 1}"
                ]
            )

        for lvl in range(num_levels):
            self.df_cells[f"{prefix}s_module_{lvl}"] = df_s_modules[
                f"modules {lvl + 1}"
            ]
            self.df_cells[f"{prefix}s_score_{lvl}"] = df_s_modules[f"level {lvl + 1}"]

        def calc_pairwise_score(n1, n2, level):
            df_pair = df_s_modules.loc[[n1, n2]]
            is_interior = is_equal(df_pair[f"modules {level}"].values)
            num_diffs = 0
            # TODO: Weight by codelengths?
            for i in range(icol_L_begin, icol_L_begin + num_partitions):
                try:
                    m1, m2 = [str(m).split(":") for m in df_pair.values[:, i]]
                    lvl = min(len(m1), len(m2), level)
                    if m1[lvl - 1] != m2[lvl - 1]:
                        num_diffs += 1
                except:
                    print("Error:", i, df_pair.values[:, i])
            return is_interior, num_diffs / num_partitions

        def add_bioregion_data(row):
            return list(
                it.chain.from_iterable(
                    [
                        calc_pairwise_score(row.n1, row.n2, lvl)
                        for lvl in range(1, num_levels + 1)
                    ]
                )
            )

        new_cols = list(
            it.chain.from_iterable(
                [
                    [f"{prefix}is_interior_{lvl}", f"{prefix}score_{lvl}"]
                    for lvl in range(num_levels)
                ]
            )
        )
        self.df_mesh[new_cols] = self.df_mesh.apply(
            add_bioregion_data, axis=1, result_type="expand"
        )

    def plot_heatmap(self, width=None, title=None, ax=None):
        cmap_continuous = plasma_cropped

        w = width or self.plot_width
        if ax is None:
            _, ax = plt.subplots(
                figsize=(w, w)
            )  # second w is height but map aspect ratio will override that

        ax.grid(False)
        ax.axis("off")

        vmin, vmax = minmax(self.df_cells["degree"])
        norm = mpl.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        self.df_cells.plot(
            column="degree",
            ax=ax,
            legend=True,
            linewidth=0.0,
            cmap=cmap_continuous,
            norm=norm,
        )
        ax.set(title=title or "Species richness")

    def plot(
        self,
        level=0,
        width=None,
        title_prefix="",
        title=None,
        ax=None,
        add_worldmap=None,
        module_boundary_width=0.1,
        module_boundary_color="white",
        ec_cells="#999999",
    ):
        w = width or self.plot_width
        if ax is None:
            _, ax = plt.subplots(
                figsize=(w, w)
            )  # second w is height but map aspect ratio will override that

        ax.grid(False)
        ax.axis("off")

        if add_worldmap or self.add_worldmap:
            worldmap = gpd.read_file(gds.get_path("naturalearth land"))
            worldmap.plot(color="lightgrey", alpha=0.5, ax=ax)

        num_bioregions = self.num_bioregions_per_level[level]

        cmap = mpl.colors.ListedColormap(colors[:num_bioregions], name="c3")
        self.df_cells.plot(
            column=f"bioregion_{level}",
            ax=ax,
            categorical=True,
            ec=ec_cells,
            linewidth=0.1,
            cmap=cmap,
        )

        if self.df_mesh is not None:
            self.df_mesh.loc[~self.df_mesh[f"is_interior_{level}"]].plot(
                ax=ax,
                fc=module_boundary_color,
                ec=module_boundary_color,
                linewidth=module_boundary_width,
            )

        # title = f"{title_prefix}Level {level}: {self.num_modules_per_level[level]} modules ({self.effective_num_modules_per_level.round()[level]} effective), {num_bioregions} bioregions, {self.num_cell_singletons}/{self.num_species_singletons}/{self.num_singletons} cell/species/node singletons, {self.codelength:.3f} bits ({self.savings:.1%})"
        # title = f"{title_prefix}Level {level+1}/{self.num_module_levels}: {self.num_modules_per_level[level]} modules ({self.effective_num_modules_per_level[level]:.1f} effective), {num_bioregions} bioregions, {self.codelength:.3f} bits ({self.savings:.1%})"
        if title is None:
            title = f"{title_prefix}Level {level + 1}/{self.num_module_levels}: {self.num_modules_per_level[level]} modules, {num_bioregions} bioregions, {self.codelength:.2f} bits ({self.savings:.1%} savings)"
        ax.set_title(title)
        return ax

    def plot_significance(
        self,
        level=0,
        width=None,
        title=None,
        title_prefix="",
        ax=None,
        add_worldmap=None,
        remap_color_index=False,
        fc_cell_alpha_min=0.2,
        ec_cells="#999999",
        no_mesh=False,
        prefix="",
    ):
        w = width or self.plot_width
        if ax is None:
            _, ax = plt.subplots(
                figsize=(w, w)
            )  # second w is height but map aspect ratio will override that

        ax.grid(False)
        ax.axis("off")

        if add_worldmap:
            worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            worldmap.plot(
                color="#eeeeee", ec="#999999", alpha=0.5, linewidth=0.05, ax=ax
            )

        module_col = f"{prefix}s_module_{level}"
        score_col = f"{prefix}s_score_{level}"

        # min_bioregion, max_bioregion = py.minmax(self.df_cells[module_col])
        min_bioregion = self.df_cells[module_col].min()
        num_bioregions = self.df_cells[module_col].nunique()

        core_fraction = self.df_cells[score_col].mean()

        cmap = mpl.colors.ListedColormap(colors[:num_bioregions], name="c3")

        if remap_color_index:
            self.df_cells.plot(
                column=module_col,
                ax=ax,
                categorical=True,
                alpha=clamp(self.df_cells[score_col], vmin=fc_cell_alpha_min),
                ec=ec_cells,
                linewidth=0.05,
                cmap=cmap,
            )
        else:
            self.df_cells["color_rgba"] = self.df_cells.apply(
                lambda row: mpl.colors.to_rgba(
                    cmap(row[module_col] - min_bioregion),
                    alpha=clamp(row[score_col], vmin=fc_cell_alpha_min),
                ),
                axis=1,
            )
            self.df_cells.plot(
                color=self.df_cells["color_rgba"],
                ax=ax,
                categorical=True,
                ec=ec_cells,
                linewidth=0.05,
            )

        if self.df_mesh is not None and not no_mesh:
            mesh = self.df_mesh.loc[~self.df_mesh[f"{prefix}is_interior_{level}"]]
            mesh.plot(
                ax=ax,
                fc="white",
                ec="white",
                alpha=clamp(mesh[f"{prefix}score_{level}"], vmin=0.4),
                linewidth=clamp(mesh[f"{prefix}score_{level}"], vmin=0.2),
            )

        # title = f"{title_prefix}Level {level}: {self.num_modules_per_level[level]} modules ({self.effective_num_modules_per_level.round()[level]} effective), {num_bioregions} bioregions, {self.num_cell_singletons}/{self.num_species_singletons}/{self.num_singletons} cell/species/node singletons, {self.codelength:.3f} bits ({self.savings:.1%})"

        if title is None:
            title = f"Level {level + 1}/{self.cluster_data[f'{prefix}_num_levels']}: {num_bioregions} bioregions"
            title += f" ({self.cluster_data[f'{prefix}_codelengths'][0]:.4f} bits, {core_fraction:.1%} mean score, {self.cluster_data[f'{prefix}_num_partitions']} partitions)"
        title = f"{title_prefix}{title}"
        ax.set_title(title)
        return ax

    def plot_perturbed(
        self,
        fraction_species_to_remove=0.95,
        width=None,
        title_prefix="",
        ax=None,
        **infomap_kwargs,
    ):
        G_orig = self.G.copy()
        G = self.get_perturbed_graph(
            fraction_species_to_remove=fraction_species_to_remove
        )
        self.G = G
        self.partition(**infomap_kwargs)
        self.plot(
            width=width,
            title_prefix=f"Removed {fraction_species_to_remove:.0%} species: {title_prefix}",
            ax=ax,
        )
        print("Restoring original modules...")
        self.G = G_orig
        self.update_partition()

    def plot_neighbor_graph(self):
        ax = self.df_cells.plot(
            linewidth=1, edgecolor="grey", facecolor="lightblue", figsize=(15, 5)
        )
        ax.axis("off")
        ax.grid(False)
        # ax.set(title=title)
        pos = nx.get_node_attributes(self.G_neighbors, "pos")
        nx.draw(self.G_neighbors, pos, ax=ax, node_size=5, node_color="r")

    def plot_degree_distribution(self, species=True, cells=True, figsize=(5, 3)):
        _, ax = plt.subplots(figsize=figsize)

        if species:
            k_dist_species = self.species_degree.value_counts()
            sns.lineplot(
                x=k_dist_species.index, y=k_dist_species.values, label="Species", ax=ax
            )

        if cells:
            k, count = np.unique(
                [k for _, k in nx.degree(self.G, self.cell_nodes)], return_counts=True
            )
            sns.lineplot(x=k, y=count, label="Cells", ax=ax)

        ax.set(
            title="Degree distribution", xlabel="Degree", ylabel="Count", yscale="log"
        )

    def plot_species_node_degrees(self, **kwargs):
        self.species_degree.plot(
            use_index=False,
            title="Species node degrees",
            xlabel="Species node",
            ylabel="Degree",
            **kwargs,
        )

    def plot_cell_node_degrees(self, **kwargs):
        self.cell_degree.plot(
            use_index=False,
            title="Cell node degrees",
            xlabel="Cell node",
            ylabel="Degree",
            **kwargs,
        )

    def write_nodes(
        self, filename="nodes.csv", dir="output/bioregions", limit=None, id_offset=1
    ):
        p = Path(dir)
        p.mkdir(exist_ok=True)
        p = p.joinpath(filename)
        print(f"Writing {get_N(self.G)} nodes to '{p}'...")
        i_end = limit - 1 if limit is not None else -1
        with open(p, "w") as fp:
            fp.write(f"id,label,size,type,module,flow\n")
            for i, (n, d) in enumerate(self.G.nodes.data()):
                fp.write(
                    f"{n + id_offset},{d['name']},{self.G.degree[n]},{d['type']},{d['community']},{d['flow']}\n"
                )
                if i == i_end:
                    break
        print("Done!")

    def write_edges(
        self, filename="edges.csv", dir="output/bioregions", limit=None, id_offset=1
    ):
        p = Path(dir)
        p.mkdir(exist_ok=True)
        p = p.joinpath(filename)
        print(f"Writing {get_E(self.G)} edges to '{p}'...")
        names = nx.get_node_attributes(self.G, "name")
        i_end = limit - 1 if limit is not None else -1
        with open(p, "w") as fp:
            fp.write(f"species_id,cell_id,species_name,cell_name,flow\n")
            for i, (u, v, f) in enumerate(self.G.edges.data("flow")):
                fp.write(f"{u + id_offset},{v + id_offset},{names[u]},{names[v]},{f}\n")
                if i == i_end:
                    break
        print("Done!")

    def write_pajek(self, filename="network.net", dir="output/bioregions"):
        p = Path(dir)
        p.mkdir(exist_ok=True)
        p = p.joinpath(filename)
        print(f"Writing network to '{p}'...")
        with open(p, "w") as fp:
            # fp.write(f"*Vertices {self.df_nodes.shape[0]}\n")
            fp.write(f"*Vertices {self.df_nodes.index.max() + 1}\n")
            for node in self.df_nodes.sort_index().itertuples():
                fp.write(f'{node.Index + 1} "{node.name}"\n')
            fp.write(f"*Edges {get_E(self.G)}\n")
            for u, v in self.G.edges:
                fp.write(f"{u + 1} {v + 1}\n")
        print("Done!")

    def write_bipartite(
        self, filename="network_bipartite.net", dir="output/bioregions"
    ):
        p = Path(dir)
        p.mkdir(exist_ok=True)
        p = p.joinpath(filename)
        print(f"Writing network to '{p}'...")
        with open(p, "w") as fp:
            # fp.write(f"*Vertices {self.df_nodes.shape[0]}\n")
            fp.write(f"*Vertices {self.df_nodes.shape[0]}\n")
            for node in self.df_nodes.sort_index().itertuples():
                fp.write(f'{node.Index} "{node.name}"\n')
            fp.write(f"*Bipartite {self.num_cells}\n")
            for u, v in self.G.edges:
                fp.write(f"{u} {v}\n")
        print("Done!")


def find_shared_paths(geom1, geom2):
    ext1 = (
        [p.exterior for p in geom1.geoms]
        if geom1.type == "MultiPolygon"
        else [geom1.exterior]
    )
    ext2 = (
        [p.exterior for p in geom2.geoms]
        if geom2.type == "MultiPolygon"
        else [geom2.exterior]
    )

    shared = []
    for e1, e2 in it.product(ext1, ext2):
        s = ops.shared_paths(e1, e2)
        if not s.is_empty:
            # s is geometry collection, first is shared path in equal direction, second in oppposite
            # We are interested in opposite as that means shared border of two polygons on opposite sides
            s = s.geoms[1]
            if not s.is_empty:
                if s.type == "MultiLineString":
                    for g in s.geoms:
                        shared.append(g)
                else:
                    shared.append(s)
    # if len(shared) == 0:
    #     return None
    if len(shared) == 1:
        return geometry.LineString(shared[0])

    return geometry.MultiLineString(shared)


class BiodiversityMetrics:
    def __init__(
        self,
        network_file: str,
        tree_file: str,
        species_start_id: int,
    ):
        self.network_file = network_file
        self.tree_file = tree_file
        self.species_start_id = species_start_id
        self.init()

    @property
    def X(self):
        return self.df_cells[self.metrics_z]

    def init(self):
        G = read_pajek(self.network_file)
        print(f"Parsed network with {get_N(G)} nodes and {get_E(G)} edges.")
        im = infomap.Infomap(no_infomap=True, silent=True)
        im.read_file(self.network_file)
        im.run(cluster_data=self.tree_file)
        print(
            f"Tree contains {im.num_top_modules} top modules in {im.num_levels} levels with codelength {im.codelength:.2f}"
        )
        df = (
            im.get_dataframe(columns=["node_id", "name", "module_id"])
            .set_index("node_id")
            .rename(columns={"module_id": "bioregion"})
        )

        df["type"] = df.apply(
            lambda row: "cell" if row.name < self.species_start_id else "species",
            axis=1,
        )

        for n, d in G.nodes.data():
            d["community"], d["type"] = df.loc[n, ["bioregion", "type"]]

        self.df = df
        self.G = G
        self.df_cells = df[df["type"] == "cell"].copy()
        self.num_cells = self.df_cells.shape[0]
        self.num_species = df.shape[0] - self.num_cells

        self.calculate_cartography_metrics()

    def calculate_cartography_metrics(self, standardize=True):
        print(
            f"Calculating biodiversity metrics for {self.num_cells} cells and {self.num_species} species..."
        )
        # metrics = ['k_intra', 'k_inter', 'sp_intra', 'sp_inter']
        # metrics = ["species_richness", "biota_overlap", "endemicity", "occupancy"]
        metrics = ["biota_overlap", "species_richness", "occupancy", "endemicity"]
        for metric in metrics:
            self.df_cells[metric] = 0

        for n, d in self.G.nodes.data():
            if d["type"] == "species":
                continue
            cell_module = d["community"]
            k_intra = 0
            k_inter = 0
            cell_sp_intra = []
            cell_sp_inter = []
            # Calculate num intra/inter links
            for sp in self.G[n]:
                sp_module = self.G.nodes[sp]["community"]
                if sp_module != cell_module:
                    k_inter += 1
                else:
                    k_intra += 1
                    sp_intra = 0
                    sp_inter = 0
                    # Calculate mean values for the regional species in the cell
                    for cell2 in self.G[sp]:
                        cell2_module = self.G.nodes[cell2]["community"]
                        if cell2_module == sp_module:
                            sp_intra += 1
                        else:
                            sp_inter += 1
                    cell_sp_intra.append(sp_intra)
                    cell_sp_inter.append(sp_inter / (sp_intra + sp_inter))

            self.df_cells.at[n, "species_richness"] = k_intra
            self.df_cells.at[n, "biota_overlap"] = k_inter / (k_intra + k_inter)
            self.df_cells.at[n, "endemicity"] = 1 - np.median(cell_sp_inter)
            self.df_cells.at[n, "occupancy"] = np.median(cell_sp_intra)

        if standardize:
            standardized_metrics = ["species_richness", "occupancy"]
            # https://stackoverflow.com/questions/54907933/pandas-groupby-and-calculate-z-score
            self.df_cells[standardized_metrics] = (
                self.df_cells[["bioregion"] + standardized_metrics]
                .groupby("bioregion")
                .transform(zscore)
                .fillna(0)
            )

        for metric in metrics:
            self.df_cells[f"{metric}_raw"] = self.df_cells[metric]

            # Standardize across all grid cells for clustering
            self.df_cells[f"{metric}_z"] = zscore(self.df_cells[metric])
            self.df_cells[f"{metric}_z"].fillna(0)

        self.metrics = metrics
        self.metrics_raw = [f"{metric}_raw" for metric in metrics]
        self.metrics_z = [f"{metric}_z" for metric in metrics]

        # if not categorize:
        #     return

        # Bin values to three intervals with equal amount of samples
        values = ["Low", "Medium", "High"]
        for metric in self.metrics:
            s = self.df_cells[metric].sort_values()
            mask_low = np.full((s.size,), False)
            q3 = s.size // 3
            mask_low[:q3] = True
            mask_medium = np.full((s.size,), False)
            mask_medium[q3 : 2 * q3] = True
            mask_medium
            mask_high = ~mask_low & ~mask_medium
            self.df_cells.loc[s.index[mask_low], metric] = "Low"
            self.df_cells.loc[s.index[mask_medium], metric] = "Medium"
            self.df_cells.loc[s.index[mask_high], metric] = "High"

        # values = ["Low", "Medium", "High"]
        # for metric in self.metrics:
        #     s = self.df_cells[metric].sort_values()
        #     self.df_cells[metric] = pd.cut(
        #         s,
        #         [s.min(), s.iloc[s.size // 3], s.iloc[2 * s.size // 3], s.max()],
        #         include_lowest=True,
        #         labels=values,
        #     )

        # sort values to avoid PerformanceWarning: indexing past lexsort depth may impact performance.
        values.sort()
        outer_dims = ["biota_overlap", "endemicity"]
        inner_dims = ["species_richness", "occupancy"]

        index = pd.MultiIndex.from_product(
            [values, values], names=[outer_dims[0], inner_dims[0]]
        )
        columns = pd.MultiIndex.from_product(
            [values, values], names=[outer_dims[1], inner_dims[1]]
        )
        df = pd.DataFrame(np.zeros((9, 9), dtype=int), index=index, columns=columns)

        i_col, j_col = outer_dims
        df0 = pd.DataFrame(np.zeros((3, 3), dtype=int), index=values, columns=values)
        df0.index.name = inner_dims[0]
        df0.columns.name = inner_dims[1]
        X = self.df_cells[metrics]

        for i_val in values:
            df_outer = X[X[i_col] == i_val]
            for j_val in values:
                df_inner_incomplete = df_outer[df_outer[j_col] == j_val][inner_dims]
                df_inner = df0.copy()
                df_inner.update(df_inner_incomplete.value_counts().unstack())
                df.loc[(i_val,), (j_val,)] = df_inner.astype(int).values

        self.df_metrics = df

    def plot_metrics_raw(self, cluster=True, method="DBSCAN"):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        metrics = self.metrics_z

        if cluster:
            if method == "DBSCAN":
                X = self.df_cells[metrics].copy()
                db = skl_cluster.DBSCAN(eps=0.3, min_samples=10).fit(X)
                labels = db.labels_

                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)

                print("Estimated number of clusters: %d" % n_clusters_)
                print("Estimated number of noise points: %d" % n_noise_)

                unique_labels = set(labels)
                num_labels = len(unique_labels)
                grouped_labels = False
                # Group if too many
                if num_labels > 9:
                    print(f"NOTE: {num_labels - 11} labels grouped in label 10")
                    labels[labels > 9] = 10
                    unique_labels = set(labels)
                    num_labels = len(unique_labels)
                    grouped_labels = True
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                X["core"] = core_samples_mask
                X["label"] = labels
                X["core_strength"] = X["core"].astype(int) + 1
                palette = sns.color_palette(n_colors=num_labels)
                palette = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
                palette[-1] = (0, 0, 0)
                if grouped_labels:
                    palette[10] = (0.7, 0.7, 0.7)
                for i, (x, y), ax in zip(
                    range(6), it.combinations(metrics, 2), axes.flatten()
                ):
                    draw_legend = (i + 1) % 3 == 0
                    sns.scatterplot(
                        X,
                        x=x,
                        y=y,
                        hue="label",
                        size="core_strength",
                        ax=ax,
                        legend=draw_legend,
                        palette=palette,
                    )
                    r = self.df_cells[[x, y]].corr().iloc[0, 1]
                    ax.set(title=f"r = {r:.2f}", xlabel=x[:-2], ylabel=y[:-2])
                    if draw_legend:
                        ax.legend(bbox_to_anchor=(1, 1))
                fig.suptitle(
                    f"{n_clusters_} clusters and {n_noise_}/{X.shape[0]} noise points"
                )
            else:
                # K-Means clustering
                X = self.df_cells[metrics].copy()
                n_clusters = 7
                if method == "gmm":
                    model = GaussianMixture(n_components=n_clusters)
                else:
                    model = skl_cluster.KMeans(n_clusters=n_clusters, n_init="auto")
                labels = model.fit_predict(X)
                unique_labels = set(labels)
                num_labels = len(unique_labels)
                X["label"] = labels
                palette = sns.color_palette(n_colors=num_labels)
                for i, (x, y), ax in zip(
                    range(6), it.combinations(metrics, 2), axes.flatten()
                ):
                    draw_legend = (i + 1) % 3 == 0
                    sns.scatterplot(
                        X,
                        x=x,
                        y=y,
                        hue="label",
                        ax=ax,
                        legend=draw_legend,
                        palette=palette,
                    )
                    r = self.df_cells[[x, y]].corr().iloc[0, 1]
                    ax.set(title=f"r = {r:.2f}", xlabel=x[:-2], ylabel=y[:-2])
                    if draw_legend:
                        ax.legend(bbox_to_anchor=(1, 1))
                fig.suptitle(f"{n_clusters} clusters with {method}")
        else:
            X = self.df_cells[["bioregion"] + metrics]
            num_bioregions = X["bioregion"].nunique()
            for i, (x, y), ax in zip(
                range(6), it.combinations(metrics, 2), axes.flatten()
            ):
                draw_legend = (i + 1) % 3 == 0
                sns.scatterplot(
                    X,
                    x=x,
                    y=y,
                    hue="bioregion",
                    ax=ax,
                    legend=draw_legend,
                    palette="deep" if num_bioregions < 13 else "flare",
                )
                r = self.df_cells[[x, y]].corr().iloc[0, 1]
                ax.set(title=f"r = {r:.2f}", xlabel=x[:-4], ylabel=y[:-4])
                if draw_legend:
                    ax.legend(bbox_to_anchor=(1, 1))
        fig.tight_layout()

    def plot_metrics_categorical(self):
        fig, axes = plt.subplots(
            nrows=3, ncols=3, sharex=True, sharey=True, figsize=(8, 8)
        )

        effective_number_of_combinations = perplexity(self.df_metrics.values.flatten())
        max_value = self.df_metrics.values.max()
        values = ["Low", "Medium", "High"]
        # outer_dims = ["endemicity", "biota_overlap"]
        # inner_dims = ['species_richness', 'occupancy']
        # i_col, j_col = outer_dims

        for i, i_val in enumerate(values):  # rows: biota_overlap
            for j, j_val in enumerate(values):  # columns: endemicity
                ax = axes[i][j]
                df = self.df_metrics.loc[(i_val,), (j_val,)].loc[values, values]
                sns.heatmap(
                    df, ax=ax, cbar=j == 2, vmin=0, vmax=max_value, cmap="flare"
                )
                if i != 2:
                    ax.set(xlabel=None)
                    if i == 0:
                        ax.set(title=f"endemicity {j_val}")
                if j != 0:
                    ax.set(ylabel=None)
                else:
                    ylabel = ax.yaxis.get_label().get_text()
                    ax.set_ylabel(f"biota_overlap {i_val}\n{ylabel}")

        random_data = [
            perplexity(sample_multinomial(n=self.df_cells.shape[0], k=81))
            for _ in range(1000)
        ]
        rand_mean, rand_std = np.mean(random_data), np.std(random_data)
        fig.suptitle(
            f"Effective number of combinations: {round(effective_number_of_combinations)} of 81"
        )
        fig.text(
            0.5,
            1,
            f"(M={rand_mean:.1f}, SD={rand_std:.1f} expected by chance)",
            ha="center",
        )
        fig.tight_layout()

    @staticmethod
    def wide_to_long(X):
        return X.melt(var_name="metric", value_name="score")

    @staticmethod
    def get_cluster_color(X, lightness_min=0.3, lightness_max=0.75):
        df_cluster = X.groupby("cluster").agg("median")

        s = df_cluster["species_richness_z"]
        s = (s - s.min()) / (s.max() - s.min())
        df_cluster["species_richness_p"] = s

        s = df_cluster["biota_overlap_z"]
        s = (s - s.min()) / (s.max() - s.min())
        df_cluster["biota_overlap_p"] = s

        df_cluster = df_cluster.sort_values(["biota_overlap_z", "species_richness_z"])

        df_cluster["color"] = df_cluster.apply(
            lambda row: get_red_blue_color(
                row["biota_overlap_p"],
                row["species_richness_p"],
                lightness_min=lightness_min,
                lightness_max=lightness_max,
            ),
            axis=1,
        )
        return df_cluster


def plot_geopandas(
    df_geo,
    column,
    categorical=True,
    width=12,
    title="",
    ax=None,
    ec_cells="#999999",
    linewidth=0.1,
):
    if ax is None:
        _, ax = plt.subplots(
            figsize=(width, width)
        )  # second w is height but map aspect ratio will override that

    ax.grid(False)
    ax.axis("off")

    num_categories = 0
    if categorical:
        num_categories = df_geo[column].nunique()

    cmap = mpl.colors.ListedColormap(colors[: num_categories + 1], name="c3")
    df_geo.plot(
        column=column,
        ax=ax,
        categorical=True,
        ec=ec_cells,
        linewidth=linewidth,
        cmap=cmap,
    )

    ax.set_title(title)
    return ax


def plot_geopandas2(
    df_geo,
    color_column="color",
    width=12,
    title="",
    ax=None,
    ec_cells="#999999",
    linewidth=0.1,
):
    if ax is None:
        _, ax = plt.subplots(
            figsize=(width, width)
        )  # second w is height but map aspect ratio will override that

    ax.grid(False)
    ax.axis("off")

    df_geo.plot(
        color=df_geo[color_column],
        ax=ax,
        categorical=True,
        ec=ec_cells,
        linewidth=linewidth,
    )

    ax.set_title(title)
    return ax
