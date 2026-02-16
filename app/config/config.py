TOP_K_OLD = {
    0.7: (
        [
            1.0,
            2.5,
            5.0,
            8.5,
            13.0,
            18.0,
            25.5,
            40.5,
            63.0,
            88.0,
            125.5,
            200.5,
            375.5,
            625.5,
        ],
        [4.6, 15, 33, 55, 67, 72, 102, 152, 217, 417, 872, 493, 1585, 2700],
    )
}
TOP_K = {
    0.7: (
        [
            1.0,
            2.4,
            5.0,
            8.3,
            12.8,
            17.5,
            24.7,
            41.5,
            59.8,
            87.0,
            123.6,
            196.2,
            303.8,
            460.0,
            592.7,
        ],  # x
        # [17, 34, 27, 58, 99, 98, 218, 210, 362, 578, 875, 776, 2509, 1589, 2937], #y
        [
            17,
            31,
            31,
            58,
            98,
            98,
            214,
            214,
            362,
            578,
            826,
            826,
            2049,
            2049,
            2937,
        ],  # smoothed y
    )
}
FIXED_TOP_K=752
COSINE_PCT_THRESHOLD=0.0266

DEBUG = False  # TODO remove

FORBIDDEN_FEATURES = {"nan"}  # used as empty abstract in csmed

PREVIOUS_RUNS = [
    "data/statistics/optuna/run_10_nodes_10tasks_1cpu_per_task_opt_beta=6.0/optuna.db",
    "data/statistics/optuna/run_10_nodes_10tasks_1cpu_per_task_opt_beta=10.0/optuna.db"
]

BOW_PARAMS = {
    "lower_case": True,
    "mesh_ancestors": True,
    "rm_numbers": True,
    "rm_punct": True,
    "related_words": True,
    "total_docs": 503679,  # 433660,
    "min_df": 100,
    "max_df": 0.5,
    "mesh": True,
}

RF_PARAMS = {
    "top_k": 1.5,
    "top_k_type": "pos_count", # alternatives: "fixed", "cosine"
    "dont_cares": 2.0,
    "rank_weight": 1.5,  # how much more weighted shall rank 1 be than rank k
    "n_estimators": 50,  # TODO change back
    "max_depth": 4,
    "min_samples_split": 3,
    "min_weight_fraction_leaf": 0.0002,
    "max_features": 0.5,  # "sqrt",
    "randomize_max_feature": 0.9,
    "randomize_min_impurity_decrease_range": 0.9,
    "min_impurity_decrease_range_start": 0.001,
    "min_impurity_decrease_range_end": 0.001,
    "bootstrap": True,
    "n_jobs": None,
    "random_state": None,
    "verbose": False,
    "class_weight": 0.2,
    "max_samples": None,
    "top_k_or_candidates": 500,
    "prefer_pos_splits": 1.1,
    "max_or_features": 10,
}

QG_PARAMS = {
    "min_tree_occ": 0.05,
    "min_rule_occ": 0.02,
    "cost_factor": 0.002,  # 50 ANDs are worth 0.1 F3 score
    "min_rule_precision": 0.01,
    "cover_beta": 2.0,  # high to prefer covering training data fully
    "pruning_beta": 0.1,  # low to prefer precise rules
    "term_expansions": True,
    "mh_noexp": True,
    "tiab": True,
    "pruning_thresholds": {
        "or": {
            False: {  # removal in negated term false -> is positive term
                "acceptance_metric": "tp_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
            True: {
                "acceptance_metric": "precision_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
        },
        "and": {
            False: {
                "acceptance_metric": "precision_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
            True: {
                "acceptance_metric": "precision_gain",
                "acceptance_threshold": -0.1,
                "removal_threshold": -0.01,
            },
        },
    },
}

TAR2017_TRAIN = [
    "CD007394",
    "CD007427",
    "CD008054",
    "CD008643",
    "CD008686",
    "CD008691",
    "CD009020",
    "CD009323",
    "CD009591",
    "CD009593",
    "CD009944",
    "CD010409",
    "CD010438",
    "CD010632",
    "CD010771",
    "CD011134",
    "CD011548",
    "CD011549",
    "CD011975",
    "CD011984",
]
TAR2017_TEST = [
    "CD007431",
    "CD008081",
    "CD008760",
    "CD008782",
    "CD008803",
    "CD009135",
    "CD009185",
    "CD009372",
    "CD009519",
    "CD009551",
    "CD009579",
    "CD009647",
    "CD009786",
    "CD009925",
    "CD010023",
    "CD010173",
    "CD010276",
    "CD010339",
    "CD010386",
    "CD010542",
    "CD010633",
    "CD010653",
    "CD010705",
    "CD010772",
    "CD010775",
    "CD010783",
    "CD010860",
    "CD010896",
    "CD011145",
    "CD012019",
]
TAR2018_TRAIN = [
    "CD007394",
    "CD007427",
    "CD008054",
    "CD008081",
    "CD008643",
    "CD008686",
    "CD008691",
    "CD008760",
    "CD008782",
    "CD008803",
    "CD009020",
    "CD009135",
    "CD009185",
    "CD009323",
    "CD009372",
    "CD009519",
    "CD009551",
    "CD009579",
    "CD009591",
    "CD009593",
    "CD009647",
    "CD009786",
    "CD009925",
    "CD009944",
    "CD010023",
    "CD010173",
    "CD010276",
    "CD010339",
    "CD010386",
    "CD010409",
    "CD010438",
    "CD010542",
    "CD010632",
    "CD010633",
    "CD010653",
    "CD010705",
    "CD011134",
    "CD011549",
    "CD011975",
    "CD012019",
]
TAR2018_TEST = [
    "CD007424",
    "CD008122",
    "CD008587",
    "CD008759",
    "CD008892",
    "CD009175",
    "CD009263",
    "CD009694",
    "CD010213",
    "CD010296",
    "CD010360",
    "CD010502",
    "CD010657",
    "CD010680",
    "CD010803",
    "CD010864",
    "CD011053",
    "CD011126",
    "CD011420",
    "CD011431",
    "CD011515",
    "CD011602",
    "CD011686",
    "CD011912",
    "CD011926",
    "CD012009",
    "CD012010",
    "CD012083",
    "CD012165",
    "CD012179",
    "CD012216",
    "CD012281",
    "CD012599",
    "CD012645",
    "CD012883",
    "CD012884",
]
CSMED_COCHRANE_REVIEWS = {
    "sigir2017": [
        "CD009784",
        "CD010206",
        "CD002764",
        "CD011146",
        "CD010264",
        "CD011721",
        "CD009782",
        "CD009771",
        "CD006962",
        "CD007470",
        "CD009946",
        "CD006569",
        "CD006900",
        "CD010910",
        "CD007953",
        "CD003504",
        "CD003160",
        "CD011209",
        "CD007505",
        "CD009400",
        "CD006014",
        "CD003559",
        "CD001800",
        "CD010375",
        "CD010381",
        "CD004920",
        "CD011066",
        "CD007103",
        "CD003557",
        "CD009436",
        "CD004944",
        "CD004178",
        "CD007736",
        "CD005059",
        "CD010927",
        "CD011837",
        "CD003353",
        "CD009669",
        "CD001457",
        "CD010387",
        "CD009634",
        "CD009633",
        "CD011241",
        "CD005019",
        "CD002832",
        "CD004903",
        "CD003914",
        "CD002062",
        "CD002250",
        "CD007379",
        "CD011472",
        "CD009210",
        "CD010139",
        "CD010333",
        "CD009887",
        "CD006832",
        "CD008963",
        "CD005614",
        "CD010767",
        "CD006748",
        "CD007497",
        "CD008873",
        "CD006770",
        "CD010411",
        "CD008482",
        "CD001957",
        "CD006172",
        "CD010225",
        "CD010473",
        "CD002783",
        "CD007037",
        "CD008881",
        "CD006127",
        "CD011732",
        "CD002117",
        "CD009593",
        "CD000284",
        "CD010685",
        "CD003266",
        "CD008265",
        "CD010632",
        "CD010269",
        "CD004250",
        "CD009780",
        "CD005340",
        "CD008807",
        "CD004054",
        "CD009125",
        "CD011724",
        "CD010266",
        "CD007028",
        "CD000031",
        "CD008435",
        "CD008056",
        "CD000009",
        "CD005346",
        "CD010693",
        "CD006995",
        "CD010235",
        "CD007271",
        "CD000313",
        "CD009403",
        "CD008345",
        "CD003137",
        "CD008170",
        "CD002283",
        "CD005260",
        "CD006612",
        "CD002073",
        "CD008185",
        "CD006678",
        "CD007356",
        "CD001431",
        "CD004520",
        "CD009831",
        "CD004376",
        "CD002840",
        "CD000384",
        "CD010912",
        "CD007719",
        "CD004396",
        "CD006638",
        "CD007525",
        "CD010901",
        "CD011447",
        "CD002898",
        "CD012004",
        "CD003344",
        "CD007115",
        "CD010709",
        "CD008366",
        "CD007716",
        "CD002069",
        "CD003987",
        "CD009685",
        "CD009823",
        "CD001843",
        "CD002201",
        "CD008969",
        "CD011047",
        "CD007122",
        "CD011281",
        "CD010533",
        "CD010534",
        "CD007745",
        "CD001150",
        "CD010479",
        "CD008812",
        "CD003855",
        "CD005397",
        "CD010226",
        "CD003067",
        "CD008020",
        "CD002115",
        "CD000014",
        "CD007699",
        "CD006546",
        "CD008426",
        "CD006745",
        "CD006342",
        "CD008213",
        "CD004288",
        "CD004875",
        "CD010447",
        "CD010845",
        "CD009530",
        "CD009537",
        "CD002143",
        "CD010425",
        "CD011145",
    ],
    "tar2019": [
        "CD012567",
        "CD008081",
        "CD010653",
        "CD010239",
        "CD011549",
        "CD001261",
        "CD009925",
        "CD010633",
        "CD012551",
        "CD009185",
        "CD008803",
        "CD011571",
        "CD010864",
        "CD011380",
        "CD008054",
        "CD008892",
        "CD012930",
        "CD011926",
        "CD009323",
        "CD011787",
        "CD009519",
        "CD010296",
        "CD011975",
        "CD011515",
        "CD011140",
        "CD012216",
        "CD012083",
        "CD010386",
        "CD008587",
        "CD005253",
        "CD009694",
        "CD012281",
        "CD012009",
        "CD012455",
        "CD009642",
        "CD012669",
        "CD011420",
        "CD010705",
        "CD012233",
        # "CD012668",
        "CD012661",
        "CD005139",
        "CD010276",
        "CD010542",
        "CD012521",
        "CD008874",
        "CD006715",
        "CD010019",
        "CD012120",
        "CD010213",
        "CD007868",
        "CD008686",
        "CD012347",
        "CD007867",
        "CD011768",
        "CD009551",
        "CD009135",
        "CD010438",
        "CD012165",
        "CD011977",
        "CD009372",
        "CD010409",
        "CD009944",
        "CD010657",
        "CD009579",
        "CD011912",
        "CD012599",
        "CD012164",
        "CD009786",
        "CD010038",
        "CD012768",
        "CD007427",
        "CD011126",
        "CD011602",
        "CD012080",
        "CD011053",
        "CD009263",
        "CD009069",
        "CD012223",
        "CD010526",
        "CD011436",
        "CD011431",
        "CD010173",
        "CD012019",
        "CD010778",
        "CD007394",
        "CD012010",
        "CD010339",
        "CD011686",
        "CD008760",
        "CD010753",
        "CD009044",
        "CD006468",
        "CD010355",
        "CD008759",
        "CD009647",
        "CD010558",
        "CD000996",
        "CD009020",
        "CD010502",
        "CD012069",
        "CD010680",
        "CD011558",
        "CD008018",
        "CD009591",
        "CD004414",
        "CD012179",
        "CD012342",
        "CD011134",
        "CD010023",
    ],
    "sr_updates": [
        "CD004069",
        "CD008127",
        "CD001298",
        "CD010847",
        "CD004241",
        "CD010089",
        "CD007020",
        "CD006902",
        "CD005128",
        "CD007428",
        "CD002733",
        "CD005055",
        "CD008392",
        "CD006839",
        "CD005083",
        "CD004479",
    ],
}

TRAIN_REVIEWS_25 = [
    "CD008892",
    "CD010864",
    "CD010296",
    "CD011431",
    "CD012010",  # tar2018
    "CD008803",
    "CD010339",
    "CD009591",
    "CD008054",
    "CD010023",
    "CD007427",  # tar2017
    "CD012661",
    "CD009069",
    "CD001261",
    "CD005139",
    "CD008874",  # tar2019
    "CD007020",  # sr_updates
    "CD002783",
    "CD006962",
    "CD006127",
    "CD002117",
    "CD005397",
    "CD002840",
    "CD008963",
    "CD003344",  # sigir2017
]
TRAIN_REVIEWS_10_OLD = [
    "CD011431",
    "CD008892",  # tar2018 # starting with the biggest sample (297 positives)
    "CD008054",
    "CD007427",  # tar2017
    "CD012661",
    "CD005139",  # tar2019
    "CD007020",  # sr_updates
    "CD002783",
    "CD002117",
    "CD003344",  # sigir2017
]

TRAIN_REVIEWS = [ #atleast 50 positiives
    "CD011431",
    "CD008892",  # tar2018 # starting with the biggest sample (297 positives)
    "CD010339",
    "CD008054",
    "CD007427",  # tar2017
    "CD012661",
    "CD005139",  # tar2019
    "CD006127",
    "CD002117",
    "CD003344",  # sigir2017
]

# Matplotlib color palette for consistent styling across all visualizations
COLORS = {
    # Primary colors
    "primary": "#2E86AB",        # Steel blue
    "primary_light": "#A3D5FF",  # Light steel blue
    "secondary": "#1B998B",      # Teal/green
    "secondary_light": "#7DD3C0", # Light teal
    
    # Semantic colors
    "positive": "#2E86AB",       # Blue for positive/high values
    "positive_light": "#A3D5FF",
    "negative": "#E94F37",       # Red for negative/low values
    "negative_light": "#F5A69A",
    
    # Neutral colors
    "neutral": "#6C757D",        # Gray
    "neutral_light": "#ADB5BD",
    
    # Category colors (for multiple groups/lines in plots)
    "category": [
        "#2E86AB",  # Blue
        "#E94F37",  # Red
        "#1B993D",  # Green
        "#F4A261",  # Orange
        "#9B5DE5",  # Purple
        "#00F5D4",  # Cyan
        "#FEE440",  # Yellow
        "#F15BB5",  # Pink
    ],
    
    # Metric-specific colors (for consistent use across visualizations)
    "precision": "#2E86AB",      # Blue
    "precision_light": "#91bfdb",
    "recall": "#E94F37",         # Red  
    "recall_light": "#fd8383",
    "f_score": "#9B5DE5",        # Purple
    "time": "#1B993D",           # Green
    "query_size": "#F4A261",     # Orange
    "map": "#F15BB5",            # Pink
    "mrr": "#1B993D",            # Green
    # Top-k strategy colors
    "fixed_k": "#2E86AB",        
    "cosine_k": "#E94F37",       
    "pos_count_k": "#1B993D",   # Green 
}

# Colormaps for heatmaps and spectrum colors
COLORMAPS = {
    "heatmap": "Blues",#"YlGnBu",
    "diverging": "RdBu_r",
    "sequential": "Blues",
    "spectrum": "viridis",  # For ordered categorical data (e.g., buckets)
    "optuna": "Blues",#YlGnBu",    # Optuna default for contour/parallel coordinate (originally Blues_r?)
}

# Figure settings for consistent sizing across all plots
# A4 paper: 210mm width, with ~25mm margins on each side = 160mm text width ≈ 6.3 inches
FIGURE_CONFIG = {
    "full_width": 5.7834646,       # inches - full text width
    "half_width": 2.7417323,       # inches - for side-by-side figures
    "aspect_ratio": 0.75,    # height = width * aspect_ratio (for single plots)
    "dpi": 300,              # resolution for saving
    "font_size": 10,         # base font size in points
    "font_family": "serif",  # or "sans-serif"
}

def apply_matplotlib_style():
    """
    Apply consistent matplotlib styling for thesis figures.
    Call this at the start of any visualization script.
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        # Figure size
        "figure.figsize": (FIGURE_CONFIG["full_width"], 
                          FIGURE_CONFIG["full_width"] * FIGURE_CONFIG["aspect_ratio"]),
        "figure.dpi": FIGURE_CONFIG["dpi"],
        "savefig.dpi": FIGURE_CONFIG["dpi"],
        "savefig.bbox": "tight",
        
        # Font settings
        "font.size": FIGURE_CONFIG["font_size"],
        "font.family": FIGURE_CONFIG["font_family"],
        "axes.titlesize": FIGURE_CONFIG["font_size"],
        "axes.labelsize": FIGURE_CONFIG["font_size"],
        "xtick.labelsize": FIGURE_CONFIG["font_size"] - 1,
        "ytick.labelsize": FIGURE_CONFIG["font_size"] - 1,
        "legend.fontsize": FIGURE_CONFIG["font_size"] - 1,
        
        # Line and marker settings
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        
        # Axes settings
        "axes.linewidth": 0.8,
        "axes.grid": False,
        
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })
    
CURRENT_BEST_RUN_FOLDER = "data/statistics/optuna/best_3"

