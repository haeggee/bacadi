metric_to_tex = {
    'mixture_esid': r'$\mathbb{E}$' + r'-SID',
    'mixture_negll_interv': r'I-NLL',
    'mixture_auprc': 'AUPRC',
    'mixture_intv_auprc': 'INTV-AUPRC',
}

method_to_linestyle = {
    'bacadi_lingauss': '--',
    r"$\bf{BaCaDI}$ (LinG)": '--',
    'bacadi_fcgauss': '-.',
    r"$\bf{BaCaDI}$ (NonlinG)": '-.',
    'JCI-PC': '-',
    'IGSP': '--',
    'UT-IGSP': '--',
    'DCDI-G': ':',
}

method_to_tex = {
    'bacadi': r"$\bf{BaCaDI}$",
    'bacadi_lingauss': r"$\bf{BaCaDI}$ LinG",
    'bacadi_fcgauss': r"$\bf{BaCaDI}$ NonlinG",
    'JCI-PC': r'$\textit{B-}$JCI-PC',
    'IGSP':     r'$\textit{B-}$UT-IGSP',
    'DCDI-G':   r'$\textit{B-}$DCDI-G',
    'UT-IGSP':  r'$\textit{B-}$UT-IGSP',
}

import seaborn as sns

palette = sns.color_palette("colorblind")
palette_sequential = sns.color_palette("crest")

method_to_color = {
    'bacadi_lingauss':palette[2],
    r"$\bf{BaCaDI}$": palette[2],
    r"$\bf{BaCaDI}$ (LinG)": palette[2],
    r"$\bf{BaCaDI}$ (NonlinG)": palette[4],
    'bacadi_fcgauss': palette[4],
    r"$\bf{BaCaDI}$ (BGe)": palette[6],
    "DCDI-G": palette[3],
    "UT-IGSP": palette[0],
    "IGSP": palette[0],
    "JCI-PC": palette[1],
    # sequential ones
    'bacadi_0': palette_sequential[0],
    'bacadi_1': palette_sequential[1],
    'bacadi_2': palette_sequential[2],
    'bacadi_3': palette_sequential[3],
    'bacadi_4': palette_sequential[4],
    'bacadi_5': palette_sequential[5],
}


COLOR_SATURATION = 0.8

DPI = 300

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_TRIPLE = (COL_WIDTH / 3, COL_WIDTH / 3 * 4/6)
FIG_SIZE_TRIPLE_TALL = (COL_WIDTH / 3, COL_WIDTH / 3 * 5/6)

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
FIG_SIZE_DOUBLE_TALL = (COL_WIDTH / 2, COL_WIDTH / 2 * 5/6)

NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)

NEURIPS_RCPARAMS = {
    "figure.autolayout": True,       # `False` makes `fig.tight_layout()` not work
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    # "figure.dpi": DPI,             # messes up figisize
    # Axes params
    "axes.linewidth": 0.5,           # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 3.0, 
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsfonts}'
    ),
}