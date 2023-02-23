import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'results')

PLOT_DICTS = [
    {
     'title': 'Linear Gaussian BNs graphs',
     'file_name': 'lingauss_metrics',
     'metrics': ['mixture_esid',  'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [("ER", [
            ("JCI-PC", "lingauss1405_20nodes_v1/synthetic_JCI-PC_unknown_interv/1273192460692766319"),
            ("UT-IGSP", "lingauss1405_20nodes_v1/synthetic_IGSP_unknown_interv/1680928129342933286"),
            ("DCDI-G", "lingauss1905_20nodes_v1/synthetic_DCDI-G_unknown_interv/9119843253167383750"),
            (r"$\bf{BaCaDI}$ (LinG)", "lingauss3007_20nodes_v2/synthetic_bacadi_lingauss_unknown_interv/6720261755586691985"),
      ]),
      ("SF-2", [
            ("JCI-PC", "lingauss1405_20nodes_v1_copy/synthetic_JCI-PC_unknown_interv/598322374597402408"),
            ("UT-IGSP", "lingauss1405_20nodes_v1_copy/synthetic_IGSP_unknown_interv/5919574287101225375"),
            ("DCDI-G", "lingauss1905_20nodes_v1/synthetic_DCDI-G_unknown_interv/668539336521743855"),
            (r"$\bf{BaCaDI}$ (LinG)", "lingauss3007_20nodes_v1/synthetic_bacadi_lingauss_unknown_interv/7996752904207021747"),
        ])
       ]
     },

    {'title': 'Nonlinear Gaussian BNs graphs',
     'file_name': 'fcgauss_metrics',
     'metrics': ['mixture_esid', 'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [("ER", [
            ("JCI-PC", "fcgauss1405_20nodes_v1_copy/synthetic_JCI-PC_unknown_interv/29249582541447628"),
            ("UT-IGSP", "fcgauss1405_20nodes_v1/synthetic_IGSP_unknown_interv/1521488760567248419"),
            ("DCDI-G", "fcgauss1905_20nodes_v1/synthetic_DCDI-G_unknown_interv/9184752844498045151"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "fcgauss3007_20nodes_v1/synthetic_bacadi_fcgauss_unknown_interv/7359201081573045649"),
            ]),
      ("SF-2", [
            ("JCI-PC", "fcgauss1405_20nodes_v1_copy/synthetic_JCI-PC_unknown_interv/1323406223509194926"),
            ("UT-IGSP", "fcgauss1405_20nodes_v1/synthetic_IGSP_unknown_interv/2174141621805202226"),
            ("DCDI-G", "fcgauss1905_20nodes_v1/synthetic_DCDI-G_unknown_interv/279654528501157411"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "fcgauss3007_20nodes_v1/synthetic_bacadi_fcgauss_unknown_interv/7062532024974682193"),
        ])
       ]
     },


     {   
     'title': 'SERGIO graphs',
     'file_name': 'sergio_metrics',
     'metrics': ['mixture_esid', 'mixture_negll_interv', 'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [
      ("SF-2", [
            ("JCI-PC", "sergio_joint1405_20nodes_v2/sergio_JCI-PC_unknown_interv/2398737007030405362"),
            ("UT-IGSP", "sergio_joint1405_20nodes_v2/sergio_IGSP_unknown_interv/5504380058496549900"),
            ("DCDI-G", "sergio_joint1905_20nodes_v1/sergio_DCDI-G_unknown_interv/5910629912283174740"),
            (r"$\bf{BaCaDI}$ (LinG)", "sergio3007_20nodes_v1/sergio_bacadi_lingauss_unknown_interv/631707605595472068"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "sergio3007_20nodes_v1/sergio_bacadi_fcgauss_unknown_interv/8849191469108681052"),
            ])
       ]
     },
]


PLOT_DICTS_APPENDIX = [
    {
     'title': 'BGe',
     'file_name': 'bge_metrics',
     'metrics': ['mixture_esid', 'mixture_negll_interv', 'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [("ER", [
            ("JCI-PC", "bge1405_20nodes_v1/synthetic_JCI-PC_unknown_interv/1675836751522820581"),
            ("UT-IGSP", "bge1405_20nodes_v1/synthetic_IGSP_unknown_interv/4271222230025986197"),
            (r"$\bf{BaCaDI}$ (BGe)", "bge1405_20nodes_v1/synthetic_bacadi_unknown_interv/8823518397427194794"),
         ]),
      ("SF-2", [
            ("JCI-PC", "bge1405_20nodes_v1/synthetic_JCI-PC_unknown_interv/5352753345304681777"),
            ("UT-IGSP", "bge1405_20nodes_v1/synthetic_IGSP_unknown_interv/673597940890787814"),
            (r"$\bf{BaCaDI}$ (BGe)", "bge1405_20nodes_v1/synthetic_bacadi_unknown_interv/1423981259255717569"),
         ])
       ]
     },

    {
     'title': 'Obs. vs. known interv. vs unknown interv.',
     'file_name': 'obs_interv_metrics',
     'metrics': ['mixture_esid', 'mixture_negll_interv', 'mixture_auprc', ],
     "plot_dirs": [("ER", [
            (r"$\bf{BaCaDI}$ (NonlinG) observ. data", "fcgauss_obs2505_20nodes_v1/synthetic_bacadi_fcgauss_obs/3772708703403832763"),
            (r"$\bf{BaCaDI}$ (NonlinG) known interv.", "fcgauss_obs2505_20nodes_v1/synthetic_bacadi_fcgauss_known_interv/5411304040418703688"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "fcgauss1405_20nodes_v1_copy/synthetic_bacadi_fcgauss_unknown_interv/2649460096289106393"),
         ]),
      ("SF-2", [
            (r"$\bf{BaCaDI}$ (NonlinG) observ. data", "fcgauss_obs2505_20nodes_v1/synthetic_bacadi_fcgauss_obs/8142462468261839642"),
            (r"$\bf{BaCaDI}$ (NonlinG) known interv.", "fcgauss_obs2505_20nodes_v1/synthetic_bacadi_fcgauss_known_interv/4421677958936975178"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "fcgauss1405_20nodes_v1_copy/synthetic_bacadi_fcgauss_unknown_interv/8089626190405569154"),
         ])
       ]
     },
    {
     'title': 'Linear Gaussian BNs graphs',
     'file_name': 'lingauss_metrics_doubledata',
     'metrics': ['mixture_esid', 'mixture_negll_interv', 'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [("ER", [
            ("JCI-PC", "lingauss_doubledata2305_20nodes_v1/synthetic_JCI-PC_unknown_interv/3326226036902143697"),
            ("UT-IGSP", "lingauss_doubledata2305_20nodes_v1/synthetic_IGSP_unknown_interv/1084497292091741578"),
            ("DCDI-G", "lingauss_doubledata2305_20nodes_v1/synthetic_DCDI-G_unknown_interv/468958850887325706"),
            (r"$\bf{BaCaDI}$ (LinG)", "lingauss_doubledata2305_20nodes_v1/synthetic_bacadi_lingauss_unknown_interv/2802285971495641188"),
        ]),
      ("SF-2", [
            ("JCI-PC", "lingauss_doubledata2305_20nodes_v1/synthetic_JCI-PC_unknown_interv/7615918075523183064"),
            ("UT-IGSP", "lingauss_doubledata2305_20nodes_v1/synthetic_IGSP_unknown_interv/5424687717926729517"),
            ("DCDI-G", "lingauss_doubledata2305_20nodes_v1/synthetic_DCDI-G_unknown_interv/6876724272590492415"),
            (r"$\bf{BaCaDI}$ (LinG)", "lingauss_doubledata2305_20nodes_v1/synthetic_bacadi_lingauss_unknown_interv/7070357337793012323"),
        ])
       ]
     },

    {'title': 'Nonlinear Gaussian BNs graphs',
     'file_name': 'fcgauss_metrics_doubledata',
     'metrics': ['mixture_esid', 'mixture_negll_interv', 'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [("ER", [
            ("JCI-PC", "fcgauss_doubledata2305_20nodes_v1/synthetic_JCI-PC_unknown_interv/1138745440487275877"),
            ("UT-IGSP", "fcgauss_doubledata2305_20nodes_v1/synthetic_IGSP_unknown_interv/3465474416686618199"),
            ("DCDI-G", "fcgauss_doubledata2305_20nodes_v1/synthetic_DCDI-G_unknown_interv/2877370592787819327"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "fcgauss_doubledata2305_20nodes_v1/synthetic_bacadi_fcgauss_unknown_interv/6473987027191733079"),
        ]),
      ("SF-2", [
            ("JCI-PC", "fcgauss_doubledata2305_20nodes_v1/synthetic_JCI-PC_unknown_interv/6504988270652523582"),
            ("UT-IGSP", "fcgauss_doubledata2305_20nodes_v1/synthetic_IGSP_unknown_interv/4704454105620319481"),
            ("DCDI-G", "fcgauss_doubledata2305_20nodes_v1/synthetic_DCDI-G_unknown_interv/3990305195171738525"),
            (r"$\bf{BaCaDI}$ (NonlinG)", "fcgauss_doubledata2305_20nodes_v1/synthetic_bacadi_fcgauss_unknown_interv/8601957485813222417"),
        ])
       ]
     },

    {'title': 'Linear Gaussian BNs graphs 50 nodes',
     'file_name': 'lingauss_metrics_50nodes_doubledata',
     'metrics': ['mixture_esid', 'mixture_negll_interv', 'mixture_auprc', 'mixture_intv_auprc', ],
     "plot_dirs": [("ER", [
            ("JCI-PC", "lingauss_doubledata2305_50nodes_v2/synthetic_JCI-PC_unknown_interv/8605751912189358583"),
            ("UT-IGSP", "lingauss_doubledata2305_50nodes_v1/synthetic_IGSP_unknown_interv/9106248906740104360"),
            ("DCDI-G", "lingauss_doubledata2305_50nodes_v1/synthetic_DCDI-G_unknown_interv/5271959378956765883"),
            (r"$\bf{BaCaDI}$ (LinG)", "lingauss_doubledata2305_50nodes_v1/synthetic_bacadi_lingauss_unknown_interv/3129840835491746350"),
        ]),
      ("SF-2", [
            ("JCI-PC", "lingauss_doubledata2305_50nodes_v2/synthetic_JCI-PC_unknown_interv/1604790696860310108"),
            ("UT-IGSP", "lingauss_doubledata2305_50nodes_v1/synthetic_IGSP_unknown_interv/7933974540725944882"),
            ("DCDI-G", "lingauss_doubledata2305_50nodes_v1/synthetic_DCDI-G_unknown_interv/2660051591581849616"),
            (r"$\bf{BaCaDI}$ (LinG)", "lingauss_doubledata2305_50nodes_v1/synthetic_bacadi_lingauss_unknown_interv/8121522171639233805"),
        ])
       ]
     },
]
