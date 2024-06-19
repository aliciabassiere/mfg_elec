{
    "hours" : 12,
    "tmin" : 0,
    "tmax" : 25, # simulation length in years
    "Nt" : 300, # Time grid size
    "discount rate": 0.08,
    "power" : 0.8, # parameters of the weight function in fictitious play
    "offset" : 5, # parameters of the weight function in fictitious play
    "iterations" : 350, # Number of iterations in fictitious play
    "tolerance" : 50, # The algorithm stops if objective improvement is less than tolerance or after the number of iterations is reached
    "carbon tax" : ([0,12,25],[50, 130, 200]), # Carbon tax is computed by linear interpolation: the first list contains dates and the second one values
    #"carbon tax" : ([0,12,25],[0, 0, 0]),
    #"res subsidy" :([0,5,25],[60, 73.40, 0]),
    "res subsidy" : ([0,12,25], [0,0,0]),
    "demand ratio" : 1.0994063118847284, #1.257580634655325, # Ratio peak to offpeak demand
    "Nfuels" : 2, # Number of different fuels
     "Fsupply" : ([-4.5, -5.5], [0.25, 0.25]), # Fuel supply functions; they are linear in this version of the model;
    # the first array contains the intercepts and the second one the coefficients

    "demand" : [ 60.55175556,  56.50764516,  64.14027867,  66.23697817,
            65.05537455,  58.0718414 ,  58.35037222,  64.70375986,
            57.55837366,  65.02603148,  61.30344982,  58.41502685,
            61.94444593,  57.807321  ,  65.61550508,  67.76042867,
            66.55164817,  59.40749375,  59.69243078,  66.19194633,
            58.88221625,  66.52163021,  62.71342917,  59.75857247,
            63.36916819,  59.13688938,  67.1246617 ,  69.31891853,
            68.08233607,  60.77386611,  61.06535669,  67.7143611 ,
            60.23650722,  68.0516277 ,  64.15583804,  61.13301964,
            64.82665906,  60.49703784,  68.66852892,  70.91325366,
            69.6482298 ,  62.17166503,  62.4698599 ,  69.2717914 ,
            61.62194689,  69.61681514,  65.63142231,  62.53907909,
            66.31767222,  61.88846971,  70.24790508,  72.54425849,
            71.25013909,  63.60161332,  63.90666667,  70.86504261,
            63.03925167,  71.21800189,  67.14094503,  63.97747791,
            67.84297868,  63.31190451,  71.8636069 ,  74.21277644,
            72.88889229,  65.06445043,  65.37652001,  72.49493859,
            64.48915446,  72.85601593,  68.68518676,  65.4489599 ,
            69.40336719,  64.76807832,  73.51646986,  75.9196703 ,
            74.56533681,  66.56093279,  66.88017997,  74.16232217,
            65.97240501,  74.5317043 ,  70.26494606,  66.95428598,
            70.99964463,  66.25774412,  75.20734867,  77.66582271,
            76.28033956,  68.09183424,  68.41842411,  75.86805558,
            67.48977032,  76.24593349,  71.88103982,  68.49423455,
            72.63263646,  67.78167223,  76.93711769,  79.45213664,
            78.03478737,  69.65794643,  69.99204786,  77.61302086,
            69.04203504,  77.99958996,  73.53430373,  70.06960195,
            74.3031871 ,  69.34065069,  78.70667139,  81.27953578,
            79.82958748,  71.2600792 ,  71.60186496,  79.39812034,
            70.63000185,  79.79358053,  75.22559272,  71.68120279,
            76.0121604 ,  70.93548566,  80.51692484,  83.1489651 ,
            81.66566799,  72.89906102,  73.24870786,  81.22427711,
            72.25449189,  81.62883289,  76.95578135,  73.32987046,
            77.76044009,  72.56700183,  82.36881411,  85.0613913 ,
            83.54397835,  74.57573942,  74.93342814,  83.09243548,
            73.9163452 ,  83.50629604,  78.72576432,  75.01645748,
            79.54893021,  74.23604287,  84.26329683,  87.0178033 ,
            85.46548986,  76.29098143,  76.65689698,  85.0035615 ,
            75.61642114,  85.42694085,  80.5364569 ,  76.741836  ,
            81.37855561,  75.94347186,  86.20135266,  89.01921277,
            87.43119612,  78.045674  ,  78.42000561,  86.95864341,
            77.35559883,  87.39176049,  82.38879541,  78.50689823,
            83.25026239,  77.69017171,  88.18398377,  91.06665467,
            89.44211363,  79.8407245 ,  80.22366574,  88.95869221,
            79.1347776 ,  89.40177098,  84.2837377 ,  80.31255689,
            85.16501842,  79.47704566,  90.2122154 ,  93.16118772,
            91.49928225,  81.67706117,  82.06881005,  91.00474213,
            80.95487749,  91.45801171,  86.22226367,  82.1597457 ,
            87.12381384,  81.30501771,  92.28709635,  95.30389504,
            93.60376574,  83.55563358,  83.95639269,  93.0978512 ,
            82.81683967,  93.56154598,  88.20537574,  84.04941985,
            89.12766156,  83.17503312,  94.40969957,  97.49588463,
            95.75665235,  85.47741315,  85.88738972,  95.23910178,
            84.72162698,  95.71346154,  90.23409938,  85.9825565 ,
            91.17759778,  85.08805888,  96.58112266,  99.73828997,
            97.95905535,  87.44339365,  87.86279968,  97.42960112,
            86.6702244 ,  97.91487116,  92.30948366,  87.9601553 ,
            93.27468253,  87.04508423,  98.80248848, 102.03227064,
           100.21211363,  89.4545917 ,  89.88364407,  99.67048195,
            88.66363956, 100.16691319,  94.43260179,  89.98323887,
            95.42000023,  89.04712117, 101.07494571, 104.37901287,
           102.51699224,  91.51204731,  91.95096789, 101.96290303,
            90.70290327, 102.4707522 ,  96.60455163,  92.05285337,
            97.61466023,  91.09520496, 103.39966946, 106.77973017,
           104.87488306,  93.6168244 ,  94.06584015, 104.3080498 ,
            92.78907005, 104.8275795 ,  98.82645632,  94.170069  ,
            99.85979742,  93.19039467, 105.77786186, 109.23566396,
           107.28700537,  95.77001136,  96.22935447, 106.70713495,
            94.92321866, 107.23861383, 101.09946481,  96.33598058,
           102.15657276,  95.33377375, 108.21075268, 111.74808423,
           109.7546065 ,  97.97272162,  98.44262963, 109.16139905,
            97.10645269, 109.70510194, 103.4247525 ,  98.55170814,
           104.50617393,  97.52645055, 110.6996    , 114.31829017,
           112.27896245, 100.22609422, 100.70681011, 111.67211123,
            99.3399011 , 112.22831929, 105.80352181, 100.81839742],
    # "demand" : [60.55175556, 56.50764516, 64.14027867, 66.23697817, 65.05537455,
    #    58.0718414 , 58.35037222, 64.70375986,
    #     57.55837366,  65.02603148,  61.30344982,  58.41502685,
    #     61.94444593,  57.807321  ,  65.61550508,  67.76042867,
    #     66.55164817,  59.40749375,  59.69243078,  66.19194633,
    #     58.88221625,  66.52163021,  62.71342917,  59.75857247,
    #     63.36916819,  59.13688938,  67.1246617 ,  69.31891853,
    #     68.08233607,  60.77386611,  61.06535669,  67.7143611 ,
    #     60.23650722,  68.0516277 ,  64.15583804,  61.13301964,
    #     64.82665906,  60.49703784,  68.66852892,  70.91325366,
    #     69.6482298 ,  62.17166503,  62.4698599 ,  69.2717914 ,
    #     61.62194689,  69.61681514,  65.63142231,  62.53907909,
    #     66.31767222,  61.88846971,  70.24790508,  72.54425849,
    #     71.25013909,  63.60161332,  63.90666667,  70.86504261,
    #     63.03925167,  71.21800189,  67.14094503,  63.97747791,
    #     67.84297868,  63.31190451,  71.8636069 ,  74.21277644,
    #     72.88889229,  65.06445043,  65.37652001,  72.49493859,
    #     64.48915446,  72.85601593,  68.68518676,  65.4489599 ,
    #     69.40336719,  64.76807832,  73.51646986,  75.9196703 ,
    #     74.56533681,  66.56093279,  66.88017997,  74.16232217,
    #     65.97240501,  74.5317043 ,  70.26494606,  66.95428598,
    #     70.99964463,  66.25774412,  75.20734867,  77.66582271,
    #     76.28033956,  68.09183424,  68.41842411,  75.86805558,
    #     67.48977032,  76.24593349,  71.88103982,  68.49423455,
    #     72.63263646,  67.78167223,  76.93711769,  79.45213664,
    #     78.03478737,  69.65794643,  69.99204786,  77.61302086,
    #     69.04203504,  77.99958996,  73.53430373,  70.06960195,
    #     74.3031871 ,  69.34065069,  78.70667139,  81.27953578,
    #     79.82958748,  71.2600792 ,  71.60186496,  79.39812034,
    #     70.63000185,  79.79358053,  75.22559272,  71.68120279,
    #     76.0121604 ,  70.93548566,  80.51692484,  83.1489651 ,
    #     81.66566799,  72.89906102,  73.24870786,  81.22427711,
    #     72.25449189,  81.62883289,  76.95578135,  73.32987046,
    #     77.76044009,  72.56700183,  82.36881411,  85.0613913 ,
    #     83.54397835,  74.57573942,  74.93342814,  83.09243548,
    #     73.9163452 ,  83.50629604,  78.72576432,  75.01645748,
    #     79.54893021,  74.23604287,  84.26329683,  87.0178033 ,
    #     85.46548986,  76.29098143,  76.65689698,  85.0035615 ,
    #     75.61642114,  85.42694085,  80.5364569 ,  76.741836  ,
    #     81.37855561,  75.94347186,  86.20135266,  89.01921277,
    #     87.43119612,  78.045674  ,  78.42000561,  86.95864341,
    #     77.35559883,  87.39176049,  82.38879541,  78.50689823,
    #     83.25026239,  77.69017171,  88.18398377,  91.06665467,
    #     89.44211363,  79.8407245 ,  80.22366574,  88.95869221,
    #     79.1347776 ,  89.40177098,  84.2837377 ,  80.31255689,
    #     85.16501842,  79.47704566,  90.2122154 ,  93.16118772,
    #     91.49928225,  81.67706117,  82.06881005,  91.00474213,
    #     80.95487749,  91.45801171,  86.22226367,  82.1597457 ,
    #     87.12381384,  81.30501771,  92.28709635,  95.30389504,
    #     93.60376574,  83.55563358,  83.95639269,  93.0978512 ,
    #     82.81683967,  93.56154598,  88.20537574,  84.04941985,
    #     89.12766156,  83.17503312,  94.40969957,  97.49588463,
    #     95.75665235,  85.47741315,  85.88738972,  95.23910178,
    #     84.72162698,  95.71346154,  90.23409938,  85.9825565 ,
    #     91.17759778,  85.08805888,  96.58112266,  99.73828997,
    #     97.95905535,  87.44339365,  87.86279968,  97.42960112,
    #     86.6702244 ,  97.91487116,  92.30948366,  87.9601553 ,
    #     93.27468253,  87.04508423,  98.80248848, 102.03227064,
    #    100.21211363,  89.4545917 ,  89.88364407,  99.67048195,
    #     88.66363956, 100.16691319,  94.43260179,  89.98323887],
}
