{
    "Nt" : 101, # Time grid size
    "tmin" : 0,
    "tmax" : 25, # simulation length in years
    "power" : 0.8, # parameters of the weight function in fictitious play
    "offset" : 5, # parameters of the weight function in fictitious play
    "iterations" : 200, # Number of iterations in fictitious play
    "tolerance" : 100, # The algorithm stops if objective improvement is less than tolerance or after the number of iterations is reached
    "carbon tax" : ([0,1025],[0, 130, 250]), # Carbon tax is computed by linear interpolation: the first list contains dates and the second one values
    #"carbon tax" : ([10,20,25],[0, 0, 0]),
    #"res subsidy" :([10,20,25],[90, 60, 0]),
    "res subsidy" : ([0,10,15], [0,0,0]),
    #"demand ratio" : 1.257580634655325, # Ratio peak to offpeak demand  # rééestimer
    "demand ratio" : 1.05,
    "Nfuels" : 2, # Number of different fuels
    "Fsupply" : ([60, 60], [1.0, 0.5]), # Fuel supply functions; they are linear in this version of the model;
    # the first array contains the intercepts and the second one the coefficients
      "demand" : [61.23322924, 55.99534092, 55.48594115, 60.34034567, 61.0531315 ,
       55.83064873, 55.3227472 , 60.16287406, 60.38244757, 55.21733508,
       54.71501297, 59.50196982, 60.69293604, 55.50126437, 54.99635931,
       59.80793085, 60.51283831, 55.3365722 , 54.83316537, 59.63045924,
       60.33274058, 55.17188002, 54.66997143, 59.45298764, 60.15264284,
       55.00718784, 54.50677748, 59.27551603, 59.97254511, 54.84249566,
       54.34358354, 59.09804443, 59.79244737, 54.67780348, 54.18038959,
       58.92057282, 59.61234964, 54.5131113 , 54.01719565, 58.74310122,
       59.4322519 , 54.34841912, 53.8540017 , 58.56562961, 59.36021281,
       54.28254225, 53.78872412, 58.49464097, 59.28817373, 54.21666538,
       53.72344655, 58.42365234, 59.21613463, 54.15078851, 53.65816898,
       58.35266369, 59.14409554, 54.08491164, 53.5928914 , 58.28167505,
       59.07205645, 54.01903477, 53.52761382, 58.21068641, 59.00001735,
       53.9531579 , 53.46233624, 58.13969777, 58.92797826, 53.88728102,
       53.39705866, 58.06870912, 58.85593917, 53.82140415, 53.33178108,
       57.99772048, 58.78390007, 53.75552728, 53.26650351, 57.92673184,
       58.71186098, 53.68965041, 53.20122593, 57.8557432 , 58.79590659,
       53.76650676, 53.2773831 , 57.93856328, 58.87995219, 53.84336311,
       53.35354027, 58.02138336, 58.9639978 , 53.92021945, 53.42969745,
       58.10420344, 59.04804341, 53.9970758 , 53.50585462, 58.18702352,
       59.04804341],
    # "demand" : [61.33322924, 56.19534092, 55.78594115, 60.74034567, 61.5531315, 56.43064873
    #             , 56.0227472, 60.96287406, 61.28244757, 56.21733508, 55.81501297, 60.70196982
    #             , 61.99293604, 56.90126437, 56.49635931, 61.40793085, 62.21283831, 57.1365722
    #             , 56.73316537, 61.63045924, 62.43274058, 57.37188002, 56.96997143, 61.85298764
    #             , 62.65264284, 57.60718784, 57.20677748, 62.07551603, 62.87254511, 57.84249566
    #             , 57.44358354, 62.29804443, 63.09244737, 58.07780348, 57.68038959, 62.52057282
    #             , 63.31234964, 58.3131113, 57.91719565, 62.74310122, 63.5322519, 58.54841912
    #             , 58.1540017, 62.96562961, 63.86021281, 58.88254225, 58.48872412, 63.29464097
    #             , 64.18817373, 59.21666538, 58.82344655, 63.62365234, 64.51613463, 59.55078851
    #             , 59.15816898, 63.95266369, 64.84409554, 59.88491164, 59.4928914, 64.28167505
    #             , 65.17205645, 60.21903477, 59.82761382, 64.61068641, 65.50001735, 60.5531579
    #             , 60.16233624, 64.93969777, 65.82797826, 60.88728102, 60.49705866, 65.26870912
    #             , 66.15593917, 61.22140415, 60.83178108, 65.59772048, 66.48390007, 61.55552728
    #             , 61.16650351, 65.92673184, 66.81186098, 61.88965041, 61.50122593, 66.2557432
    #             , 67.29590659, 62.36650676, 61.9773831, 66.73856328, 67.77995219, 62.84336311
    #             , 62.45354027, 67.22138336, 68.2639978, 63.32021945, 62.92969745, 67.70420344
    #             , 68.74804341, 63.7970758, 63.40585462, 68.18702352, 69.14804341]
    # "demand" : [96.60242196, 101.27819   , 103.59543383, 107.442322  ,
    #    110.39681958, 112.91689704, 115.19913393, 118.68136811,
    #    126.22128513, 134.72608543, 136.43860934,  98.5344704 ,
    #    103.3037538 , 105.6673425 , 109.59116844, 112.60475598,
    #    115.17523498, 117.50311661, 121.05499547, 128.74571083,
    #    137.42060714, 139.16738153, 100.46651884, 105.3293176 ,
    #    107.73925118, 111.74001488, 114.81269237, 117.43357292,
    #    119.80709929, 123.42862284, 131.27013653, 140.11512885,
    #    141.89615371, 102.39856728, 107.3548814 , 109.81115986,
       # 113.88886132, 117.02062876, 119.69191086, 122.11108196,
       # 125.8022502 , 133.79456224, 142.80965056, 144.6249259 ,
       # 104.33061572, 109.3804452 , 111.88306853, 116.03770776,
       # 119.22856515, 121.9502488 , 124.41506464, 128.17587756,
       # 136.31898794, 145.50417227, 147.35369809, 106.26266416,
       # 111.406009  , 113.95497721, 118.1865542 , 121.43650154,
       # 124.20858674, 126.71904732, 130.54950492, 138.84341364,
       # 148.19869398, 150.08247027, 108.1947126 , 113.4315728 ,
       # 116.02688589, 120.33540064, 123.64443793, 126.46692468,
       # 129.02303   , 132.92313229, 141.36783934, 150.89321569,
       # 152.81124246, 110.12676103, 115.4571366 , 118.09879456,
       # 122.48424708, 125.85237433, 128.72526263, 131.32701268,
       # 135.29675965, 143.89226505, 153.58773739, 155.54001465,
       # 112.05880947, 117.4827004 , 120.17070324, 124.63309352,
       # 128.06031072, 130.98360057, 133.63099536, 137.67038701,
       # 146.41669075, 156.2822591 , 158.26878683, 113.99085791,
       # 119.5082642 , 122.24261192, 126.78193996, 130.26824711,
       # 133.24193851, 135.93497804, 140.04401437, 148.94111645,
       # 158.97678081, 160.99755902, 115.92290635, 121.533828  ,
       # 124.31452059, 128.9307864 , 132.4761835 , 135.50027645,
       # 138.23896071, 142.41764173, 151.46554215, 161.67130252,
       # 163.72633121, 117.85495479, 123.5593918 , 126.38642927,
       # 131.07963284, 134.68411989, 137.75861439, 140.54294339,
       # 144.7912691 , 153.98996786, 164.36582423, 166.45510339,
       # 119.78700323, 125.5849556 , 128.45833795, 133.22847928,
       # 136.89205628, 140.01695233, 142.84692607, 147.16489646,
       # 156.51439356, 167.06034594, 169.18387558, 121.71905167,
       # 127.6105194 , 130.53024662, 135.37732572, 139.09999268,
       # 142.27529027, 145.15090875, 149.53852382, 159.03881926,
       # 169.75486765, 171.91264777, 123.65110011, 129.6360832 ,
       # 132.6021553 , 137.52617216, 141.30792907, 144.53362821,
       # 147.45489143, 151.91215118, 161.56324496, 172.44938936,
       # 174.64141996]
}
