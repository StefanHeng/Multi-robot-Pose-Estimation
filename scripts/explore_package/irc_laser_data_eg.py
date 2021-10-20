import numpy as np


src_pts = np.array([[8.75010138e-16, -1.42900000e+01, 1.00000000e+00],
                    [1.10354741e+00, -1.57814633e+01, 1.00000000e+00],
                    [5.76441994e-01, -4.69474330e+00, 1.00000000e+00],
                    [5.77568369e-01, -4.10961249e+00, 1.00000000e+00],
                    [5.88193589e-01, -3.71370816e+00, 1.00000000e+00],
                    [6.00822695e-01, -3.40743483e+00, 1.00000000e+00],
                    [6.12496875e-01, -3.15102326e+00, 1.00000000e+00],
                    [6.21655956e-01, -2.92466133e+00, 1.00000000e+00],
                    [6.27613442e-01, -2.71849248e+00, 1.00000000e+00],
                    [6.36254585e-01, -2.55187776e+00, 1.00000000e+00],
                    [6.44459422e-01, -2.40515531e+00, 1.00000000e+00],
                    [6.58773280e-01, -2.29741545e+00, 1.00000000e+00],
                    [6.66607487e-01, -2.18037484e+00, 1.00000000e+00],
                    [6.79837388e-01, -2.09232434e+00, 1.00000000e+00],
                    [6.90204487e-01, -2.00449938e+00, 1.00000000e+00],
                    [6.94300891e-01, -1.90757602e+00, 1.00000000e+00],
                    [7.05984861e-01, -1.83915344e+00, 1.00000000e+00],
                    [7.11752527e-01, -1.76164932e+00, 1.00000000e+00],
                    [7.26759899e-01, -1.71213903e+00, 1.00000000e+00],
                    [7.32125958e-01, -1.64438182e+00, 1.00000000e+00],
                    [7.39581958e-01, -1.58603863e+00, 1.00000000e+00],
                    [7.45230950e-01, -1.52794988e+00, 1.00000000e+00],
                    [7.49084325e-01, -1.47016076e+00, 1.00000000e+00],
                    [7.60543932e-01, -1.43037510e+00, 1.00000000e+00],
                    [7.65999200e-01, -1.38189914e+00, 1.00000000e+00],
                    [7.70000000e-01, -1.33367912e+00, 1.00000000e+00],
                    [7.77707493e-01, -1.29432262e+00, 1.00000000e+00],
                    [7.84280511e-01, -1.25511118e+00, 1.00000000e+00],
                    [7.84280210e-01, -1.20768562e+00, 1.00000000e+00],
                    [7.82870065e-01, -1.16065260e+00, 1.00000000e+00],
                    [7.97271247e-01, -1.13862134e+00, 1.00000000e+00],
                    [7.99387943e-01, -1.10026311e+00, 1.00000000e+00],
                    [8.06432131e-01, -1.07017158e+00, 1.00000000e+00],
                    [8.06516533e-01, -1.03229409e+00, 1.00000000e+00],
                    [8.05530101e-01, -9.94746831e-01, 1.00000000e+00],
                    [8.09912388e-01, -9.65215998e-01, 1.00000000e+00],
                    [8.20073786e-01, -9.43386975e-01, 1.00000000e+00],
                    [8.23030646e-01, -9.14068135e-01, 1.00000000e+00],
                    [8.25218016e-01, -8.84937979e-01, 1.00000000e+00],
                    [8.26643461e-01, -8.56014362e-01, 1.00000000e+00],
                    [8.27314934e-01, -8.27314934e-01, 1.00000000e+00],
                    [8.27240770e-01, -7.98857126e-01, 1.00000000e+00],
                    [8.33743220e-01, -7.77478130e-01, 1.00000000e+00],
                    [8.39753653e-01, -7.56117585e-01, 1.00000000e+00],
                    [8.37727634e-01, -7.28225522e-01, 1.00000000e+00],
                    [8.42648887e-01, -7.07066371e-01, 1.00000000e+00],
                    [8.39317638e-01, -6.79666022e-01, 1.00000000e+00],
                    [8.43171506e-01, -6.58757779e-01, 1.00000000e+00],
                    [8.46553641e-01, -6.37923925e-01, 1.00000000e+00],
                    [8.49467844e-01, -6.17174515e-01, 1.00000000e+00],
                    [8.51918126e-01, -5.96519494e-01, 1.00000000e+00],
                    [8.62199075e-01, -5.81560620e-01, 1.00000000e+00],
                    [8.72217391e-01, -5.66424596e-01, 1.00000000e+00],
                    [8.73489539e-01, -5.45816842e-01, 1.00000000e+00],
                    [8.65738974e-01, -5.20188456e-01, 1.00000000e+00],
                    [8.66025404e-01, -5.00000000e-01, 1.00000000e+00],
                    [8.74619707e-01, -4.84809620e-01, 1.00000000e+00],
                    [8.74118117e-01, -4.64776847e-01, 1.00000000e+00],
                    [9.08826655e-01, -4.63070310e-01, 1.00000000e+00],
                    [9.43733749e-01, -4.60289704e-01, 1.00000000e+00],
                    [1.00600164e+00, -4.69106271e-01, 1.00000000e+00],
                    [1.08711909e+00, -4.84016605e-01, 1.00000000e+00],
                    [1.33473204e+00, -5.66560136e-01, 1.00000000e+00],
                    [2.63320215e+00, -1.06388273e+00, 1.00000000e+00],
                    [2.74472645e+00, -1.05360177e+00, 1.00000000e+00],
                    [2.86606249e+00, -1.04316144e+00, 1.00000000e+00],
                    [2.98783870e+00, -1.02879537e+00, 1.00000000e+00],
                    [3.11946537e+00, -1.01357574e+00, 1.00000000e+00],
                    [3.27056227e+00, -9.99911230e-01, 1.00000000e+00],
                    [3.34519070e+00, -9.59217998e-01, 1.00000000e+00],
                    [3.36142188e+00, -9.00690277e-01, 1.00000000e+00],
                    [3.41544096e+00, -8.51565073e-01, 1.00000000e+00],
                    [3.67337514e+00, -8.48065475e-01, 1.00000000e+00],
                    [3.89302745e+00, -8.27488529e-01, 1.00000000e+00],
                    [4.13265044e+00, -8.03305871e-01, 1.00000000e+00],
                    [4.43163489e+00, -7.81416800e-01, 1.00000000e+00],
                    [4.75078092e+00, -7.52449777e-01, 1.00000000e+00],
                    [5.09988055e+00, -7.16741470e-01, 1.00000000e+00],
                    [5.53840753e+00, -6.80030936e-01, 1.00000000e+00],
                    [6.01685747e+00, -6.32397203e-01, 1.00000000e+00],
                    [6.62469474e+00, -5.79585689e-01, 1.00000000e+00],
                    [7.02285091e+00, -4.91085575e-01, 1.00000000e+00],
                    [7.01037933e+00, -3.67398413e-01, 1.00000000e+00],
                    [7.00572970e+00, -2.44645472e-01, 1.00000000e+00],
                    [8.69867495e+00, -1.51835936e-01, 1.00000000e+00],
                    [9.85000000e+00, 1.78388554e-14, 1.00000000e+00],
                    [1.20081708e+01, 2.09603401e-01, 1.00000000e+00],
                    [1.84587476e+01, 6.44593669e-01, 1.00000000e+00],
                    [1.84846327e+01, 9.68738550e-01, 1.00000000e+00],
                    [1.96420371e+01, 1.37350504e+00, 1.00000000e+00],
                    [2.20457877e+01, 1.92875650e+00, 1.00000000e+00],
                    [1.40525944e+01, 1.47698719e+00, 1.00000000e+00],
                    [7.97014560e+00, 9.78610828e-01, 1.00000000e+00],
                    [7.93204723e+00, 1.11477654e+00, 1.00000000e+00],
                    [7.91138361e+00, 1.25304006e+00, 1.00000000e+00],
                    [7.89815818e+00, 1.39265838e+00, 1.00000000e+00],
                    [7.88246628e+00, 1.53219623e+00, 1.00000000e+00],
                    [7.87408819e+00, 1.67368911e+00, 1.00000000e+00],
                    [7.86316642e+00, 1.81535501e+00, 1.00000000e+00],
                    [6.39424884e+00, 1.59426529e+00, 1.00000000e+00],
                    [5.71828089e+00, 1.53220875e+00, 1.00000000e+00],
                    [5.17158792e+00, 1.48292897e+00, 1.00000000e+00],
                    [4.70501940e+00, 1.43846879e+00, 1.00000000e+00],
                    [4.35583884e+00, 1.41529783e+00, 1.00000000e+00],
                    [4.01845395e+00, 1.38366466e+00, 1.00000000e+00],
                    [3.72118278e+00, 1.35439977e+00, 1.00000000e+00],
                    [3.46358338e+00, 1.32954509e+00, 1.00000000e+00],
                    [3.25441533e+00, 1.31486914e+00, 1.00000000e+00],
                    [3.06528116e+00, 1.30113466e+00, 1.00000000e+00],
                    [2.87766819e+00, 1.28122043e+00, 1.00000000e+00],
                    [2.71892336e+00, 1.26785479e+00, 1.00000000e+00],
                    [2.56156303e+00, 1.24935777e+00, 1.00000000e+00],
                    [2.43244781e+00, 1.23939406e+00, 1.00000000e+00],
                    [2.31332269e+00, 1.23001549e+00, 1.00000000e+00],
                    [2.20404166e+00, 1.22172024e+00, 1.00000000e+00],
                    [2.12176224e+00, 1.22500000e+00, 1.00000000e+00],
                    [2.15148992e+00, 1.29274557e+00, 1.00000000e+00],
                    [2.13708120e+00, 1.33539655e+00, 1.00000000e+00],
                    [2.13860995e+00, 1.38882954e+00, 1.00000000e+00],
                    [2.13062656e+00, 1.43712576e+00, 1.00000000e+00],
                    [2.15436988e+00, 1.50850603e+00, 1.00000000e+00],
                    [2.16816554e+00, 1.57526448e+00, 1.00000000e+00],
                    [2.16430223e+00, 1.63091871e+00, 1.00000000e+00],
                    [2.16702957e+00, 1.69306906e+00, 1.00000000e+00],
                    [2.15269431e+00, 1.74321748e+00, 1.00000000e+00],
                    [2.14492444e+00, 1.79980531e+00, 1.00000000e+00],
                    [2.13582811e+00, 1.85664705e+00, 1.00000000e+00],
                    [2.14025710e+00, 1.92709615e+00, 1.00000000e+00],
                    [2.12823927e+00, 1.98461523e+00, 1.00000000e+00],
                    [3.61108580e+00, 3.48718502e+00, 1.00000000e+00],
                    [3.47189430e+00, 3.47189430e+00, 1.00000000e+00],
                    [3.35519993e+00, 3.47441124e+00, 1.00000000e+00],
                    [3.30769205e+00, 3.54706545e+00, 1.00000000e+00],
                    [3.31219650e+00, 3.67856689e+00, 1.00000000e+00],
                    [3.28685574e+00, 3.78109500e+00, 1.00000000e+00],
                    [3.27178893e+00, 3.89916622e+00, 1.00000000e+00],
                    [3.16548157e+00, 3.90904419e+00, 1.00000000e+00],
                    [3.05368092e+00, 3.90853334e+00, 1.00000000e+00],
                    [2.93083916e+00, 3.88935493e+00, 1.00000000e+00],
                    [2.82136921e+00, 3.88328157e+00, 1.00000000e+00],
                    [2.71301654e+00, 3.87458917e+00, 1.00000000e+00],
                    [2.60583893e+00, 3.86331509e+00, 1.00000000e+00],
                    [2.51078595e+00, 3.86627132e+00, 1.00000000e+00],
                    [2.41113265e+00, 3.85861884e+00, 1.00000000e+00],
                    [2.31252096e+00, 3.84868118e+00, 1.00000000e+00],
                    [2.21500000e+00, 3.83649254e+00, 1.00000000e+00],
                    [2.12346614e+00, 3.83083432e+00, 1.00000000e+00],
                    [2.02811715e+00, 3.81433360e+00, 1.00000000e+00],
                    [1.94307934e+00, 3.81350792e+00, 1.00000000e+00],
                    [1.85869366e+00, 3.81088676e+00, 1.00000000e+00],
                    [1.77077052e+00, 3.79742963e+00, 1.00000000e+00],
                    [1.69202444e+00, 3.80034910e+00, 1.00000000e+00],
                    [1.60590494e+00, 3.78327495e+00, 1.00000000e+00],
                    [1.52464884e+00, 3.77363829e+00, 1.00000000e+00],
                    [1.44422284e+00, 3.76232912e+00, 1.00000000e+00],
                    [1.37492098e+00, 3.77756434e+00, 1.00000000e+00],
                    [1.29250557e+00, 3.75370875e+00, 1.00000000e+00],
                    [3.49189204e-01, 1.07469386e+00, 1.00000000e+00],
                    [3.18685158e-01, 1.04237218e+00, 1.00000000e+00],
                    [2.97688344e-01, 1.03816263e+00, 1.00000000e+00],
                    [2.79524569e-01, 1.04319989e+00, 1.00000000e+00],
                    [2.58856428e-01, 1.03821643e+00, 1.00000000e+00],
                    [2.40697628e-01, 1.04257597e+00, 1.00000000e+00],
                    [2.20386392e-01, 1.03683646e+00, 1.00000000e+00],
                    [2.04165625e-01, 1.05034109e+00, 1.00000000e+00],
                    [1.84067068e-01, 1.04389622e+00, 1.00000000e+00],
                    [1.64256188e-01, 1.03707276e+00, 1.00000000e+00],
                    [1.46131756e-01, 1.03978147e+00, 1.00000000e+00],
                    [1.27962811e-01, 1.04217346e+00, 1.00000000e+00],
                    [1.08709602e-01, 1.03430277e+00, 1.00000000e+00],
                    [9.06419725e-02, 1.03604249e+00, 1.00000000e+00],
                    [7.25467327e-02, 1.03746661e+00, 1.00000000e+00],
                    [5.33826754e-02, 1.01860213e+00, 1.00000000e+00],
                    [3.59464816e-02, 1.02937255e+00, 1.00000000e+00],
                    [1.78014546e-02, 1.01984465e+00, 1.00000000e+00]])
tgt_pts = np.array([[3.75354244e-16, -6.13000000e+00, 1.00000000e+00],
                    [1.12393497e-01, -6.43901916e+00, 1.00000000e+00],
                    [2.39410547e-01, -6.85582107e+00, 1.00000000e+00],
                    [3.86762717e-01, -7.37987226e+00, 1.00000000e+00],
                    [5.53866402e-01, -7.92065856e+00, 1.00000000e+00],
                    [7.56511847e-01, -8.64696998e+00, 1.00000000e+00],
                    [9.91975116e-01, -9.43801279e+00, 1.00000000e+00],
                    [1.27719072e+00, -1.04018837e+01, 1.00000000e+00],
                    [1.97208284e+00, -1.40320985e+01, 1.00000000e+00],
                    [2.21198334e+00, -1.39659131e+01, 1.00000000e+00],
                    [2.47969598e+00, -1.40630547e+01, 1.00000000e+00],
                    [2.72284436e+00, -1.40078199e+01, 1.00000000e+00],
                    [4.08675272e+00, -1.52519688e+01, 1.00000000e+00],
                    [1.44001919e+00, -4.43192337e+00, 1.00000000e+00],
                    [1.32831807e+00, -3.85771579e+00, 1.00000000e+00],
                    [1.27231493e+00, -3.49565655e+00, 1.00000000e+00],
                    [1.22203471e+00, -3.18350925e+00, 1.00000000e+00],
                    [1.19124897e+00, -2.94844466e+00, 1.00000000e+00],
                    [1.15656414e+00, -2.72469437e+00, 1.00000000e+00],
                    [1.12666050e+00, -2.53052092e+00, 1.00000000e+00],
                    [1.09880748e+00, -2.35640025e+00, 1.00000000e+00],
                    [1.08716044e+00, -2.22900923e+00, 1.00000000e+00],
                    [1.07141758e+00, -2.10277540e+00, 1.00000000e+00],
                    [1.06100573e+00, -1.99546156e+00, 1.00000000e+00],
                    [1.05688497e+00, -1.90667096e+00, 1.00000000e+00],
                    [1.05000000e+00, -1.81865335e+00, 1.00000000e+00],
                    [1.04552729e+00, -1.74004962e+00, 1.00000000e+00],
                    [1.03864176e+00, -1.66217427e+00, 1.00000000e+00],
                    [1.03481417e+00, -1.59347408e+00, 1.00000000e+00],
                    [1.03450687e+00, -1.53371951e+00, 1.00000000e+00],
                    [1.02096606e+00, -1.45809064e+00, 1.00000000e+00],
                    [1.02274634e+00, -1.40768957e+00, 1.00000000e+00],
                    [1.01706739e+00, -1.34969401e+00, 1.00000000e+00],
                    [1.01584143e+00, -1.30021774e+00, 1.00000000e+00],
                    [1.00691263e+00, -1.24343354e+00, 1.00000000e+00],
                    [1.00917655e+00, -1.20268978e+00, 1.00000000e+00],
                    [1.00377031e+00, -1.15470566e+00, 1.00000000e+00],
                    [1.00369591e+00, -1.11471724e+00, 1.00000000e+00],
                    [9.95717606e-01, -1.06777640e+00, 1.00000000e+00],
                    [1.00030805e+00, -1.03584931e+00, 1.00000000e+00],
                    [9.82878426e-01, -9.82878426e-01, 1.00000000e+00],
                    [9.85495526e-01, -9.51681968e-01, 1.00000000e+00],
                    [9.87327497e-01, -9.20697786e-01, 1.00000000e+00],
                    [9.88382618e-01, -8.89943706e-01, 1.00000000e+00],
                    [9.73575358e-01, -8.46316147e-01, 1.00000000e+00],
                    [9.72876443e-01, -8.16340264e-01, 1.00000000e+00],
                    [9.79203911e-01, -7.92943693e-01, 1.00000000e+00],
                    [9.77133334e-01, -7.63420229e-01, 1.00000000e+00],
                    [9.74335322e-01, -7.34214328e-01, 1.00000000e+00],
                    [9.62730223e-01, -6.99464450e-01, 1.00000000e+00],
                    [9.58407892e-01, -6.71084431e-01, 1.00000000e+00],
                    [9.61683584e-01, -6.48663768e-01, 1.00000000e+00],
                    [9.56084447e-01, -6.20888500e-01, 1.00000000e+00],
                    [9.58294349e-01, -5.98808769e-01, 1.00000000e+00],
                    [9.60027377e-01, -5.76842644e-01, 1.00000000e+00],
                    [9.52627944e-01, -5.50000000e-01, 1.00000000e+00],
                    [9.44589284e-01, -5.23594390e-01, 1.00000000e+00],
                    [9.44753924e-01, -5.02334572e-01, 1.00000000e+00],
                    [9.53376981e-01, -4.85769835e-01, 1.00000000e+00],
                    [9.43733749e-01, -4.60289704e-01, 1.00000000e+00],
                    [9.51623176e-01, -4.43749175e-01, 1.00000000e+00],
                    [9.59222731e-01, -4.27073475e-01, 1.00000000e+00],
                    [9.48119999e-01, -4.02453062e-01, 1.00000000e+00],
                    [9.45727532e-01, -3.82098725e-01, 1.00000000e+00],
                    [9.42916231e-01, -3.61951629e-01, 1.00000000e+00],
                    [9.39692621e-01, -3.42020143e-01, 1.00000000e+00],
                    [9.45518576e-01, -3.25568154e-01, 1.00000000e+00],
                    [9.41545951e-01, -3.05926824e-01, 1.00000000e+00],
                    [9.37178661e-01, -2.86524271e-01, 1.00000000e+00],
                    [9.32423845e-01, -2.67368235e-01, 1.00000000e+00],
                    [9.75585085e-01, -2.61407236e-01, 1.00000000e+00],
                    [1.04791938e+00, -2.61275647e-01, 1.00000000e+00],
                    [1.11078187e+00, -2.56444202e-01, 1.00000000e+00],
                    [1.19334007e+00, -2.53652263e-01, 1.00000000e+00],
                    [2.77800493e+00, -5.39989457e-01, 1.00000000e+00],
                    [2.88548672e+00, -5.08789161e-01, 1.00000000e+00],
                    [3.00257256e+00, -4.75560774e-01, 1.00000000e+00],
                    [3.11934442e+00, -4.38395268e-01, 1.00000000e+00],
                    [3.25555138e+00, -3.99731446e-01, 1.00000000e+00],
                    [3.39131966e+00, -3.56442060e-01, 1.00000000e+00],
                    [3.45679560e+00, -3.02430427e-01, 1.00000000e+00],
                    [3.46154725e+00, -2.42054964e-01, 1.00000000e+00],
                    [3.50518967e+00, -1.83699206e-01, 1.00000000e+00],
                    [3.75770951e+00, -1.31222108e-01, 1.00000000e+00],
                    [3.96939535e+00, -6.92860536e-02, 1.00000000e+00],
                    [4.22000000e+00, 7.64263652e-15, 1.00000000e+00],
                    [4.49931463e+00, 7.85358290e-02, 1.00000000e+00],
                    [4.80706988e+00, 1.67866579e-01, 1.00000000e+00],
                    [5.17290099e+00, 2.71100253e-01, 1.00000000e+00],
                    [5.57638304e+00, 3.89938688e-01, 1.00000000e+00],
                    [6.05686376e+00, 5.29906916e-01, 1.00000000e+00],
                    [6.63346104e+00, 6.97204850e-01, 1.00000000e+00],
                    [6.98752491e+00, 8.57960178e-01, 1.00000000e+00],
                    [6.95168184e+00, 9.76995169e-01, 1.00000000e+00],
                    [6.92369527e+00, 1.09660560e+00, 1.00000000e+00],
                    [8.56782745e+00, 1.51073915e+00, 1.00000000e+00],
                    [1.10433058e+01, 2.14660120e+00, 1.00000000e+00],
                    [1.17671156e+01, 2.50117764e+00, 1.00000000e+00],
                    [1.79966141e+01, 4.15484575e+00, 1.00000000e+00],
                    [1.91633406e+01, 4.77795744e+00, 1.00000000e+00],
                    [1.90190805e+01, 5.09614726e+00, 1.00000000e+00],
                    [2.12823330e+01, 6.10261078e+00, 1.00000000e+00],
                    [1.37803515e+01, 4.21307627e+00, 1.00000000e+00],
                    [7.77964230e+00, 2.52775901e+00, 1.00000000e+00],
                    [7.56414860e+00, 2.60454524e+00, 1.00000000e+00],
                    [7.51754097e+00, 2.73616115e+00, 1.00000000e+00],
                    [7.48731502e+00, 2.87411096e+00, 1.00000000e+00],
                    [7.44528635e+00, 3.00809095e+00, 1.00000000e+00],
                    [7.41006407e+00, 3.14538558e+00, 1.00000000e+00],
                    [7.37231184e+00, 3.28236471e+00, 1.00000000e+00],
                    [7.33203000e+00, 3.41898174e+00, 1.00000000e+00],
                    [5.44669192e+00, 2.65652915e+00, 1.00000000e+00],
                    [4.91835601e+00, 2.50602756e+00, 1.00000000e+00],
                    [4.46771482e+00, 2.37552611e+00, 1.00000000e+00],
                    [4.09322023e+00, 2.26890902e+00, 1.00000000e+00],
                    [3.77587076e+00, 2.18000000e+00, 1.00000000e+00],
                    [3.48009924e+00, 2.09105458e+00, 1.00000000e+00],
                    [3.22258277e+00, 2.01369320e+00, 1.00000000e+00],
                    [3.01082734e+00, 1.95525414e+00, 1.00000000e+00],
                    [2.81872775e+00, 1.90125587e+00, 1.00000000e+00],
                    [2.62947806e+00, 1.84118036e+00, 1.00000000e+00],
                    [2.46750183e+00, 1.79274502e+00, 1.00000000e+00],
                    [2.33201569e+00, 1.75729987e+00, 1.00000000e+00],
                    [2.19855000e+00, 1.71769552e+00, 1.00000000e+00],
                    [2.08275118e+00, 1.68657865e+00, 1.00000000e+00],
                    [1.96873422e+00, 1.65196416e+00, 1.00000000e+00],
                    [1.85658557e+00, 1.61390521e+00, 1.00000000e+00],
                    [1.85786206e+00, 1.67282652e+00, 1.00000000e+00],
                    [1.85032487e+00, 1.72545585e+00, 1.00000000e+00],
                    [1.83431649e+00, 1.77137884e+00, 1.00000000e+00],
                    [1.81726443e+00, 1.81726443e+00, 1.00000000e+00],
                    [1.80611176e+00, 1.87028348e+00, 1.00000000e+00],
                    [1.82775560e+00, 1.96002792e+00, 1.00000000e+00],
                    [1.81334394e+00, 2.01392248e+00, 1.00000000e+00],
                    [1.79760174e+00, 2.06790425e+00, 1.00000000e+00],
                    [1.78694955e+00, 2.12960355e+00, 1.00000000e+00],
                    [1.76839030e+00, 2.18378015e+00, 1.00000000e+00],
                    [1.74847859e+00, 2.23795054e+00, 1.00000000e+00],
                    [1.73322727e+00, 2.30007027e+00, 1.00000000e+00],
                    [1.71633294e+00, 2.36232962e+00, 1.00000000e+00],
                    [2.90803253e+00, 4.15310086e+00, 1.00000000e+00],
                    [2.77359680e+00, 4.11202636e+00, 1.00000000e+00],
                    [2.65239210e+00, 4.08432567e+00, 1.00000000e+00],
                    [2.58600601e+00, 4.13847471e+00, 1.00000000e+00],
                    [2.55458885e+00, 4.25154981e+00, 1.00000000e+00],
                    [2.51500000e+00, 4.35610778e+00, 1.00000000e+00],
                    [2.48222526e+00, 4.47805290e+00, 1.00000000e+00],
                    [2.37552611e+00, 4.46771482e+00, 1.00000000e+00],
                    [2.26541259e+00, 4.44612256e+00, 1.00000000e+00],
                    [2.15240233e+00, 4.41307877e+00, 1.00000000e+00],
                    [2.04547239e+00, 4.38652969e+00, 1.00000000e+00],
                    [1.94013379e+00, 4.35761183e+00, 1.00000000e+00],
                    [1.83643630e+00, 4.32637281e+00, 1.00000000e+00],
                    [1.73817459e+00, 4.30213309e+00, 1.00000000e+00],
                    [1.63774153e+00, 4.26646255e+00, 1.00000000e+00],
                    [1.54593105e+00, 4.24741065e+00, 1.00000000e+00],
                    [1.45528965e+00, 4.22646803e+00, 1.00000000e+00],
                    [1.36276495e+00, 4.19415924e+00, 1.00000000e+00],
                    [1.27474063e+00, 4.16948874e+00, 1.00000000e+00],
                    [1.18799700e+00, 4.14303791e+00, 1.00000000e+00],
                    [1.10256913e+00, 4.11484402e+00, 1.00000000e+00],
                    [1.02332962e+00, 4.10435092e+00, 1.00000000e+00],
                    [9.40295407e-01, 4.07286687e+00, 1.00000000e+00],
                    [8.62833517e-01, 4.05931254e+00, 1.00000000e+00],
                    [7.84224971e-01, 4.03448772e+00, 1.00000000e+00],
                    [7.06748083e-01, 4.00816755e+00, 1.00000000e+00],
                    [6.31995239e-01, 3.99026090e+00, 1.00000000e+00],
                    [5.56692404e-01, 3.96107227e+00, 1.00000000e+00],
                    [1.42587132e-01, 1.16127900e+00, 1.00000000e+00],
                    [1.18117163e-01, 1.12380974e+00, 1.00000000e+00],
                    [9.58713170e-02, 1.09581417e+00, 1.00000000e+00],
                    [7.67321211e-02, 1.09732046e+00, 1.00000000e+00],
                    [5.75695519e-02, 1.09849249e+00, 1.00000000e+00],
                    [3.83894464e-02, 1.09932991e+00, 1.00000000e+00],
                    [1.90231230e-02, 1.08983399e+00, 1.00000000e+00]])