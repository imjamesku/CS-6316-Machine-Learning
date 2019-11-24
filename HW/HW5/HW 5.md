# HW 5

## 1.3
```
thetaPos = [8.17407853e-05 1.08987714e-04 1.15485058e-03 6.28775272e-05
 3.35346811e-05 2.09591757e-04 1.92824417e-04 2.66181532e-04
 1.78152994e-04 1.63481571e-04 1.46714230e-04 1.55097900e-04
 1.34138725e-04 1.48810148e-04 1.53001983e-04 1.99112169e-04
 1.55097900e-04 1.99112169e-04 1.46714230e-04 2.20071345e-04
 1.61385653e-04 1.61385653e-04 1.71865241e-04 2.03304004e-04
 1.78152994e-04 2.22167263e-04 1.04795879e-04 3.39538647e-04
 1.73961158e-04 2.49414191e-04 1.73961158e-04 1.55097900e-04
 1.65577488e-04 1.57193818e-04 2.03304004e-04 1.02699961e-04
 1.88632581e-04 3.96128421e-04 6.91652799e-05 1.97016252e-04
 2.03304004e-04 1.82344829e-04 1.90728499e-04 2.85044790e-04
 2.53606026e-04 2.66181532e-04 2.41030521e-04 2.59893779e-04
 2.28455015e-04 3.10195801e-04 2.87140707e-04 1.67673406e-04
 3.47922317e-04 7.12611974e-05 2.53606026e-04 2.30550933e-04
 6.39254859e-04 2.74565202e-04 2.70373367e-04 3.06003965e-04
 3.10195801e-04 2.55701944e-04 6.64405870e-04 3.58401905e-04
 3.37442729e-04 3.79361080e-04 3.31154976e-04 2.97620295e-04
 3.45826399e-04 3.29059059e-04 3.50018234e-04 3.87744751e-04
 3.64689657e-04 3.68881493e-04 4.27567185e-04 5.86856920e-04
 3.39538647e-04 4.52718195e-04 4.02416174e-04 3.64689657e-04
 5.51226321e-04 4.52718195e-04 5.57514074e-04 4.02416174e-04
 5.88952838e-04 4.79965124e-04 5.57514074e-04 6.49734447e-04
 6.37158942e-04 6.83269128e-04 6.41350777e-04 1.22611178e-03
 1.13179549e-03 1.49438923e-03 1.14437099e-03 1.07310980e-03
 1.14856283e-03 5.57514074e-04 1.43570354e-03 1.73332383e-03
 9.62045029e-01]
```
```
thetaNeg = [4.03038581e-04 3.06403600e-05 9.00355192e-04 4.71390153e-05
 1.95626914e-04 2.21553372e-04 8.01363260e-05 1.93269963e-04
 1.10776686e-04 9.89919322e-05 1.50844849e-04 1.31989243e-04
 1.29632292e-04 1.48487898e-04 1.41417046e-04 1.06062784e-04
 1.08419735e-04 9.89919322e-05 1.48487898e-04 9.66349814e-05
 1.36703144e-04 1.64986554e-04 1.43773997e-04 8.95641291e-05
 1.31989243e-04 2.54550683e-04 3.13474452e-04 1.62629603e-04
 1.57915701e-04 9.42780306e-05 1.29632292e-04 2.28624224e-04
 1.79128258e-04 1.64986554e-04 1.48487898e-04 2.47479830e-04
 1.36703144e-04 3.37043959e-04 3.44114812e-04 2.59264584e-04
 1.62629603e-04 1.97983864e-04 1.86199110e-04 1.81485209e-04
 1.62629603e-04 1.86199110e-04 1.31989243e-04 2.02697766e-04
 1.86199110e-04 1.62629603e-04 3.01689698e-04 2.78120190e-04
 1.60272652e-04 4.24251138e-04 2.33338126e-04 2.23910323e-04
 7.73079851e-04 2.63978486e-04 3.18188353e-04 2.21553372e-04
 2.92261895e-04 3.51185664e-04 5.63311233e-04 2.42765929e-04
 2.61621535e-04 1.90913012e-04 2.54550683e-04 3.37043959e-04
 2.66335437e-04 3.01689698e-04 3.62970418e-04 3.98324679e-04
 2.66335437e-04 3.79469073e-04 2.49836781e-04 6.08093298e-04
 4.45463695e-04 5.53883430e-04 3.70041270e-04 3.22902255e-04
 3.11117501e-04 4.38392842e-04 8.69714833e-04 6.90586574e-04
 6.92943525e-04 5.46812578e-04 6.03379396e-04 4.78461005e-04
 6.74087919e-04 6.85872673e-04 8.72071783e-04 8.27289719e-04
 6.05736347e-04 1.64986554e-03 7.63652048e-04 8.95641291e-04
 1.23032830e-03 1.64043773e-03 1.36938839e-03 1.89263146e-03
 9.62915737e-01]
```
## 1.4
```
MNBC classification accuracy = 0.7633333333333333
Sklearn MultinomialNB accuracy = 0.7633333333333333
```
## 1.5
```
thetaPosTrue = [0.05128205 0.06552707 0.40883191 0.04131054 0.02279202 0.11680912
 0.10683761 0.14814815 0.0982906  0.0982906  0.08404558 0.0968661
 0.08119658 0.08974359 0.08547009 0.11253561 0.08974359 0.1011396
 0.07834758 0.12108262 0.0954416  0.0982906  0.0968661  0.12678063
 0.0968661  0.11823362 0.06125356 0.14245014 0.0997151  0.12393162
 0.1011396  0.06410256 0.0968661  0.1011396  0.11538462 0.05840456
 0.11396011 0.1965812  0.03846154 0.09259259 0.10541311 0.07834758
 0.11253561 0.14672365 0.13960114 0.14387464 0.13390313 0.14814815
 0.11396011 0.15099715 0.13247863 0.1025641  0.17236467 0.04558405
 0.13960114 0.12535613 0.31908832 0.14529915 0.13675214 0.17236467
 0.16096866 0.14102564 0.32193732 0.15669516 0.17663818 0.1951567
 0.17094017 0.16809117 0.18376068 0.18376068 0.18376068 0.21367521
 0.19230769 0.20512821 0.21509972 0.31054131 0.18233618 0.23646724
 0.18091168 0.15099715 0.21082621 0.20512821 0.3048433  0.21937322
 0.24216524 0.21652422 0.26638177 0.28205128 0.30626781 0.31054131
 0.2977208  0.49287749 0.41595442 0.56267806 0.45726496 0.41452991
 0.45584046 0.26210826 0.54273504 0.56695157 0.9985755 ]
thetaNegTrue = [0.18518519 0.01851852 0.34045584 0.02706553 0.10541311 0.11396011
 0.04558405 0.0997151  0.05982906 0.04700855 0.08262108 0.07264957
 0.06837607 0.07977208 0.07834758 0.06267806 0.05698006 0.04985755
 0.07692308 0.05698006 0.07549858 0.08547009 0.08119658 0.05270655
 0.07549858 0.13105413 0.15527066 0.08547009 0.08262108 0.05270655
 0.06837607 0.05555556 0.09259259 0.08974359 0.07407407 0.12108262
 0.07834758 0.16239316 0.15954416 0.08262108 0.07264957 0.08119658
 0.10541311 0.0954416  0.08547009 0.1025641  0.07264957 0.10826211
 0.08831909 0.08974359 0.11823362 0.14529915 0.07122507 0.19230769
 0.11396011 0.12250712 0.33903134 0.12393162 0.15099715 0.12393162
 0.15099715 0.16809117 0.25783476 0.11253561 0.12393162 0.0982906
 0.11396011 0.17663818 0.13817664 0.14529915 0.18518519 0.19373219
 0.14102564 0.17236467 0.12393162 0.28062678 0.20512821 0.24786325
 0.15669516 0.13532764 0.12678063 0.16524217 0.36324786 0.32193732
 0.25356125 0.20512821 0.26068376 0.20655271 0.29202279 0.29202279
 0.32051282 0.35042735 0.26638177 0.54273504 0.31766382 0.34472934
 0.45868946 0.49145299 0.49430199 0.57692308 0.9985755 ]
--------------------
BNBC classification accuracy = 0.6183333333333333
--------------------
```

## 2 Sample QA Questions:
![](https://i.imgur.com/YuaSy1X.jpg)