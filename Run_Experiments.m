%Run my experiments

[statisticsLT, statisticsADMM] = basis_pursuit_experiments(1);


%should be near zero, to verify both ADMM and LT had the same solutions
max(statisticsLT.LTvsADMM) 

%view the wins of LT
LTwins = sum(statisticsLT.wins)

%view the failures of LT
LTfails = sum(statisticsLT.fails)

%view the wins of ADMM
ADMMwins = sum(statisticsADMM.wins)

%view the failures of ADMM
ADMMfails = sum(statisticsADMM.fails)

%view quantiles
pp = 0:0.25:1;

LTdata = quantile(statisticsLT.iterates,pp)

ADMMdata = quantile(statisticsADMM.iterates,pp)
