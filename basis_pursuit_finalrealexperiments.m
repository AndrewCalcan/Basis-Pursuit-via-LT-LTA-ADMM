function [statisticsLT, statisticsADMM] = basis_pursuit_finalrealexperiments(num_experiments)

%%Time specifications: 44 800 samples = 1 second

RecovLength = num_experiments; %amount of time (seconds) we recover of the song. 134 seconds is full Chopin piece.
TimeStart=0.1; %time (in seconds) where we want to start recovering

%Time each problem covers (seconds).
ProblemTime = 0.1;

[bEntire,SamPerSec] = audioread('chopin.wav'); %reading the file
bEntire(:,2)=[]; %removing the second sample column (this column is due to stereo sound).

%Percentage of samples kept
PerKept = 0.1;

%Number of samples to be kept for each problem. 
NumPoints = SamPerSec * ProblemTime * PerKept;

%Size of each problem ie Freq \times seconds.
ProblemSize = SamPerSec * ProblemTime;

%dx
dx = 1/SamPerSec; %1/44 800

%The domain of the graphs
GraphDomain = (0:dx:RecovLength-dx); %(ProblemTime*num_experiments)-dx);

%Building IDCTM
%Creating the first column of the IDCTM by multiplier a column vector of
%ones by (1/sqrt(2)): 
IDCTMCol11 = ones(ProblemSize,1);
IDCTMC1 = (1/sqrt(2))*IDCTMCol11;

%Remaining part of the IDCTM generated: 
piCoeff = (1:2:(2*ProblemSize))'.*(1:(ProblemSize-1));
IDCTMRem = cos((piCoeff*pi)/(2*ProblemSize));

%Creating the full IDCTM by concating IDCTMC1 (column 1) and IDCTMRem:
FullIDCTM = cat(2,IDCTMC1,IDCTMRem);
FullIDCTM = (sqrt(2/ProblemSize))*FullIDCTM;

k = 1;
h=1; %the number of the bLT and bADMM columns

%ProblemStart=1;

%The starting point of the first problem ie sample "ProblemStart" of
%bEntire
ProblemStart = (TimeStart*(1/ProblemTime))*ProblemSize;
bEntire = bEntire(ProblemStart:(ProblemStart+(SamPerSec*num_experiments))-1); %shortening bEntire for plotting the original signal

disp(size(GraphDomain));
disp(size(bEntire));

%Initialisation
bLT=zeros(ProblemSize,(num_experiments/ProblemTime));
bADMM=zeros(ProblemSize,(num_experiments/ProblemTime));

ProblemStart=1;

for j = 1:(num_experiments*(1/ProblemTime)) %while num_experiments > 0 %while the number of experiments is positive
    
    %Isolating the section of bEntire tackled by each problem.
    btest=bEntire(ProblemStart:(ProblemStart+ProblemSize)-1);  %%%%%%% (ProblemStart*ProblemSize)-1

    disp(size(btest));

    %The rows to be retained during audio compression. getting NumPoints many
    %numbers from 1 to ProblemmSize.
    xKept = randperm(ProblemSize,NumPoints);

    b=btest(xKept,:); %%%%%%%%
    
    A = FullIDCTM(xKept,:);

    %Calling upon function to solve problem using Lyapunov-Surrogate Method
    [LTx, LThistory, LTRachold] = fast_basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0); 

    %Calling upon function to solve problem using Lyapunov-Surrogate variation Method
    [VLTx, VLThistory, VLTRachold] = var_basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0); 

    %Calling upon function to solve problem using ADMM
	[x, history, Rachold] = basis_pursuit(A, b, 1.0, 1.0);

	K = length(history.objval);
	LTK = length(LThistory.objval);

	P3= history.Rach_diff;

	%Objective function values:
    statisticsVLT.objval(k) = VLThistory.objval(length(VLThistory.objval)); %LT variation
	statisticsLT.objval(k) = LThistory.objval(length(LThistory.objval)); %LT
	statisticsADMM.objval(k) = history.objval(length(history.objval)); %ADMM
	
    statisticsVLT.r_norm(k) = VLThistory.r_norm(length(VLThistory.r_norm)); %LT variation
	statisticsLT.r_norm(k) = LThistory.r_norm(length(LThistory.r_norm)); %LT
	statisticsADMM.r_norm(k) = history.r_norm(length(history.r_norm)); %ADMM
	
    %Number of iterates it takes each algorithm to solve:
    statisticsVLT.iterates(k) = length(VLThistory.objval); %LT variation
	statisticsLT.iterates(k) = length(LThistory.objval);   %LT
	statisticsADMM.iterates(k) = length(history.objval);    %ADMM
    
    if length(LThistory.objval) < length(history.objval) && length(LThistory.objval) < length(VLThistory.objval)
        statisticsLT.wins(k) = 1;
        statisticsADMM.wins(k) = 0;
        statisticsVLT.wins(k) = 0;
    elseif length(LThistory.objval) > length(history.objval) && length(history.objval) < length(VLThistory.objval)
        statisticsADMM.wins(k) = 1;
        statisticsLT.wins(k) = 0;
        statisticsVLT.wins(k) = 0;
    elseif length(VLThistory.objval) < length(history.objval) && length(VLThistory.objval) < length(LThistory.objval)
        statisticsVLT.wins(k) = 1;
        statisticsLT.wins(k) = 0;
        statisticsADMM.wins(k) = 0;
    else
        statisticsLT.wins(k) = 0;
        statisticsADMM.wins(k) = 0;
        statisticsVLT.wins(k) = 0;
    end

    %%Statistics for Algorithm Performance Profiles%%

    %Identifying the algorithm with shorter computation time
    fastest_time = min([length(LThistory.objval),length(history.objval),length(VLThistory.objval)]);

    %Creating the performance ratio for Lyapunov-Surrogate Method
    statisticsLT.perf_prof(k) = length(LThistory.objval)/fastest_time;

    %Creating the performance ratio for ADMM
    statisticsADMM.perf_prof(k) = length(history.objval)/fastest_time;

    %Creating the performance ratio for LT variation
    statisticsVLT.perf_prof(k) = length(VLThistory.objval)/fastest_time;
    
    if length(history.objval) == 2000
        statisticsADMM.fails(k) = 1;
        statisticsLT.LTvsADMM(k) = 0; %I record the difference as a zero if ADMM fails to solve
    else
        statisticsADMM.fails(k) = 0;
        statisticsLT.LTvsADMM(k) = norm(x-LTx); %I record the difference between LT solutions and ADMM solutions when they both solved the problem
    end
    
    if length(LThistory.objval) == 2000 %value corresponds to MAX_ITER
        statisticsLT.fails(k) = 1;
    else
        statisticsLT.fails(k) = 0;
    end

    if length(VLThistory.objval) == 2000 
        statisticsVLT.fails(k) = 1;
    else
        statisticsVLT.fails(k) = 0;
    end

    K = length(history.objval);
    LTK = length(LThistory.objval); 

    P3= history.Rach_diff;
    
    %Reconstructing the waves produced by LT
    bLT(:,h) = FullIDCTM * LTx; 

    %Reconstructing the waves produced by ADMM
    bADMM(:,h) = FullIDCTM * x;

    %%Generate performance profiles%%
    tau_max = 2;
    [ADMM_performance] =  performance_profile(statisticsADMM.perf_prof,0.0001,tau_max); %.perf_prof
    [LT_performance] =  performance_profile(statisticsLT.perf_prof,0.0001, tau_max); %.perf_prof
    [VLT_performance] = performance_profile(statisticsVLT.perf_prof,0.0001, tau_max);
    tau= 1:0.0001:tau_max;

    %Combining problems to create single signals
%     bLTFull = [bLT;bLT];
%     bADMMFull = [bADMM;bADMM];

    

    ProblemStart=ProblemStart+ProblemSize;

    num_experiments = num_experiments-ProblemTime;

    %g=g+1;
    h = h+1;

    k=k+1;

end


%Stacking the collumns of bLT and bADMM
bLTFull = reshape(bLT,1,[])';
bADMMFull = reshape(bADMM,1,[])';

% disp(size(bLTFull));
% disp(size(bADMMFull));
% 
% disp(size(A));

figure(1)
%subplot(2,2,4);
stairs(tau,LT_performance.val,'DisplayName','LT','LineWidth',2); %.val
hold on
stairs(tau,VLT_performance.val,'DisplayName','LT V','LineWidth',2);
stairs(tau,ADMM_performance.val, 'DisplayName','Vanilla ADMM','LineWidth',2); %.val
lgd2 = legend('Location','southeast');
fontsize(lgd2,34,'points');
ylim([0 1.1]);
% xlim([1 2]);
%yticks([0 1]);
set(gca,'YTick',[0 1],'FontSize',18);
set(gca, 'XTick',[1 1.2 1.4 1.6 1.8 2], 'FontSize',36);

figure(2)
%subplot(2,2,1);
plot(GraphDomain,bEntire);
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

figure(3)
%subplot(2,2,2);
plot(GraphDomain,bLTFull);
title('Signal Reconstructed Using LT Variation');
xlabel('Time (s)');
ylabel('Amplitude');

figure(4)
%subplot(2,2,3);
plot(GraphDomain,bADMMFull);
title('Signal Reconstructed Using ADMM');
xlabel('Time (s)');
ylabel('Amplitude');


% K = length(history.objval);
% LTK = length(LThistory.objval); 
% 
% P3= history.Rach_diff;

% sound(y,SamPerSec);
% disp(size(y));


%Playing Sound
% sound(btest,SamPerSec);
% clear sound
% sound(bLT,SamPerSec);
% 
% sound(bADMM,SamPerSec);

%Saving our LT reconstructed signal as a file for later listening
% audiowrite('chopincompletelyreconstructedLT2.wav',bLTFull,SamPerSec);

