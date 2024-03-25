function [statisticsLT, statisticsADMM] = basis_pursuit_experiments(num_experiments)
%This function 



%%Time specifications:

%Number of samples taken per second:
SamPerSec = 300; 

%Sample increments:
dx = 1/SamPerSec;     

%Number of seconds samples are collected over:
StopTime = 0.1;  

%Total number of samples:
TotSam = SamPerSec * StopTime;

%Percentage of samples kept:
PerKept = 0.1;

%Number of samples to be kept:
NumPoints = SamPerSec * StopTime * PerKept;

%Domain:
Dom = (0:dx:StopTime - dx)';   

%Calculating the number of entries in the domain:
DomSize = size(Dom,1);%size(Dom,1);

%The number of frequencies:
NumFreq = 2;


%%Creating the IDCTM:

%Creating the first column of the IDCTM by multiplier a column vector of
%ones by (1/sqrt(2)): 
IDCTMCol11 = ones(DomSize,1);
IDCTMC1 = (1/sqrt(2))*IDCTMCol11;

%Remaining part of the IDCTM generated: 
piCoeff = (1:2:(2*DomSize))'.*(1:(DomSize-1));
IDCTMRem = cos((piCoeff*pi)/(2*DomSize));

%Creating the full IDCTM by concating IDCTMC1 (column 1) and IDCTMRem:
FullIDCTM = cat(2,IDCTMC1,IDCTMRem);
FullIDCTM = (sqrt(2/DomSize))*FullIDCTM;
 %use dctmx function, then transpose
%FullIDCTM = transpose(dctmtx(DomSize));

%pleasebetheidentity = FullIDCTM * transpose(FullIDCTM);

%disp(IDCTMC1);

NumFig = 1;

for k = 1:num_experiments

    %Generating a row vector containing NumPoints many numbers from 1 to 
    %DomSize. This is achieved by using randperm() to generate a row vector
    %of NumPoints unique integers from 1 to DomSize.
    xKept = randperm(DomSize,NumPoints);

	
	%Creating our modified IDCTM by deleting all rows not included in
    %xKept.
    A = FullIDCTM(xKept,:);
%take out after using dctmx function above

    %Generating a vector of (NumFreq x 1) numbers from [0,1].
    a = rand(NumFreq,1); 
    
    %Generating a vector of (NumFreq x 1) integers from [1,500].
    c = randi(30,NumFreq,1); 
    
    
    b = zeros(DomSize,1); %setting dimensions of b.
    for i = 1:NumFreq
        
        b = b + (a(i)*sin(pi*c(i)*Dom)); %Our signal
    
    end 

    bOG = b;

    %Establishes b as a vector of signal values containing only the entries
    %chosen to be kept.
    b = b(xKept,:);


    [LTx, LThistory] = basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0);
	[x, history] = basis_pursuit(A, b, 1.0, 1.0);

	K = length(history.objval);
	LTK = length(LThistory.objval);

	P3= history.Rach_diff;
	
	statisticsLT.objval(k) = LThistory.objval(length(LThistory.objval)); %objective function values
	statisticsADMM.objval(k) = history.objval(length(history.objval));  %objective function values
	
	statisticsLT.r_norm(k) = LThistory.r_norm(length(LThistory.r_norm));
	statisticsADMM.r_norm(k) = history.r_norm(length(history.r_norm));
	
	statisticsLT.iterates(k) = length(LThistory.objval);    %number of iterates it took LT to solve
	statisticsADMM.iterates(k) = length(history.objval);    %number of iterates it took ADMM to solve
    
    if length(LThistory.objval) < length(history.objval)
        statisticsLT.wins(k) = 1;
        statisticsADMM.wins(k) = 0;
    elseif length(LThistory.objval) > length(history.objval)
        statisticsADMM.wins(k) = 1;
        statisticsLT.wins(k) = 0;
    else
        statisticsLT.wins(k) = 0;
        statisticsADMM.wins(k) = 0;
    end

    fastest_time = min(length(LThistory.objval),length(history.objval));
    statisticsLT.perf_prof(k) = length(LThistory.objval)/fastest_time;
    statisticsADMM.perf_prof(k) = length(history.objval)/fastest_time;
    
    if length(history.objval) == 50000
        statisticsADMM.fails(k) = 1;
        statisticsLT.LTvsADMM(k) = 0; %I record the difference as a zero if ADMM fails to solve
    else
        statisticsADMM.fails(k) = 0;
        statisticsLT.LTvsADMM(k) = norm(x-LTx); %I record the difference between LT solutions and ADMM solutions when they both solved the problem
    end
    
    if length(LThistory.objval) == 50000
        statisticsLT.fails(k) = 1;
    else
        statisticsLT.fails(k) = 0;
    end
    

K = length(history.objval);
LTK = length(LThistory.objval); 

P3= history.Rach_diff;

figure(1)
%subplot(3,2,1);
semilogy(1:K,P3,'DisplayName','ADMM dual') 
title('Regular ADMM vs LT centering') 
 
hold on 
 
P1=history.r_norm;
plot(1:K,P1,'DisplayName','ADMM |x-z|');
 
hold on

LTP3= LThistory.Rach_diff; 
plot(1:LTK,LTP3,'DisplayName','LT dual');
 
hold on

LTP1=LThistory.r_norm; 
plot(1:LTK,LTP1,'DisplayName','LT |x-z|');


hold off

% %figure(2)
% subplot(3,2,2);
% plot(Dom,bOG);    
% title('Original Signal');
% 
% 
% %figure(3)
% subplot(3,2,3);
% plot(Dom,FullIDCTM*x);
% title('Reconstructed Signal from ADMM');
% 
% %figure(4)
% subplot(3,2,4);
% plot(Dom,FullIDCTM*LTx);
% title('Reconstructed Signal from Lyapunov-Surrogate Method');
% 
% %%generate performance profile%%
% tau_max = 2;
% [ADMM_performance] =  performance_profile(statisticsADMM.perf_prof,0.05,tau_max); %.perf_prof
% [LT_performance] =  performance_profile(statisticsLT.perf_prof,0.05, tau_max); %.perf_prof
% tau= 1:0.05: tau_max;
% 
% subplot(3,2,[5,6]);
% stairs(tau,LT_performance.val, 'DisplayName','LT'); %.val
% hold on
% stairs(tau,ADMM_performance.val, 'DisplayName','vanilla ADMM'); %.val
% title('Performance Profile');



NumFig = NumFig + 1;
	
end	