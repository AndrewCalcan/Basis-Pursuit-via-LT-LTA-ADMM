function [statisticsLT, statisticsADMM] = rand_basis_pursuit_experiments(num_experiments)

% compute IDCT (6600 by 6600) here, so that it is outside the loop, so that you don't compute it every single time.
% Since we keep 10% of samples, IDCTM is (4410,44100)

for k = 1:num_experiments

%     rng('default');
%     rng(8);
% % 	rand('seed', k);
% % 	randn('seed', k);

    n = 5000; %4410; %6600
%     %Creating the first column of the IDCTM by multiplier a column vector of
%     %ones by (1/sqrt(2)): 
%     IDCTMCol11 = ones(n,1);
%     IDCTMC1 = (1/sqrt(2))*IDCTMCol11;
%     
%     %Remaining part of the IDCTM generated: 
%     piCoeff = (1:2:(2*n))'.*(1:(n-1));
%     IDCTMRem = cos((piCoeff*pi)/(2*n));
%     
%     %Creating the full IDCTM by concating IDCTMC1 (column 1) and IDCTMRem:
%     FullIDCTM = cat(2,IDCTMC1,IDCTMRem);
%     A = (sqrt(2/n))*FullIDCTM;

	%n = 30; %4410; %6600
	m = 500; %441; %660
	A = randn(m,n); %instead of calling a random A, you want to specify that A is the IDCT with the correct rows removed (size 660 x 6600)

	x = sprandn(n, 1, 0.1*n); %you will not have x
	b = A*x; %instead of calling b = A*x, you just want b to be the signal that you generated with the correct entries removed (size 660)

	%xtrue = x; %you will not use xtrue, and indeed, it looks as though neither have I.

%     disp(A);
%     disp(x);
    
    [LTx LThistory] = fast_basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0);
    [VLTx VLThistory] = var_basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0);
	[x history] = basis_pursuit(A, b, 1.0, 1.0);

	K = length(history.objval);
	LTK = length(LThistory.objval);

	P3= history.Rach_diff;

%     figure(1)
% 	semilogy(1:K,P3,'DisplayName','ADMM dual','LineWidth',2);
% 	title('Regular ADMM vs LT centering')
% 
% 	hold on
% 
% 	P1=history.r_norm;
% 	plot(1:K,P1,'DisplayName','ADMM |x-z|','LineWidth',2);
% 
% 	LTP3= LThistory.Rach_diff;
% 	plot(1:LTK,LTP3,'DisplayName','LT dual','LineWidth',2);
% 
% 	hold on
% 
% 	LTP1=LThistory.r_norm;
% 	plot(1:LTK,LTP1,'DisplayName','LT |x-z|','LineWidth',2);
% 
% 	hold on 
% 
% 	P2 = history.s_norm;
% 	LTP2 = LThistory.u_diff;
% % 	plot(1:K,P2,'LineWidth',2);
% % 	plot(1:LTK,LTP2,'LineWidth',2);
% 
% 	hold off
% 
% 	legend
	
	%update statistics
	statisticsVLT.objval(k) = VLThistory.objval(length(VLThistory.objval)); %LT variation
	statisticsLT.objval(k) = LThistory.objval(length(LThistory.objval)); %objective function values
	statisticsADMM.objval(k) = history.objval(length(history.objval));  %objective function values
	
    statisticsVLT.r_norm(k) = VLThistory.r_norm(length(VLThistory.r_norm)); %LT variation
	statisticsLT.r_norm(k) = LThistory.r_norm(length(LThistory.r_norm));
	statisticsADMM.r_norm(k) = history.r_norm(length(history.r_norm));
	
    statisticsVLT.iterates(k) = length(VLThistory.objval); %LT variation
	statisticsLT.iterates(k) = length(LThistory.objval);    %number of iterates it took LT to solve
	statisticsADMM.iterates(k) = length(history.objval);    %number of iterates it took ADMM to solve
    
%     if length(LThistory.objval) < length(history.objval)
%         statisticsLT.wins(k) = 1;
%         statisticsADMM.wins(k) = 0;
%     elseif length(LThistory.objval) > length(history.objval)
%         statisticsADMM.wins(k) = 1;
%         statisticsLT.wins(k) = 0;
%     else
%         statisticsLT.wins(k) = 0;
%         statisticsADMM.wins(k) = 0;
%     end
%     
%     if length(history.objval) == 50000
%         statisticsADMM.fails(k) = 1;
%         statisticsLT.LTvsADMM(k) = 0; %I record the difference as a zero if ADMM fails to solve
%     else
%         statisticsADMM.fails(k) = 0;
%         statisticsLT.LTvsADMM(k) = norm(x-LTx); %I record the difference between LT solutions and ADMM solutions when they both solved the problem
%     end
%     
%     if length(LThistory.objval) == 50000
%         statisticsLT.fails(k) = 1;
%     else
%         statisticsLT.fails(k) = 0;
%     end

%     %%Statistics for Algorithm Performance Profiles%%
% 
%     %Identifying the algorithm with shorter computation time
%     fastest_time = min(length(LThistory.objval),length(history.objval));
% 
%     %Creating the performance ratio for Lyapunov-Surrogate Method
%     statisticsLT.perf_prof(k) = length(LThistory.objval)/fastest_time;
% 
%     %Creating thr performance ratio for ADMM
%     statisticsADMM.perf_prof(k) = length(history.objval)/fastest_time;

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
    
%     figure(1)
%     %subplot(3,2,1);
%     semilogy(1:K,P3,'DisplayName','ADMM','LineWidth',2); %semilogy(1:K,P3,'DisplayName','ADMM') 
%     title('ADMM vs LT') 
%     xlabel('Iteration');
%     ylabel('Change');
% 
%     hold on 
%  
%     P1=history.r_norm;
%     plot(1:K,P1,'DisplayName','Douglas-Rachford','LineWidth',2); %plot
%     %title('LT Centering'); 
%     hold on
% 
%     LTP1=LThistory.r_norm; 
%     plot(1:LTK,LTP1,'DisplayName','LT Primal','LineWidth',2); %plot
%     legend;
%     
%     hold on
% 
%     LTP3= LThistory.Rach_diff; 
%     plot(1:LTK,LTP3,'DisplayName','LT Dual','LineWidth',2); %plot
% 
%     hold off
% 
%     set(gca,'FontSize',22);

%Solving Using Moore Penrose Pseud Inverse
%MPx = pinv(A)*b;

    %Generate performance profiles%%
    tau_max = 10;
    [ADMM_performance] =  performance_profile(statisticsADMM.perf_prof,0.0001,tau_max); %.perf_prof
    [LT_performance] =  performance_profile(statisticsLT.perf_prof,0.0001, tau_max); %.perf_prof
    [VLT_performance] = performance_profile(statisticsVLT.perf_prof,0.0001, tau_max);
    tau= 1:0.0001:tau_max;
	
end	
	
figure(2)
stairs(tau,LT_performance.val,'DisplayName','LT', 'LineWidth',2); %.val
hold on
stairs(tau,VLT_performance.val,'DisplayName','LTA','LineWidth',2);
stairs(tau,ADMM_performance.val,'DisplayName','ADMM', 'LineWidth', 2); %.val
%title('Performance Profile');
lgd2 = legend('Location','southeast');
fontsize(lgd2,32,'points');
ylim([0 1.1]);
xlim([1 10]);
%yticks([0 1]);
set(gca,'YTick',[0 1],'FontSize',50);
set(gca, 'XTick',[1 2 3 4 5 6 7 8 9 10], 'FontSize',40);

%disp(x - MPx);
