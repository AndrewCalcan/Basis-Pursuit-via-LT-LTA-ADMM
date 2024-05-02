function [statisticsLT, statisticsADMM] = rand_basis_pursuit_experiments(num_experiments)

% compute IDCT (6600 by 6600) here, so that it is outside the loop, so that you don't compute it every single time.
% Since we keep 10% of samples, IDCTM is (4410,44100)

for k = 1:num_experiments

    n = 5000; %4410; %6600

	m = 500; %441; %660
	A = randn(m,n); %instead of calling a random A, you want to specify that A is the IDCT with the correct rows removed (size 660 x 6600)

	x = sprandn(n, 1, 0.1*n); %you will not have x
	b = A*x; %instead of calling b = A*x, you just want b to be the signal that you generated with the correct entries removed (size 660)

    
    [LTx LThistory] = fast_basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0);
    [VLTx VLThistory] = var_basis_pursuit_LT_smart_fast(A, b, 1.0, 1.0);
	[x history] = basis_pursuit(A, b, 1.0, 1.0);

	K = length(history.objval);
	LTK = length(LThistory.objval);

	P3= history.Rach_diff;

	
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
lgd2 = legend('Location','southeast');
fontsize(lgd2,32,'points');
ylim([0 1.1]);
xlim([1 10]);
set(gca,'YTick',[0 1],'FontSize',50);
set(gca, 'XTick',[1 2 3 4 5 6 7 8 9 10], 'FontSize',40);
