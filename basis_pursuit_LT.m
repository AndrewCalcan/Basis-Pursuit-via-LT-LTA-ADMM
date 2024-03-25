function [z, history, LTRachold, LTFullRach, FullyLT] = basis_pursuit_LT_smart_fast(A, b, rho, alpha)
% basis_pursuit  Solve basis pursuit via LT
%
% [x, history] = basis_pursuit(A, b, rho, alpha)
% 
% Solves the following problem via ADMM:
% 
%   minimize     ||x||_1
%   subject to   Ax = b
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
% 
% rho is the augmented Lagrangian parameter. 
%
% alpha is the over-relaxation parameter (typical values for alpha are 
% between 1.0 and 1.8).
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 10000; %%%%%
ABSTOL   = 1e-4;
RELTOL   = 1e-4;

%% Data preprocessing

[m n] = size(A);

%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);
y1 = zeros(n,1);
y2 = zeros(n,1);
y3 = zeros(n,1);

LTRach = zeros(n,1); %%%%%%%%%%%%%%%%%%%%%%%%

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% precompute static variables for x-update (projection on to Ax=b)
AAt = A*A';
P = eye(n) - A' * (AAt \ A);
q = A' * (AAt \ b);


%Now we do the accelerated version
uREG = u;
uLT = u;
zLT = z;
j = -1;

FullyLT = zeros(30,100);

LTAcceptCounter = 0; %the number of LT update candidates accepted

zREG = z;
% LTFullRach = zeros(n, MAX_ITER); %%%%%%%%%%%%%%%%%%%%%%%%
% i = 1; %%%%%%%%%%%%%%%%%%%%%%%%
l = 1;
for k = 1:MAX_ITER
    
    uold = u;
    
    if j == 2 %if it's time to check the centering step, enter
        % x-update
       xREG = P*(z - uREG) + q;
        xLT = P*(zLT - uLT) + q;
%         if norm(xLT - zLT,2) < norm(xREG - zREG,2)
       if objective(A, b, xREG) > objective(A,b,xLT)
%        if 2 > 1
            x = xLT;
            u = uLT;
            j=0; %when we accept the centering step, we set the counter back to zero
            
       else %we reject the centered update, in which case we must update indices and leave counter unchanged
           x = xREG;
           u = uREG; 
           y1 = y2;
           y2 = y3;
           %j = 2; counter stays unchanged
           
       end
    else %if it isn't time to check the centering step, skip the objective function check and update normally
        x = P*(z - u) + q;
        j = j+1;
    end

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, 1/rho);
	
	%update the multiplier
	u = u + (x_hat - z);
    
    %y_{k-1} update
    if j == 0
        y1 = u+rho*z;
    elseif j == 1
        y2 = u+rho*z;
    else %then j == 2 and so
        y3 = u+rho*z;
    end
    
    %compute the alternative multiplier candidate based on centering
    if j == 2 && colinear_check(y1,2*y2-y1,piT(y1,y2,y3),10^(-20)) == 1

        %on third updates, I circumcenter the dual if not colinear.
        %I then set shadow u to be prox_cd2 of the dual governing sequence
		yLT=circumcenter(y1,2*y2-y1,piT(y1,y2,y3));

%         FullyLT(:,l) = y1;
%         l = l+1;
%         FullyLT(:,l) = y2;
%         l = l+1;
%         FullyLT(:,l) = y3;
%         l = l+1;
%         FullyLT(:,l) = yLT;
%         l = l+1;
% 
%         LTAcceptCounter = LTAcceptCounter + 1;
        
        uLT = proj_box(yLT,1); 
        zLT = yLT-uLT;
    else
        % otherwise, I do the regular ADMM update
        uLT = u; 
    end
    uREG = u;
    zREG = z;
        
%     threshold = 0.1*max(z);
%     for h = 1:4410
%         z(h,:) = shrinkage(z(h,:),threshold);
%     end    
    

    
    

    % diagnostics, reporting, termination checks
	
	    
    LTRachold = LTRach; %I save the previous DR iterate
    LTRach = u + rho*z; %I compute the next DR iterate
	
	
    history.objval(k)  = objective(A, b, x);
    
    history.u_diff(k)  = norm(uold-u);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    
    %I record the DR iterate subsequent differences
    history.Rach_diff(k) = norm(LTRachold-LTRach);
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
    history.eps_u(k)= sqrt(n)*ABSTOL + RELTOL*norm(u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end



% disp(LTAcceptCounter);





if ~QUIET
    toc(t_start);
end

end







function obj = objective(A, b, x)
    obj = norm(x,1);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end