function [z, history, Rachold, FullRach] = basis_pursuit(A, b, rho, alpha)
% basis_pursuit  Solve basis pursuit via ADMM
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
MAX_ITER = 10000;
ABSTOL   = 1e-3;
RELTOL   = 1e-3;

%% Data preprocessing

[m n] = size(A);

%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);
Rach = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% precompute static variables for x-update (projection on to Ax=b)
AAt = A*A';
P = eye(n) - A' * (AAt \ A);
q = A' * (AAt \ b);

FullRach = zeros(n,MAX_ITER); %%%%%%%%%%%%%%%%%%%%%%%%
i = 1; %%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:MAX_ITER
    % x-update
    x = P*(z - u) + q;

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, 1/rho);

    uold=u;
    u = u + (x_hat - z);
    
    %I save the previous DR iterate
    Rachold = Rach;
    %I compute the next DR iterate
    Rach = u + rho*z; 
    
    
      FullRach(:,i) = Rach; %%%%%%%%%%%%%%%%%%%%%%%%
      i = i+1; %%%%%%%%%%%%%%%%%%%%%%%%
   

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, x);
    
    history.u_diff(k)  = norm(uold-u);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold)); %residual check
    
    %I record the DR iterate subsequent differences
    history.Rach_diff(k) = norm(Rachold-Rach);
    
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

%%Taking out the 0 columns
FullRach(:,all(FullRach==0)) = []; %%%%%%%%%%%%%%%%%%%%%%%%

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