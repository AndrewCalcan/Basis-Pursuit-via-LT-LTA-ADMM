function [y] = proj_box(x, rho)
    
    K = length(x);
    y = zeros(K,1);
    
    for k=1:K
        if abs(x(k)) > rho
            y(k) = rho*x(k)/abs(x(k));
        else
            y(k) = x(k);
        end

end