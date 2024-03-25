function [performance_data] = performance_profile(r_list, dx, tau_max)

sorted = sort(r_list);

tau = 1;
y = 0;
j = 1;
n = length(r_list);

for k = 0:ceil((tau_max-1)/dx)
    tau = tau + dx;
    while j <= n && sorted(j) <= tau
        y = y+1;
        j = j+1;
    end
    performance_data.val(k+1) = y/n;
    
end    
    
    

