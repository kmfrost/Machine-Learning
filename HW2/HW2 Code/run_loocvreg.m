%this script runs each version of the loocverror calculation 100 times
%and computes the average time for each function and the average speedup

load('cars.mat');

num_runs = 100;
slow_time = zeros(1,num_runs);
fast_time = zeros(1, num_runs);

for ii=1:num_runs
    %run slow version
    f = @() loocvreg_slow(xTr.', yTr.');
    slow_time(ii) = timeit(f);

    %run fast version
    g = @() loocvreg_fast(xTr.', yTr.');
    fast_time(ii) = timeit(g);
end

avg_slow_time = mean(slow_time)
avg_fast_time = mean(fast_time)
avg_speedup = avg_slow_time/avg_fast_time
