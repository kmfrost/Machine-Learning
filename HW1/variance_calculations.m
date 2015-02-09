%variance calculator

load('SceneCateg.mat')

[ntrain, ndim] = size(trainfeatgist);
part_size = ntrain/8;

train_var = zeros(8, ndim);

for i=1:8
    train_var(i, :) = (var(trainfeatgist(1+(i-1)*part_size:i*part_size, :)).^(1/2));
end

weights = 1./mean(train_var,1);  % average the variance for each column
weights = weights/max(weights);
diag_weights = diag(weights);

save diag_weights.mat diag_weights
%{
for i=1:8
    figure;
    plot(train_var(i, :));
    hold on;
    plot(weights);
    title(i);
    legend('Column Data', 'Weights');
end
%}