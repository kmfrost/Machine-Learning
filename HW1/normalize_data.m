%normalize data

load('SceneCateg.mat')

[ntrain, ndim] = size(trainfeatgist);

mean_vector = repmat(mean(trainfeatgist,1), ntrain, 1);
max_vector = max(trainfeatgist, [], 1);
min_vector = min(trainfeatgist, [], 1);
range_vector = repmat(max_vector-min_vector, ntrain, 1);

n_trainfeatgist = (trainfeatgist-mean_vector)./range_vector;



save n_trainfeatgist
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