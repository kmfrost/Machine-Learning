function   cv_error = knncv_iter(traindata, trainlabels, n, k, f, D)

% traindata is a ntrain x ndim matrix that represents your training data (each row is one picture and each column is a feature);
% trainlabels is an ntrain x 1 vector holding the labels (integers between 1 and 8) for the training data;
% n is how many groups to partition the training data into
% k is number of neighbors;
% f is the distance function you will use (e.g f=?sqeuclidean?);
% D is an ntrain x ntest matrix of precomputed pairwise distances (optional in the function input and your inside code should handle the case where it does not exist, e.g. by calling f); 
[ntrain, ndim] = size(traindata);

traindata_shuf = shuffle8(traindata);
trainlabels_shuf = shuffle8(trainlabels);

if mod(ntrain, n) ~= 0
    error('Please select an n that is a multiple of %d', ntrain);
end

part_size = ntrain/n;

train_sect = zeros(n, 2);
for part=1:n
    train_sect(part, :) = [(part-1)*part_size+1, part * part_size];
end

cross_val_percent = zeros(1, k);
train_error = zeros(1,k);

for k_iter=1:k
    fprintf('\nk = %d\n', k_iter);
    errors = zeros(n, 1);

    for iter=1:n
        fprintf('Iteration: %d\n', iter);

        %divide the data into a test section and a training section
        %where the test section represents 1 part and the training section
        %represents n-1 parts
        test_data= traindata_shuf(train_sect(iter, 1):train_sect(iter,2), :);
        test_labels = trainlabels_shuf(train_sect(iter,1):train_sect(iter,2));
        if train_sect(iter,1) == 1
            iter_train_data = traindata_shuf(train_sect(iter,2)+1:end, :);
            iter_train_labels = trainlabels_shuf(train_sect(iter,2)+1:end);
        else
            iter_train_data = zeros(part_size*(n-1), ndim);
            iter_train_labels = zeros(part_size*(n-1),1);

            iter_train_data(1:train_sect(iter,1)-1, :) = traindata_shuf(1:train_sect(iter,1)-1, :);
            iter_train_labels(1:train_sect(iter,1)-1) = trainlabels_shuf(1:train_sect(iter,1)-1);

            iter_train_data(train_sect(iter,1):end, :) = traindata_shuf(train_sect(iter,2)+1:end, :);
            iter_train_labels(train_sect(iter,1):end) = trainlabels_shuf(train_sect(iter,2)+1:end);
        end

        labels = knn(iter_train_data, iter_train_labels, test_data, k_iter, f, D);
        errors(iter) = nnz(labels-test_labels);

    end

    cross_val_percent(k_iter) = mean(errors)/part_size;

    labels = knn(traindata, trainlabels, traindata, k_iter, f, D);
    train_error(k_iter) = nnz(labels-trainlabels)/ntrain;
end

plot(cross_val_percent);
hold on;
plot(train_error);
legend('Cross-value Error', 'Train Error');
grid on;


