function predlabels = knn(traindata, trainlabels, testdata, k, f, D)

% traindata is a ntrain x ndim matrix that represents your training data (each row is one picture and each column is a feature);
% trainlabels is an ntrain x 1 vector holding the labels (integers between 1 and 8) for the training data;
% testdata is a ntest x ndim matrix that represents your testing data;
% k is number of neighbors;
% f is the distance function you will use (e.g f=?sqeuclidean?);
% D is an ntrain x ntest matrix of precomputed pairwise distances (optional in the function input and your inside code should handle the case where it does not exist, e.g. by calling f); 
% predlabels is ntest x 1 vector holding predicted labels for the test instances.

load('diag_weights.mat');

%check the dimensions of both the incoming vectors
[ntrain, ndim] = size(traindata);
[ntest, ~] = size(testdata);

predlabels = zeros(ntest,1);

for testpoint = 1:ntest
    
    %Create a matrix where each row is a copy of the test data
    test_matrix = ones(ntrain, 1) * testdata(testpoint, :);

    if D ~= 0
        [~, indices] = n_min(diag(D), k);
    elseif strcmp(f, 'sqeuclidean')
        difference_matrix = traindata - test_matrix;
        distance_matrix = difference_matrix * difference_matrix.';
        [~, indices] = n_min(diag(distance_matrix), k);  
        
    elseif strcmp(f, 'w_sqeuclidean')
        difference_matrix = traindata - test_matrix;
        distance_matrix = difference_matrix * diag_weights * difference_matrix.';
        [~, indices] = n_min(diag(distance_matrix), k);
        
    elseif strcmp(f, 'mahalanobis')
        difference_matrix = traindata - test_matrix;
        distance_matrix = difference_matrix * nancov(traindata) * difference_matrix.';
        [~, indices] = n_min(diag(distance_matrix), k);
        
    elseif strcmp(f, 'w_mahalanobis')
        difference_matrix = traindata - test_matrix;
        distance_matrix = difference_matrix * diag_weights * nancov(traindata) * difference_matrix.';
        [~, indices] = n_min(diag(distance_matrix), k);        
    end

    n_labels = trainlabels(indices);
    predlabels(testpoint) = mode(n_labels);
end

csv_data = [(1:ntest)', predlabels];
dlmwrite('predlabels.csv','Image_ID,Category', 'delimiter', '', 'coffset', 1);
dlmwrite('predlabels.csv', csv_data, '-append');
