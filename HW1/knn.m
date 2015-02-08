function predlabels = knn(traindata, trainlabels, testdata, k, f, D)

% traindata is a ntrain x ndim matrix that represents your training data (each row is one picture and each column is a feature);
% trainlabels is an ntrain x 1 vector holding the labels (integers between 1 and 8) for the training data;
% testdata is a ntest x ndim matrix that represents your testing data;
% k is number of neighbors;
% f is the distance function you will use (e.g f=?sqeuclidean?);
% D is an ntrain x ntest matrix of precomputed pairwise distances (optional in the function input and your inside code should handle the case where it does not exist, e.g. by calling f); 
% predlabels is ntest x 1 vector holding predicted labels for the test instances.

%check the dimensions of both the incoming vectors
[ntrain, ndim] = size(traindata);
[ntest, ~] = size(testdata);

predlabels = zeros(ntest,1);

for testpoint = 1:ntest
    
    %Create a matrix where each row is a copy of the test data
    test_matrix = ones(ntrain, 1) * testdata(testpoint, :);

    if D ~= 0
        [~, index] = min(diag(D));
    elseif strcmp(f, 'sqeuclidean')
        difference_matrix = traindata - test_matrix;
        distance_matrix = difference_matrix * difference_matrix.';
        [~, index] = min(diag(distance_matrix));

        %[~, index] = min(diag(pdist2(traindata, test_matrix)));
        
    elseif strcmp(f, 'mahalanobis')
        disp('using Mahalanobis');
    end

    predlabels(testpoint) = trainlabels(index(1));
end

csv_data = [(1:ntest)', predlabels];
dlmwrite('predlabels.csv','Image_ID,Category', 'delimiter', '', 'coffset', 1);
dlmwrite('predlabels.csv', csv_data, '-append');
