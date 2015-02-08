function [values, indices] = n_min(array, n)
     [sorted_array, array_indices] = sort(array);
     values = sorted_array(1:n);
     indices = array_indices(1:n);
end

