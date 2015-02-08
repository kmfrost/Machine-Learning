function   shuffled = shuffle8(array)

%shuffle the input array by looping through
%select one input, then skip 7 (that's why it's called shuffle8)

shuffled = [array(1:8:end, :),
            array(2:8:end, :),
            array(3:8:end, :),
            array(4:8:end, :),
            array(5:8:end, :),
            array(6:8:end, :),
            array(7:8:end, :),
            array(8:8:end, :)];