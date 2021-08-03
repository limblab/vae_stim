function [ dist ] = getDistance( in_1, in_2 )

    dist = sqrt(sum((in_1 - in_2).^2,2));

end

