function [mex_vals] = getMexicanHat(s,x)


    mex_vals = (1.0-0.5*(x.^2/s^2)).*exp(-0.5*(x.^2/s^2));

    mex_vals(abs(mex_vals) < 1E-5) = 0;

end

