function [Y,Xnew,Ynew]=predMIMO4(X,H,numsides,fs,Yact)
    %this is a method of the binnedData class and should live in the
    %@binnedData folder with the other class methods
    %
    %function to compute the output of a MIMO system
    %
    %    USAGE:   Y=predMIMO(X,H,numsides,fs,Yact)
    %
    %
    %    Y        : Columnwise outputs [y1 y2 ...] to the unknown system
    %    X        : Columnwise inputs  [x1 x2 ...] to the unknown system
    %    H        : the nonparametric filters between X and Y.
    %    numsides : determine a causal (1 side) or noncausal 
    %               (2 sides) response.
    %    fs		  : Sampling rate (default=1)
    %    Yact     : Actual Y values so they can be truncated to match the
    %               predicted data length
    %
    % The  filter matrix, H needs to be organized in columns as:
    %     H=[h11 h21 h31 ....;
    %        h12 h22 h32 ....;
    %        h13 h23 h33 ...;
    %        ... ... ... ...]
    %  Which represents the system:
    %  y1=h11 + h12*x1 + h13*x2 + h14*x3 + ...     
    %  y2=h21 + h22*x1 + h23*x2 + h24*x3 + ...     
    %  y3=h31 + h32*x1 + h33*x2 + h34*x3 + ...    
    %  ... 
    %
    %[Y,Xnew,Ynew]=predMIMO(X,H,numsides,fs,Yact)
    %   same as above, but returns truncated X and Y with lengths that match
    %   the Y vector for easy plotting/VAF comparison

    % EJP April 1997, CE October 2013, TT 2016

    if size(X,2)>size(X,1) 
        error('predMIMO4:rowmatrix','inputs must be column matrixes')
    elseif ~(all(size(X,1)==size(Yact,1)))
        error('predMIMO4:sizeMismatch','actual output must havesame number of rows as input')
    end

    mxy = H(1,:);
    H = H(2:end,:);

    [numpts,Nx]=size(X);
    [nr,Ny]=size(H);
    fillen=nr/Nx;

    if (rem(nr,Nx) ~= 0)
       disp('Input size does not match dimensions of filter matrix')
       return
    end
    %Allocate memory for the outputs
    Y=zeros(numpts,Ny);
    numpts=size(X,1);
    halflen=ceil(length(fillen)/2);
    for i=1:Ny
        for j=1:Nx
            fil=H(1+(j-1)*fillen:j*fillen,i);
            if numsides==2
                y=filter(fil,1,[X(:,j);zeros(halflen,1)]);
                Y(:,i)=y(halflen:numpts +halflen -1);
            else
                Y(:,i)=filter(fil,1,X(:,j));
            end
        end
        %add back means:
        Y(:,i) = Y(:,i)+mxy(i);
    end

    if nargout>1
        if numsides==2
            skip=(fillen-1)/2;
            Y=Y(skip+1:numpts-skip,:);
            Xnew=X(skip+1:numpts-skip,:);
            Ynew=Yact(skip+1:numpts-skip,:);
        else
            Y=Y(fillen:numpts,:);
            Xnew=X(fillen:numpts,:);
            Ynew=Yact(fillen:numpts,:);
        end
    end
		
end