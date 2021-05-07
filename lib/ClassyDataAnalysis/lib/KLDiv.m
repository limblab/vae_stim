function div=KLDiv(Q,P,varargin)
    %computes the Kullback–Leibler divergence from Q to P (note that the KL
    %divergence is assymmetric and KLDiv(Q,P)~=KLDIV(P,Q). Accepts an
    %optional argument to define whether the output should be in bits or
    %nats (log base 10 or log base e). Wikipedia actually has a pretty good
    %article on this.
    if isempty(varargin)
        basis='bits';
    else
        basis=varargin{1};
    end
    switch basis
        case 'bits'
            div=sum(P(:).*log10(Q(:)./P(:)));
            %note that P(:) expands P into a vector from whatever N-D matrix
            %it originally was. That ensures that this will work on N-d arrays, 
            %rather than only on 1D intput
        case 'nats'
            div=sum(P(:).*log(Q(:)./P(:)));
        otherwise
            error('KLDiv:badBasis','KLDiv only accepts the strings bits or nats as input for the basis')
    end
end