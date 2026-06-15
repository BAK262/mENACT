function [q, pThresh, pSignif] = my_fdr(p, varargin)
% Return the q-values, p-value threshold, and significance of the given 
%   matrix `p` (uncorrected p-values) at the given alpha-level (optional,
%   default=0.05).
%
% [q, pThresh, pSignif] = my_fdr(p, [alpha])

% Get the alpha level
if isempty(varargin)
    a = 0.05;
elseif 0<varargin{1} && varargin{1}<1
    a = varargin{1};
else
    error('Optional input argument [alpha] must be a scalar between 0 and 1.')
end

% Sort the uncorrected p-values
pSize   = size(p);
pVals   = reshape(p, numel(p), 1);
n       = length(pVals);
[pVals, idxSort] = sort(pVals, 'ascend');

% Calculate the q-values
qVals   = pVals * n ./ (1:n)';

% Determine the significance
pSignif = qVals < a;

% Get the significance threshold of p-values
pThresh = pVals(find(pSignif, 1, 'last'));
if isempty(pThresh)
    pThresh = 0;
end

% Recover the sorting and shape
[~, idxUnsort] = sort(idxSort);
q       = reshape(qVals(idxUnsort), pSize);
pSignif = reshape(pSignif(idxUnsort), pSize);

end