function  out = my_ttestpara(X,varargin)
% Do t-test via function `ttest(X)` and return the parameter(s) specified by
%   `params` as following. If no `param` was specified, return all.
%
% Params (see also `ttest`):
%   - h
%   - p
%   - ci
%   - tstat
%   - df
%   - sd
%
% out = my_ttestpara(X,[params])


[h,p,ci,stats] = ttest(X);

params = {'h','p','ci','tstat','df','sd'};
if ~isempty(varargin)
    assert(all(ismember(varargin, params)), ...
        'The optional input argument(s) must be chose from "h", "p", "ci", "tstat", "df", and "sd".')
    params = varargin;
end

out = [];
for i = 1:length(params)
    param   = params{i};
    out     = appendvars(out, h, p, ci, stats, param);
end

    function out = appendvars(out, h, p, ci, stats, param)
        
        switch param
        
            case 'h'
                out = [out; h];
        
            case 'p'
                out = [out; p];
        
            case 'ci'
                out = [out; reshape(ci,numel(ci),1)];
        
            case 'tstat'
                out = [out; stats.tstat];
        
            case 'df'
                out = [out; stats.df];
        
            case 'sd'
                out = [out; stats.sd];
        end

    end

end