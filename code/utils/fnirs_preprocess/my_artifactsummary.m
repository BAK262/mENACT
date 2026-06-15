function [nMA, percMA] = my_artifactsummary(data, probe, mlActAuto, tMotion, tMask, stdThresh, ampThresh, tIncMan)
% Summary the number of motion artifacts and the data percentage
%   contaminated by motion artifacts.
%
% [nMA, percMA] = my_artifactsummary(data, probe, mlActAuto, tMotion, tMask, stdThresh, ampThresh, tIncMan)
[~, tIncCh] = hmrR_MotionArtifactByChannel( ...
        data, probe, {}, mlActAuto, {}, ...
        tMotion, tMask, stdThresh, ampThresh);

tIncDiff    = diff(tIncMan{1});
trialBegin  = find(tIncDiff == 1)+1;
if tIncMan{1}(1)==1
    trialBegin = [1; trialBegin];
end
trialEnd    = find(tIncDiff == -1);
if tIncMan{1}(end)==1
    trialEnd = [trialEnd; length(tIncMan{1})];
end
if length(trialBegin) ~= length(trialEnd)
    error('Trial detection error')
end

nMA = zeros(1,size(data.dataTimeSeries,2));
tMA = zeros(1,size(data.dataTimeSeries,2));
for j = 1:length(trialBegin)
    Inc     = tIncCh{1}(trialBegin(j):trialEnd(j),:);
    tMA     = tMA + sum(1-Inc);
    firstMA = 1-Inc(1,:);
    otherMA = sum(diff(Inc) == -1);
    nMA     = nMA + firstMA + otherMA;
end
mlActSamples = sum(tIncMan{1}) * sum(mlActAuto{1}(:,3));
nMA    = sum(nMA);
percMA = 100*sum(tMA)/mlActSamples;
end