% SYNTAX:
% mlActAuto = my_prunechannels_phoebe(data, probe, mlActMan, tIncMan, hpf, lpf, SCIthresh, PPthresh, winsecs, propthresh)
%
% DESCRIPTION:
% Prune channels from the measurement list by the scalp coupling index (SCI)
% and peak power metric, which measure the quality of the connection between
% the optode and the scalp. This function updates MeasListAct based on whether
% data 'd' meets these conditions as specified by 'SCIthresh', 'PPthresh', 
% 'propThresh'. This algorithm follows the procedure described by ->
%       Pollonini, L., Olds, C., Abaya, H., Bortfeld, H., Beauchamp, M. S., 
%       & Oghalai, J. S. (2014). Auditory cortex activation to natural speech
%       and simulated cochlear implant speech measured with functional 
%       near-infrared spectroscopy. Hearing Research, 309, 84–93. 
%       https://doi.org/10.1016/j.heares.2013.11.007
%
% INPUTS:
% d         - SNIRF object containing time course data (nTpts x nChannels)
% probe     - SNIRF object describing the probe - optode positions and
%             wavelengths.
% mlActMan  - Cell array of vectors, one for each time base in data,
%             specifying active/inactive channels with 1 meaning active,
%             0 meaning inactive.
% tInc      - Cell array for each data block with vectors of length time
%             points where 1's indicating data included and 0's not.
% hpf       - high pass filter frequency (Hz)﻿ to preserve only the cardiac component.
% lpf       - low pass filter frequency (Hz) ﻿to preserve only the cardiac component.
% SCIthresh - the quality is not good if one channels' SCI < SCIthresh.
% PPthresh  - the quality is not good if one channels' peak power < PPthresh.
% winsecs   - moving window in second for computing SCI and peak power.
% propthresh- if the proportion of bad time windows was greater than propthresh,
%             the channel will be excluded as active channels.
%
% OUTPUTS:
% mlActAuto - cell array of all data blocks - each data block is an array
%             of true/false for all channels in the contanining data block
%             specifying active/inactive status. (# of data blocks x # of Channels)
% 
% PARAMETERS:
% hpf: 0.5
% lpf: 2.5
% SCIthresh: 0.8
% PPthresh: 0.1
% winsecs: 3
% propthresh: 0.8

function mlActAuto = my_prunechannels_phoebe(data, probe, mlActMan, tIncMan, hpf, lpf, SCIthresh, PPthresh, winsecs, propthresh)

% Init output 
mlActAuto = cell(length(data),1);

% Check input args
if nargin<10
    disp( 'USAGE: mlActAuto = my_prunechannels_phonebe(data, probe, mlActMan, tIncMan, hpf, lpf, SCIthresh, PPthresh, winsecs, propthresh)' )
    return
end
if isempty(tIncMan)
    tIncMan = cell(length(data),1);
end
if isempty(mlActMan)
    mlActMan = cell(length(data),1);
end
if isempty(hpf)
    hpf = 0.5;
end
if isempty(lpf)
    lpf = 2.5;
end
if isempty(SCIthresh)
    SCIthresh = 0.8;
end
if isempty(PPthresh)
    PPthresh = 0.1;
end
if isempty(winsecs)
    winsecs = 3;
end
if isempty(propthresh)
    propthresh = 0.8;
end

% band-pass filter to preserve only the cardiac component
data = hmrR_BandpassFilt(data,hpf,lpf);

% prune channels
for iBlk = 1:length(data)

    d        = data(iBlk).GetDataTimeSeries();
    t        = data(iBlk).GetTime();
    fs       = 1/(t(2)-t(1));
    MeasList = data(iBlk).GetMeasList();
    Lambda   = probe.GetWls();
    if length(Lambda)~=2
        error(['Channel pruning based on the scalp coupling index is ' ...
            'currently only applicable to signal acquisition systems ' ...
            'using two wavelengths.'])
    end
    
    % only consider included data time (by trials)
    if isempty(tIncMan{iBlk})
        tIncMan{iBlk} = ones(length(t),1);
    end
    tInc = tIncMan{iBlk};
    startsample = find(diff(tInc)==1)+1;
    endsample   = find(diff(tInc)==-1);
    nTrials     = length(startsample);
    trialWins   = floor((endsample-startsample)./(winsecs*fs));
    
    % Start by including all channels
    chnList = ones(size(MeasList,1),1);
    chnExcl = [];

    % calculate scalp coupling index and peak power
    SrcDetPairs = data(iBlk).GetMeasListSrcDetPairs();
    for p = 1:size(SrcDetPairs,1)
        SrcIdx = SrcDetPairs(p,1);
        DetIdx = SrcDetPairs(p,2);
        ChnIdx = intersect(find(MeasList(:,1)==SrcIdx),find(MeasList(:,2)==DetIdx));
        sci = zeros(1,sum(trialWins));
        pp  = zeros(1,sum(trialWins));
        i = 0;
        for trialNo = 1:nTrials
            for winNo = 1:trialWins(trialNo)
                i = i+1;
                starti  = startsample(trialNo)+winsecs*fs*(winNo-1);
                endi    = startsample(trialNo)+winsecs*fs*winNo-1;
                x   = d(starti:endi, ChnIdx(1));
                y   = d(starti:endi, ChnIdx(2));
                [xy, lags]  = xcorr(zscore(x),zscore(y),'unbiased');
                sci(i)      = xy(lags==0); % scalp coupling index
                [pxy,~]     = periodogram(xy,hamming(length(xy)),length(xy),fs,'power');
                pp(i)       = max(pxy); % peak power
            end
        end
        if mean(sci>SCIthresh & pp>PPthresh) < propthresh
            chnExcl = [chnExcl, ChnIdx];
        end
    end
    chnExcl = unique(chnExcl);
    chnList(chnExcl) = 0;
    
    % merge with manully excluded channels
    mlActMan{iBlk} = mlAct_Initialize(mlActMan{iBlk}, MeasList);
    manExcl = find(mlActMan{iBlk}(:,3) == 0);
    chnList = chnList(:) & mlActMan{iBlk}(:,3);
    
    % print results
    if ~isempty(chnExcl)
        fprintf('Excluded channels [ %s ] based on SCI < %s and peak power < %s\n', num2str(chnExcl(:)'), num2str(SCIthresh), num2str(PPthresh));
    end
    if ~isempty(manExcl)
        fprintf('Manually excluded channels [ %s ]\n', num2str(manExcl(:)'));
    end
    
    % update MeasListAct  
    mlActAuto{iBlk} = mlAct_Initialize(chnList, MeasList);
end

