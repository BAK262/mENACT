% SYNTAX:
% mlActAuto = my_motioncorrect_phoebe(data, probe, mlActMan, tIncMan, hpf, lpf, SCIthresh, PPthresh, winsecs, propthresh)
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

function data = my_motioncorrect_phoebe(data, probe, mlActMan, tIncMan, hpf, lpf, SCIthresh, PPthresh, winsecs, propthresh)

% Check input args
if nargin<10
    disp( 'USAGE: data_d = my_motioncorrect_phonebe(data, probe, mlActMan, tIncMan, hpf, lpf, SCIthresh, PPthresh, winsecs, propthresh)' )
    return
end
if isempty(tIncMan)
    tIncMan = cell(length(data_filt),1);
end
if isempty(mlActMan)
    mlActMan = cell(length(data_filt),1);
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
mlActAuto = cell(length(data),1);
stepsecs = 1;

% band-pass filter to preserve only the cardiac component
data_filt = hmrR_BandpassFilt(data,hpf,lpf);

% prune channels
for iBlk = 1:length(data_filt)

    d_filt   = data_filt(iBlk).GetDataTimeSeries();
    t        = data_filt(iBlk).GetTime();
    fs       = 1/(t(2)-t(1));
    MeasList = data_filt(iBlk).GetMeasList();
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
    startsample = find(diff(tInc)==1)+1 - 3*fs;
    endsample   = find(diff(tInc)==-1) + 3*fs;
    nTrials     = length(startsample);
    trialWins   = 1+floor((endsample-startsample+1-winsecs*fs)./(stepsecs*fs));
    tIncCh      = ones(length(t),size(d_filt,2));

    % calculate scalp coupling index and peak power
    SrcDetPairs = data_filt(iBlk).GetMeasListSrcDetPairs();
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
                starti  = startsample(trialNo)+stepsecs*fs*(winNo-1);
                endi    = starti+winsecs*fs-1;
                x   = d_filt(starti:endi, ChnIdx(1));
                y   = d_filt(starti:endi, ChnIdx(2));
                [xy, lags]  = xcorr(zscore(x),zscore(y),'unbiased');
                sci(i)      = xy(lags==0); % scalp coupling index
                [pxy,~]     = periodogram(xy,hamming(length(xy)),length(xy),fs,'power');
                pp(i)       = max(pxy); % peak power
                if sci(i)<SCIthresh || pp(i)<PPthresh
                    tIncCh(starti:endi, ChnIdx) = 0;
                end
            end
        end
    end

    % correct motion using spline interpolation
    [dod, t, SD.MeasList, order] = data(iBlk).GetDataTimeSeries('matrix : reshape : wavelength');
    dod = dod(:,:);
    mlActAuto{iBlk} = mlAct_Initialize(mlActAuto{iBlk}, SD.MeasList);
    SD.MeasListAct  = mlAct_Matrix2BinaryVector(mlActAuto{iBlk}, SD.MeasList);
    dodLP = hmrR_BandpassFilt_Nirs(dod, fs, 0, 2);
    dodr = my_MotionCorrectSpline_Nirs(dodLP, t, SD, tIncCh, 0.99);
    dodr(:,order) = dodr(:,:);
    data(iBlk).SetDataTimeSeries(dodr);
end

