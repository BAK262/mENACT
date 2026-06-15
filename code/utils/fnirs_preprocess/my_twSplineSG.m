% SYNTAX:
% data_d = my_SplineSG(data_d, mlActAuto, p, FrameSize_sec, turnon)
%
% DESCRIPTION et. al.:
% Trial-wise SplineSG, revised from hmrR_MotionCorrectSplineSG.m
%
% NOTE:
% Here, the 'trial' I denote is generally longer than 15s in my dataset.
% Hence, I haven't tested if it is possible and useful to do SplineSG
% on short signal series. BE CAREFUL!!!

function data_d = my_twSplineSG(data_d, mlActAuto, tIncMan, p, FrameSize_sec, turnon)

%%%%% added by myself to do SplineSG trial-wise
if isempty(tIncMan)
    tIncMan = cell(length(data_d),1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('turnon','var')
   turnon = 1;
end
if turnon==0
    return;
end
if isempty(mlActAuto)
    mlActAuto = cell(length(data_d),1);
end

for iBlk = 1:length(data_d)

    [dod, t, SD.MeasList, order] = data_d(iBlk).GetDataTimeSeries('matrix : reshape : wavelength');
    dod = dod(:,:);

    mlActAuto{iBlk} = mlAct_Initialize(mlActAuto{iBlk}, SD.MeasList);
    SD.MeasListAct  = mlAct_Matrix2BinaryVector(mlActAuto{iBlk}, SD.MeasList);

    fs = abs(1/(t(2)-t(1)));
    
    %%%%% added by myself to do SplineSG only for trial data
    if isempty(tIncMan{iBlk})
        tIncMan{iBlk} = ones(length(t),1);
    end
    tInc        = tIncMan{iBlk};
    idxStart    = find(diff(tInc)==1)+1;
    idxEnd      = find(diff(tInc)==-1);
    extend      = round(5*fs);
    idxStart    = idxStart - extend;
    idxEnd      = idxEnd + extend;
    nTrials     = length(idxStart);
    tIncCh      = ones(length(t), size(dod,2));
    
    for iTrial = 1:nTrials
        dtrial      = dod(idxStart(iTrial):idxEnd(iTrial),:);
        ttrial      = t(idxStart(iTrial):idxEnd(iTrial),:);
        tIncCh(idxStart(iTrial):idxEnd(iTrial),:) = hmrR_tInc_baselineshift_Ch_Nirs(dtrial, ttrial); % finding the baseline shift motions
    end
    %%%%%%%%%%%%%%%%%%
    
    % extending signal for motion detection purpose (12 sec from each edge)
    extend = round(12*fs);
    
    tIncCh1 = repmat(tIncCh(1,:),extend,1);
    tIncCh2 = repmat(tIncCh(end,:),extend,1);
    tIncCh  = [tIncCh1;tIncCh;tIncCh2];
    
    d1 = repmat(dod(1,:),extend,1);
    d2 = repmat(dod(end,:),extend,1);
    dod = [d1;dod;d2];
    
    t2 = (0:(1/fs):(length(dod)/fs))';
    t2 = t2(1:length(dod),1);
    
    dodLP = hmrR_BandpassFilt_Nirs(dod, fs, 0, 2);
    
    %%%% Spline Interpolation
    dod = my_MotionCorrectSpline_Nirs(dodLP, t2, SD, tIncCh, p);
    dod = dod(extend+1:end-extend,:); % removing the extention
    tIncCh = tIncCh(extend+1:end-extend,:);
    
    %%%% Savitzky_Golay filter
    K = 3; % polynomial order
    FrameSize_sec = round(FrameSize_sec * fs);
    if mod(FrameSize_sec,2)==0
        FrameSize_sec = FrameSize_sec  + 1;
    end
    dod = sgolayfilt(dod,K,FrameSize_sec);

    dod(:,order) = dod(:,:);
    data_d(iBlk).SetDataTimeSeries(dod);
    
end

