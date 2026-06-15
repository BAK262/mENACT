% SYNTAX:
% data_d = my_MotionCorrect(data_d, tIncMan)
%
% DESCRIPTION:

function data_d = my_MotionCorrect(data_d, mlActAuto, tIncMan, p)

if isempty(tIncMan)
    tIncMan = cell(length(data_d),1);
end

for iBlk = 1:length(data_d)

    [dod, t, SD.MeasList, order] = data_d(iBlk).GetDataTimeSeries('matrix : reshape : wavelength');
    dod = dod(:,:);

    mlActAuto{iBlk} = mlAct_Initialize(mlActAuto{iBlk}, SD.MeasList);
    SD.MeasListAct  = mlAct_Matrix2BinaryVector(mlActAuto{iBlk}, SD.MeasList);
    
    % Low-pass filter at 2Hz
    fs = abs(1/(t(2)-t(1)));
    dodLP = hmrR_BandpassFilt_Nirs(dod, fs, 0, 2);

    % Detect artifacts by outliers in peak-peak amplitudes
    tInc        = tIncMan{iBlk};
    idxStart    = find(diff(tInc)==1)+1;
    idxEnd      = find(diff(tInc)==-1);
    extend      = round(5*fs);
    idxStart    = idxStart - extend;
    idxEnd      = idxEnd + extend;
    nTrials     = length(idxEnd);
    trials      = mat2cell([idxStart idxEnd],ones(nTrials,1),2);
    tIncCh      = my_detectMotionArtifacts(dodLP', trials);
    tIncCh      = tIncCh';
    %%%% Spline Interpolation
    t2 = (0:(1/fs):(length(dod)/fs))';
    t2 = t2(1:length(dod),1);
    
    dod = my_MotionCorrectSpline_Nirs(dodLP, t2, SD, tIncCh, p);

    dod(:,order) = dod(:,:);
    data_d(iBlk).SetDataTimeSeries(dod);

end



end