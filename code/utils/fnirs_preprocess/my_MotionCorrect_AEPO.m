function [dataDOD, tIncCh] = my_MotionCorrect_AEPO(dataDOD, ...
    hpf, timwid, timint, mm, ...
    tMotion, tMask, std_thresh, amp_thresh)

if isempty(hpf)
    hpf = 0.75;
end
if isempty(timwid)
    timwid = 4;
end
if isempty(timint)
    timint = 2;
end
if isempty(mm)
    mm = 0.3;
end
if isempty(tMotion)
    tMotion = 0.5;
end
if isempty(tMask)
    tMask = 1;
end
if isempty(std_thresh)
    std_thresh = 7; %15
end
if isempty(amp_thresh)
    amp_thresh = 0.5;
end

for iBlk = 1:length(dataDOD)
    dr = dataDOD(iBlk).GetDataTimeSeries();
    t = dataDOD(iBlk).GetTime();
    SD.MeasList = dataDOD(iBlk).GetMeasList();
    fs = 1/(t(2)-t(1));
    tIncMan = {ones(size(dr,1),1)};
    
    % Add one data point
    dodr = Addonedata(dr);
    
    % Calculate adaptive std
    nnn = highpassfilt(dr, fs, hpf);
    st = findsmallstd(nnn, fs, timwid, timint, mm);
    
    % Recognize and correct artifacts -- Spline
    [tInc, tIncCh1] = textmotion(dodr, fs, SD, tIncMan, tMotion, tMask, st, std_thresh, amp_thresh);
    dodS = MotionCorrectSpline(dodr, t, SD, tIncCh1, 0.99);
    
    % Recognize and correct artifacts -- Replace
    [tInc, tIncCh2] = textmotion(dodS, fs, SD, tIncMan, tMotion, tMask, st, std_thresh, amp_thresh);
    dodS1 = Motionreplace(dodS, t, SD, tIncCh2, st);
    
    % Delete one data point
    dodS2 = deleteonedata(dodS1);
    dataDOD(iBlk).SetDataTimeSeries(dodS2);

    tIncCh = tIncCh1 & tIncCh2;
    tIncCh = tIncCh(1:end-1,:);
end