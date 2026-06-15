function [yhpf] = highpassfilt( y, fs, hpf)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
% convert t to fs
% assume fs is a time vector if length>1
if length(fs)>1
    fs = 1/(fs(2)-fs(1));
end


% high pass filter
FilterType = 1;
FilterOrder = 5;
if FilterType==1 | FilterType==5
    [fb,fa] = MakeFilter(FilterType,FilterOrder,fs,hpf,'high');
elseif FilterType==4
%    [fb,fa] = MakeFilter(FilterType,FilterOrder,fs,hpf,'high',Filter_Rp,Filter_Rs);
else
%    [fb,fa] = MakeFilter(FilterType,FilterOrder,fs,hpf,'high',Filter_Rp);
end


yhpf=filtfilt(fb,fa,y);


end

