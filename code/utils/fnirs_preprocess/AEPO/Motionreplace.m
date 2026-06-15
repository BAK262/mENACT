function dodSpline = Motionreplace(dod, t, SD, tIncCh,st )

% % 如果p超出其授权范围，则设置为0.99
% if p>1 || p<0
%     display('Parameter has to be between 0 and 1. Returning with no correction');
%     dodSpline = dod;
%     return;
% end

fs = 1/mean(t(2:end)-t(1:end-1));%计算频率

% 窗口宽度限制，用于计算段偏移的平均值
dtShort = 0.3;  % seconds
dtLong  = 3;    % seconds

ml = SD.MeasList;
SD.MeasListAct=ones(size(SD.MeasList,1),1);%BY_SHEN
mlAct = SD.MeasListAct; % prune bad channels

lstAct = find(mlAct==1);
dodSpline = dod;

for ii = 1:length(lstAct)   %从第一个通道到最后一个
    
    idx_ch = lstAct(ii);
    
    lstMA = find(tIncCh(:,idx_ch)==0);   %找到尾迹所在位置 tIncCh在通道idx_ch中为0的位置，即尾迹位置。
    
    if ~isempty(lstMA)   %如果不是空集， 存在尾迹
        
        % Find indexes of starts and ends of MA segments   找到尾迹起始位置
        lstMs = find(diff(tIncCh(:,idx_ch))==-1);   % starting indexes of mvt segments  尾迹起的位置
        lstMf = find(diff(tIncCh(:,idx_ch))==1);    % ending indexes of mvt segments  尾迹结束的位置
         
        % Case where there's a single MA segment, that either starts at the
        % beginning or ends at the end of the total time duration
        if isempty(lstMf)       %如果结束的位置是空集
            lstMf = size(tIncCh,1);  %则结束的位置是最后一个点
        end
        if isempty(lstMs)  %如果开始的位置是空集
            lstMs = 1;   %则开始位置是第一个点
        end
        % If any MA segment either starts at the beginning or
        % ends at the end of the total time duration
        if lstMs(1)>lstMf(1)    %如果第一个起始的时间大于第一个结束的时间（正常应该第一个起始时间小于结束时间）
            lstMs = [1;lstMs];  %则在起始时间加一个数值，数值设置为1
        end
        if lstMs(end)>lstMf(end)    %如果其实位置的最后一个值，大于结束为止的最后一个值(正常应该小于)
            lstMf(end+1,1) = size(tIncCh,1);  %则结束的位置加一个点，点的值为数据长
        end
        
        lstMl = lstMf-lstMs;    % lengths of MA segments   计算尾迹长度 
        nbMA = length(lstMl);   % number of MA segments    尾迹段的个数
        
        % Do the spline interpolation on each MA segment
        % only include channels in the active meas list
        stdE=st(1,ii);
        for jj = 1:nbMA          %从第一段尾迹到最后一段尾迹
            lst = lstMs(jj):(lstMf(jj)-1);   %从这段开始第一个点到结束为止的点
            ff=length(lst);
            % spline interp   样条差值
            
            randX=stdE.*randn(ff,1);
            %             SplInterp = csaps(t(lst)', dod(lst,idx_ch)', p, t(lst)')';
            % corrected signal = original signal - spline interpolation  校正信号=原始信号-样条插值
            %             dodSpline(lst,idx_ch) = dod(lst,idx_ch) - SplInterp';
            dodSpline(lst,idx_ch) = randX;
        end
        
        
        % Reconstruction of the whole time series (shift each segment)  重建整个时间序列（移动每个段）
        
        %% First MA segment: shift to the previous noMA segment if it exists,   从第一个尾迹段开始，  如果存在，移动到没有尾迹段的位置
        % to the next noMA segment otherwise   否则进入下一个没有尾迹的位置
        lst = (lstMs(1)):(lstMf(1)-1);     %从第一段尾迹开始到尾迹结束
        SegCurrLength = lstMl(1);     %第一段尾迹长度
        if SegCurrLength < dtShort*fs    %如果尾迹长度小于0.3s内采集点的个数
            windCurr = SegCurrLength;     %设置长度为尾迹长度
        elseif SegCurrLength < dtLong*fs   %如果尾迹长度小于3s内采集的点的个数
            windCurr = floor(dtShort*fs);   %设置的窗口长度为3s内才到的数值小1 
        else
            windCurr = floor(SegCurrLength/10);  %否则为尾迹长度除10  
        end
        
        if lstMs(1)>1        %如果第一个尾迹点是在第一次采集点之后  
            SegPrevLength = length(1:(lstMs(1)-1));   %计算第一段前没有尾迹的长度  
            if SegPrevLength < dtShort*fs  %如果正常数据段小于0.3s 
                windPrev = SegPrevLength;    %窗口为非尾迹数据段长度
            elseif SegPrevLength < dtLong*fs   %如果正常数据段小于3s 
                windPrev = floor(dtShort*fs);%设置的窗口长度为3s内才到的数值小1 
            else
                windPrev = floor(SegPrevLength/10); %否则为非尾迹长度除10  
            end
            meanPrev = mean(dodSpline(lst(1)-windPrev:(lst(1)-1), idx_ch));  %计算非尾迹段平均值 
            meanCurr = mean(dodSpline(lst(1):(lst(1)+windCurr-1), idx_ch));  %计算尾迹段平均值 
            dodSpline(lst,idx_ch) = dodSpline(lst,idx_ch) - meanCurr + meanPrev;  %在校正信号的基础上移动整个数据段
            
        else
            if length(lstMs)>1    %如果开始的时间点的个数大于1 
                SegNextLength = length(lstMf(1):(lstMs(2)));  %则第二段的长度是有开始的第一个点到第二个点
            else 
                SegNextLength = length(lstMf(1):size(tIncCh,1));  %否则这个长度从第一个点到最后
            end
            if SegNextLength < dtShort*fs   %如果2段之间的数据段小于0.3s
                windNext = SegNextLength;   %窗口为2段尾迹数据段开始之间长度
            elseif SegNextLength < dtLong*fs  %如果2个尾迹数据段之间小于3s 
                windNext = floor(dtShort*fs);  %设置的窗口长度为3s内才到的数值小1 
            else
                windNext = floor(SegNextLength/10);  %否则为非尾迹长度除10  
            end
            meanCurr = mean(dodSpline((lst(end)-windCurr):(lst(end)-1),  idx_ch));  %计算非尾迹段平均值 
            meanNext = mean(dodSpline((lst(end)+1):(lst(end)+windNext), idx_ch));  %计算尾迹段平均值 
            dodSpline(lst,idx_ch) = dodSpline(lst,idx_ch) - meanCurr + meanNext;  %在校正信号的基础上移动整个数据段
        end
        
        
        %% Intermediate segments   中间的尾迹段
        for kk=1:(nbMA-1)   %从1到倒数第2段的尾迹段
            % no motion  没有尾迹的位置
            lst = lstMf(kk):(lstMs(kk+1)-1);    %上一个尾迹结束的点到下一段尾迹开始的点减一
            SegPrevLength = lstMl(kk);   %第KK段尾迹长度
            SegCurrLength = length(lst);  %前一段非尾迹长度
            if SegPrevLength < dtShort*fs  
                windPrev = SegPrevLength;
            elseif SegPrevLength < dtLong*fs
                windPrev = floor(dtShort*fs);
            else
                windPrev = floor(SegPrevLength/10);
            end
            if SegCurrLength < dtShort*fs
                windCurr = SegCurrLength;
            elseif SegCurrLength < dtLong*fs
                windCurr = floor(dtShort*fs);
            else
                windCurr = floor(SegCurrLength/10);
            end
            meanPrev = mean(dodSpline((lst(1)-windPrev):(lst(1)-1), idx_ch));
            meanCurr = mean(dod(lst(1):(lst(1)+windCurr-1), idx_ch));
            
            dodSpline(lst,idx_ch) = dod(lst,idx_ch) - meanCurr + meanPrev;
            
            % motion
            lst = (lstMs(kk+1)):(lstMf(kk+1)-1);
            SegPrevLength = SegCurrLength;
            SegCurrLength = lstMl(kk+1);
            if SegPrevLength < dtShort*fs
                windPrev = SegPrevLength;
            elseif SegPrevLength < dtLong*fs
                windPrev = floor(dtShort*fs);
            else
                windPrev = floor(SegPrevLength/10);
            end
            if SegCurrLength < dtShort*fs
                windCurr = SegCurrLength;
            elseif SegCurrLength < dtLong*fs
                windCurr = floor(dtShort*fs);
            else
                windCurr = floor(SegCurrLength/10);
            end
            meanPrev = mean(dodSpline((lst(1)-windPrev):(lst(1)-1), idx_ch));
            meanCurr = mean(dodSpline(lst(1):(lst(1)+windCurr-1), idx_ch));
            
            dodSpline(lst,idx_ch) = dodSpline(lst,idx_ch) - meanCurr + meanPrev;
        end
        
        %% Last not MA segment
        if lstMf(end)<length(t)
            lst = (lstMf(end)-1):length(t);
            SegPrevLength = lstMl(end);
            SegCurrLength = length(lst);
            if SegPrevLength < dtShort*fs
                windPrev = SegPrevLength;
            elseif SegPrevLength < dtLong*fs
                windPrev = floor(dtShort*fs);
            else
                windPrev = floor(SegPrevLength/10);
            end
            if SegCurrLength < dtShort*fs
                windCurr = SegCurrLength;
            elseif SegCurrLength < dtLong*fs
                windCurr = floor(dtShort*fs);
            else
                windCurr = floor(SegCurrLength/10);
            end
            meanPrev = mean(dodSpline((lst(1)-windPrev):(lst(1)-1), idx_ch));
            meanCurr = mean(dod(lst(1):(lst(1)+windCurr-1), idx_ch));
            
            dodSpline(lst,idx_ch) = dod(lst,idx_ch) - meanCurr + meanPrev;
        end
    
    %else
     %   dodSpline(:,i_ch) = dod;
    end
end











