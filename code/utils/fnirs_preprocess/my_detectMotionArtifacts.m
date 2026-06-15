function tIncCh = my_detectMotionArtifacts(signal, trials)
% 输入:
%   signal:  fNIRS信号矩阵
%   trials: 试次信息，cell数组，每个元素为 [startIdx, endIdx]，表示试次的起始和结束索引
% 输出:
%   tIncCh: 伪迹标记矩阵，维度 [Nchn, Nt]，1表示正常数据，0表示伪迹

    [Nchn, Nt] = size(signal);
    tIncCh = ones(Nchn, Nt); % 初始化输出矩阵，全为1（正常）
    
    % 为每个通道收集所有试次的峰谷幅值差
    all_amp_diffs = cell(Nchn, 1);
    
    % 第一步：收集所有通道的所有峰谷幅值差
    for ch = 1:Nchn
        ch_diffs = [];
        for tIdx = 1:length(trials)
            trial = trials{tIdx};
            startIdx = trial(1);
            endIdx = trial(2);
            segment = signal(ch, startIdx:endIdx); % 提取当前通道的试次数据
            
            % 寻找局部极大值（峰值）
            [~, locs_max] = findpeaks(segment);
            
            % 寻找局部极小值（谷值）
            [~, locs_min] = findpeaks(-segment);
            
            % 合并峰谷位置并排序
            all_locs = sort([locs_max, locs_min]);
            if isempty(all_locs) || numel(all_locs) < 2
                continue; % 无峰谷可配对，跳过
            end
            all_locs = [1 all_locs length(segment)];
            
            % 计算相邻峰谷的幅值差
            for i = 1:(length(all_locs) - 1)
                idx1 = all_locs(i);
                idx2 = all_locs(i+1);
                diff_val = abs(segment(idx1) - segment(idx2));
                ch_diffs = [ch_diffs, diff_val];
            end
        end
        all_amp_diffs{ch} = ch_diffs;
    end
    
    % 第二步：计算每个通道的全局统计量并标记伪迹
    for ch = 1:Nchn
        amp_diffs = all_amp_diffs{ch};
        
        if isempty(amp_diffs) || numel(amp_diffs) < 2
            continue; % 无有效峰谷差，跳过该通道
        end
        
        % 计算全局统计量
        mean_diff = mean(amp_diffs);
        std_diff = std(amp_diffs);
        threshold = mean_diff + 2.5 * std_diff; % 阈值：均值+2.5倍标准差
        
        % 第三步：再次遍历试次进行伪迹标记
        for tIdx = 1:length(trials)
            trial = trials{tIdx};
            startIdx = trial(1);
            endIdx = trial(2);
            segment = signal(ch, startIdx:endIdx);
            
            % 寻找峰谷位置
            [~, locs_max] = findpeaks(segment);
            [~, locs_min] = findpeaks(-segment);
            all_locs = sort([locs_max, locs_min]);
            
            if numel(all_locs) < 2
                continue; % 无峰谷可配对，跳过
            end
            all_locs = [1 all_locs length(segment)];
            
            % 标记异常峰谷区间
            for i = 1:(length(all_locs) - 1)
                idx1 = all_locs(i);
                idx2 = all_locs(i+1);
                diff_val = abs(segment(idx1) - segment(idx2));
                
                if diff_val > threshold
                    startArtifact = min(idx1, idx2);
                    endArtifact = max(idx1, idx2);
                    
                    % 转换为全局索引并标记伪迹
                    globalIndices = startIdx - 1 + (startArtifact:endArtifact);
                    tIncCh(ch, globalIndices) = 0;
                end
            end
        end
    end
end