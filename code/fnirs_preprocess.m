% Preprocess the raw fNIRS recordings.
%   The raw .nirs files are stored under `~root/data/all_raw` folder. This
%   script should be under `~root/Code`, and its dependencies (Homer3,
%   FieldTrip, and some functions) under `~root/code/utils`.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Author: Ming Li <liming16@tsinghua.org.cn> %
% % Update: 2025/05/30                         %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Environments
%   - Win11
%   - MATLAB R2023b
%   - Homer3-1.80.2
%   - fieldtrip-20240110
%
% Preprocessing Pipeline
%   - For .nirs data files, utilize Homer3 package to:
%       Step0 - remove invalid channels (those located in the occipital
%               lobe) and manually include time points in segments of
%               interest, i.e., (-3s:end) of each trial.
%       Step1 - convert raw intensity data to delta optical density (DOD).
%       Step2 - recognize and correct artifacts using AEPO.
%       Step3 - perform band-pass filter.
%   - Then save DODs as .snirf data files, and utilize FieldTrip package to:
%       Step4 - segment continuous DOD into trials.
%       Step5 - correct the baseline for each trial.
%       Step6 - convert DODs into delta hemoglobin concentration (DC).
%
% Note
%   - To successfully save SNIRF objects in MacOS, the 'H5T_NATIVE_ULONG' in
%       line-98 of Homer3's 'hdf5write_safe.m' has been modified to
%       'H5T_NATIVE_INT' according to this post
%       (https://github.com/BUNPC/Homer3/issues/181).
%   - The logic of func 'hmrR_PruneChannels.m' is a mess, sometimes deleting
%       two wavelengths simultaneously for the same channel, and sometimes
%       deleting only one wavelength. Hence, I modified it to ensure that
%       if arbitrary wavelength fails the quality check then all
%       wavelengths from the same ScrDet pair will be ultimately pruned.
%       See the modification in hmrR_PruneChannels.m.
%   - Sometimes when filtering, homer3 may remind you that some parameters
%       are empty. It's a stupid but definitely safe bug, just click
%       "proceed anyway". Homer will create an ".error" file in your working
%       directory and not bother you next time.

% Set the environment and paths
rootPath  = fileparts(fileparts(mfilename('fullpath')));
codePath  = [rootPath,filesep,'code'];
rawPath   = [rootPath,filesep,'data',filesep,'all_raw'];
savePath  = [rootPath,filesep,'data',filesep,'fnirs_signals',filesep];
utilsPath = [codePath, filesep, 'utils'];
homerPath = fullfile(utilsPath, 'Homer3-1.80.2');
ftPath    = fullfile(utilsPath, 'fieldtrip-20240110');
if ~isfolder(homerPath)
    error(['Missing Homer3 at ', homerPath, newline, ...
        'Install v1.80.2 there and apply patches; see docs/install.md']);
end
if ~isfolder(ftPath)
    error(['Missing FieldTrip at ', ftPath, newline, ...
        'Install release 20240110 there; see docs/install.md']);
end
addpath(utilsPath);
addpath([utilsPath, filesep, 'fnirs_preprocess']);
addpath([utilsPath, filesep, 'fnirs_preprocess', filesep, 'AEPO']);
cd(homerPath);
setpaths;
cd(rootPath);
addpath(ftPath);
ft_defaults;
ft_hastoolbox('artinis',1);
ft_info off
ft_warning off

%% Determine the preprocessing steps and set parameters

% Step 0 - Invalid channels parameters
invalidSource   = {'S4','S7','S9','S12','S14'};
invalidDetector = {'D5','D8','D10'};

% Step 3 - Band-pass filtering parameters
doFilt      = 1; % enable filtering (recommended)

% Step 6 - Concentration types for conversion
concType    = {'O2Hb', 'HHb'};

% AEPO artifact detection and correction parameters
std_prop    = 0.3;
stdThresh   = 7;
ampThresh   = 0.5;
tMotion     = 0.5;
tMask       = 1;
savePath            = [savePath,'AEPO_'];
if doFilt % see hmrR_BandpassFilt.m
    filter_hpf              = 0.01; % min([(1/trialSecs) 0.01])
    filter_lpf              = 0.2; % typically 0.2 Hz or higher
    prefix1                 = num2str(filter_hpf);
    prefix1(prefix1=='.')   = [];
    prefix2                 = num2str(filter_lpf);
    prefix2(prefix2=='.')   = [];
    savePath                = [savePath,prefix1,'filt',prefix2,'_'];
end

%% Prepare the container for pre-processing report
while savePath(end)=='_'
    savePath = savePath(1:end-1);
end
if ~isfolder(savePath)
    mkdir(savePath)
end

summaryRow = {NaN,"",""};
prepSummary = cell2table( ...
    summaryRow, ...
    'VariableNames', ...
    {'subject','rawFile','state'});

prepQuality = cell2table( ...
    {NaN,"","",NaN,"",NaN,NaN,NaN,NaN,NaN,NaN,NaN}, ...
    'VariableNames', ...
    {'subject','experiment','task','trialNo','channel', ...
    'ratioMA','ratioSCI','ratioPeakPower','ratioBoth','SCI','PeakPower','corrHb'});

%% Preprocess and visualize

% Go through all subjects
nSubs   = 53;
nExp    = 3;
fileNo  = 0;
expTasks = {{'Perception'},{'Production'},{'Performance'}};
for sub = 1:nSubs
    fprintf(['\n---------- Processing subject ',num2str(sub), ...
        ' (total=',num2str(nSubs),') ----------\n'])
    inPath  = [rawPath,filesep,num2str(sub)];
    outPath = [savePath,filesep,num2str(sub)];
    if ~isfolder(outPath)
        mkdir(outPath);
    end

    % Prepare containers for visualized signals
    s10d9   = {};
    s19d16  = {};
    times   = {};
    tasks   = {};

    % Start by .nirs files
    nirsFiles   = dir([inPath, filesep,'*.nirs']);
    nirsFiles   = {nirsFiles.name};

    % Go through all experiments
    %   Every subject should participate 3 different experiments and each
    %   experiment may has one or multiple recording files (if subject took
    %   a break whithin this experiment)
    for expNo       = 1:nExp
        fileIdx     = contains(nirsFiles,['exp',num2str(expNo)]);
        nParts      = sum(fileIdx);
        outFile     = [outPath,filesep,'trialsDC_exp',num2str(expNo),'.mat'];

        % Check the existence of recording files
        if nParts == 0
            fprintf(['Found 0 .nirs recording files for [Experiment ', ...
                num2str(expNo), '].\n'])
            continue
        end
        fprintf(['Found ',num2str(nParts),' .nirs recording files for [Experiment ', ...
            num2str(expNo), '].\n'])

        %% Start to preprocess
        prepSummary = [prepSummary(1:fileNo,:);
            repmat(summaryRow,nParts,1)];
        prepSummary.subject((fileNo+1):(fileNo+nParts)) = sub;
        prepSummary.state((fileNo+1):(fileNo+nParts))   = 'unknown';
        if nParts == 1
            prepSummary.rawFile(fileNo+1) = "exp"+num2str(expNo)+".nirs";
        else
            prepSummary.rawFile((fileNo+1):(fileNo+nParts)) = "exp"+num2str(expNo)+"_"+string(1:nParts)+".nirs";
        end
        try
            % Homer3 preprocessing steps (Step 0-3)
            trialIdx = 1;
            for partNo = 1:nParts
                fileName = char(prepSummary.rawFile{fileNo+partNo});

                % ------ Step 0 ------
                % Load data and remove invalid channels
                %   These channels are located in the occipital lobe, which
                %   were not included in our experiments but recorded
                %   simultaneously by the device.
                snirf = SnirfClass(load([inPath,filesep,fileName],'-mat'));
                snirf = my_removechannels(snirf, ...
                    invalidSource, ...
                    invalidDetector);
                mlActAuto = {snirf.data(1).GetMeasList()};
                % For some of following steps, consider only the time
                %   points in segments of interest, i.e., (-3s:end) of each
                %   trials.
                cfg                     = [];
                cfg.dataset             = [inPath,filesep,fileName];
                cfg.trialfun            = ['my_trialfun_exp',num2str(expNo)];
                cfg.trialdef.pre        = 3;
                cfg.trialdef.post       = 0;
                cfg.trialdef.inpath     = inPath;
                cfg.trialdef.startidx   = trialIdx;
                cfg.trialdef.prefix     = 's'; % for .nirs file
                cfg.showcallinfo        = 'no';
                cfg                     = ft_definetrial(cfg);
                trialIdx                 = trialIdx + height(cfg.trl);
                % Indicate the manually included time points
                tIncMan = {zeros(length(snirf.data.time),1)};
                for j = 1:height(cfg.trl)
                    tIncMan{1}(cfg.trl.begsample(j):cfg.trl.endsample(j)) = 1;
                end

                % ------ Step 1 ------
                % Convert raw light intensity to optical density change
                %   Note: the OD change is relative to the mean of
                %   whole data time series.
                dataDOD = hmrR_Intensity2OD(snirf.data);

                % ------ Step 2 ------
                % Recognize and correct artifacts using AEPO
                [dataDOD, tIncCh] = my_MotionCorrect_AEPO(dataDOD, ...
                    [], [], [], std_prop, ...
                    tMotion, tMask, stdThresh, ampThresh);

                % ------ Check Point ------
                [tempTbl, ratioMA, ratioSCI, ratioPeakPower, ratioBoth, sciVal, peakPowerVal] = qualitycheck( ...
                    dataDOD, tIncCh, mlActAuto, snirf.probe, cfg, sub, expNo, trialIdx);
                prepQuality = [prepQuality;tempTbl];
                prepQuality.ratioMA((end-height(tempTbl)+1):end) = ratioMA;
                prepQuality.ratioSCI((end-height(tempTbl)+1):end) = ratioSCI;
                prepQuality.ratioPeakPower((end-height(tempTbl)+1):end) = ratioPeakPower;
                prepQuality.ratioBoth((end-height(tempTbl)+1):end) = ratioBoth;
                prepQuality.SCI((end-height(tempTbl)+1):end) = sciVal;
                prepQuality.PeakPower((end-height(tempTbl)+1):end) = peakPowerVal;

                % ------ Step 3 ------
                % Perform band-pass filter
                dataDOD = hmrR_BandpassFilt(dataDOD, filter_hpf, filter_lpf);

                % Replace the pruned channels with NaNs
                dataDOD.dataTimeSeries(:,mlActAuto{1}(:,3)==0) = NaN;

                % % Replace the MA with NaNs %%%%%%%%%%%%%%%%%%%%
                % dataDOD.dataTimeSeries(~tIncCh) = NaN;

                % Save the preprocessed optical density data
                snirf.data = dataDOD;
                snirf.Save([outPath,filesep,'continuousDOD_',fileName(1:end-5),'.snirf'])

            end % Loop over recording files

            % FieldTrip preprocessing steps (Step 4-6)
            %   Note: FieldTrip's 'channel' concept is different
            %   from above. For fNIRS data it means source-detector pair,
            %   however, for FieldTrip it means single data time series
            %   (a specific wavelength or haemoglobin at one scr-det
            %   pair). Please don't misunderstand it.
            mergeData   = cell(nParts,1);
            trialIdx = 1;
            for partNo = 1:nParts
                fileName = char(prepSummary.rawFile{fileNo+partNo});
                fileName = fileName(1:end-5);

                % ------ Step 4 ------
                % Load continuous OD data and segment it into trials
                cfg                     = [];
                cfg.dataset             = [outPath,filesep,'continuousDOD_',fileName,'.snirf'];
                cfg.trialfun            = ['my_trialfun_exp',num2str(expNo)];
                cfg.trialdef.pre        = 3; % fixation time
                cfg.trialdef.post       = 0;
                cfg.trialdef.inpath     = inPath;
                cfg.trialdef.startidx   = trialIdx;
                cfg.trialdef.prefix     = ''; % for .snirf file
                cfg.showcallinfo        = 'no';
                cfg                     = ft_definetrial(cfg);
                cfg.target              = concType;
                cfg.continuous          = 'yes';
                cfg.channel             = 'nirs';
                cfg.showcallinfo        = 'no';
                partDOD                 = ft_preprocessing(cfg);
                trialIdx                 = trialIdx + height(cfg.trl);

                % ------ Step 5 ------
                % Remember that the transformed OD change is relative
                %   to the mean of whole continuous recording, and one
                %   experiment may have multiple recordings with
                %   different mean values. So the baseline must be
                %   corrected (here, to the pre-trial fixation phase)
                %   before merging trials from different recordings.
                for i = 1:length(partDOD.trial)
                    trialPreIdx     = partDOD.time{i}<0;
                    trialPreMean    = mean(partDOD.trial{i}(:,trialPreIdx),2);
                    partDOD.trial{i} = partDOD.trial{i}-trialPreMean;
                end

                % Save for further combination
                mergeData{partNo}   = partDOD;

            end % Loop over recording files

            % Combine trials from different recording files
            cfg                 = [];
            cfg.keepsampleinfo  = 'no';
            cfg.showcallinfo    = 'no';
            trialsDOD           = ft_appenddata(cfg,mergeData{:});
            assert(length(trialsDOD.trial) == height(trialsDOD.trialinfo), ...
                'The number of trials in the combined data does not match the number of trials in the trialinfo table.');

            % ----- Step 6 -----
            % Convert optical density changes to haemoglobin
            %   concentration changes.
            cfg         = [];
            cfg.target  = concType;
            cfg.channel = 'nirs';
            cfg.dpf     = 6;
            trialsDC    = ft_nirs_transform_ODs(cfg, trialsDOD);

            % ------ Check Point ------
            % Final quality evaluation by testing the negative
            %   correlation between O2Hb and HHb concentration changes.
            tempTbl = prepQuality( ...
                prepQuality.subject==sub & strcmp(prepQuality.experiment,['exp',num2str(expNo)]), ...
                {'trialNo','channel'});
            corrHb = hbcorr(trialsDC, tempTbl);
            prepQuality{ ...
                prepQuality.subject==sub & strcmp(prepQuality.experiment,['exp',num2str(expNo)]), ...
                "corrHb"}  = corrHb;

            % Save fNIRS trials to disk with both variables for cross-language compatibility
            trialinfo = table2cell(trialsDC.trialinfo);
            save(outFile,'trialsDC','trialinfo','-v7');

            % Update visualized signals
            cur_task = expTasks{expNo};
            [fs, s10d9, s19d16, times, tasks] = update_plotdata(trialsDC, ...
                s10d9, s19d16, times, tasks, cur_task);

            % Update preprocessing summary
            prepSummary.state((fileNo+1):(fileNo+nParts)) = 'success';
            fileNo = fileNo+nParts;

        catch ME
            errorReport = getReport(ME);
            errorFile   = [outPath,filesep,'exp',num2str(expNo),'_ERROR.txt'];
            fileID = fopen(errorFile,'w+');
            fprintf(fileID, errorReport);
            fclose(fileID);
            prepSummary.state((fileNo+1):(fileNo+nParts)) = 'ERROR';
            fileNo = fileNo+nParts;
        end

    end % Loop over experiments

    % Visualize the preprocessed signals in two specific channels
    plot_channels(s10d9, s19d16, times, tasks, fs, outPath)

end % Loop over subjects

%% Save the preprocessing summary
prepQuality = prepQuality(~isnan(prepQuality.corrHb),:);
writetable(prepQuality,[savePath,filesep,'quality.csv']);
writetable(prepSummary,[savePath,filesep,'summary.csv']);

%% Custom functions

function corrHb = hbcorr(trialsDC, tempTbl)
    corrHb = nan(height(tempTbl), 1);
    for rowNo = 1:height(tempTbl)
        trialNo = tempTbl.trialNo(rowNo);
        chnLabel = tempTbl.channel(rowNo);
        idxPair = contains(trialsDC.label, chnLabel + " ");
        tempCorr = corr(trialsDC.trial{trialNo}(idxPair, :)', 'rows', 'complete');
        corrHb(rowNo) = tempCorr(1, 2);
    end
end

function [tbl, ratioMA, ratioSCI, ratioPeakPower, ratioBoth, SCI, PeakPower] = qualitycheck(dataDOD, tIncCh, mlActAuto, probe, cfg, sub, expNo, trialIdx)
    data = hmrR_BandpassFilt(dataDOD, 0.5, 2.5);
    d = data(1).GetDataTimeSeries();
    t = data(1).GetTime();
    fs = 1 / (t(2) - t(1));

    nTrials = height(cfg.trl);
    nPairs = size(mlActAuto{1, 1}, 1) / 2;
    nTotal = nTrials * nPairs;

    % Ratio-style metrics (legacy; derived from 3-s windows with thresholds)
    ratioMA = nan(nTotal, 1);
    ratioSCI = nan(nTotal, 1);
    ratioPeakPower = nan(nTotal, 1);
    ratioBoth = nan(nTotal, 1);

    % Scalar metrics (new; one value per trial × channel from full time series)
    SCI = nan(nTotal, 1);
    PeakPower = nan(nTotal, 1);

    winLength = 3 * fs;
    trialWins = floor((cfg.trl.endsample - cfg.trl.begsample) / winLength);

    for trialNo = 1:nTrials
        trialStart = cfg.trl.begsample(trialNo);
        trialEnd = cfg.trl.endsample(trialNo);

        for pairNo = 1:nPairs
            idx = (trialNo - 1) * nPairs + pairNo;

            if mlActAuto{1}(pairNo, 3) == 0
                continue
            end

            ratioMA(idx) = 1 - mean(...
                tIncCh(trialStart:trialEnd, pairNo) & ...
                tIncCh(trialStart:trialEnd, pairNo + nPairs));

            x = d(trialStart:trialEnd, pairNo);
            y = d(trialStart:trialEnd, pairNo + nPairs);

            % Full-epoch scalar SCI and PeakPower
            xz = zscore(x);
            yz = zscore(y);
            if all(isfinite(xz)) && all(isfinite(yz)) && numel(xz) >= 3
                SCI(idx) = corr(xz, yz, 'rows', 'complete');
            end
            if all(isfinite(xz)) && all(isfinite(yz)) && numel(xz) >= 3
                [xyFull, lagsFull] = xcorr(xz, yz, 'unbiased');
                sciLag0 = xyFull(lagsFull == 0);
                if ~isempty(sciLag0)
                    SCI(idx) = sciLag0;
                end
                [pxyFull, ~] = periodogram(xyFull, hamming(length(xyFull)), length(xyFull), fs, 'power');
                PeakPower(idx) = max(pxyFull);
            end

            nWins = trialWins(trialNo);
            sci = zeros(nWins, 1);
            pp = zeros(nWins, 1);

            for winNo = 1:nWins
                begpt = (winNo - 1) * winLength + 1;
                endpt = winNo * winLength;

                [xy, lags] = xcorr(zscore(x(begpt:endpt)), zscore(y(begpt:endpt)), 'unbiased');
                sci(winNo) = xy(lags == 0);
                [pxy, ~] = periodogram(xy, hamming(length(xy)), length(xy), fs, 'power');
                pp(winNo) = max(pxy);
            end

            ratioSCI(idx) = mean(sci > 0.8);
            ratioPeakPower(idx) = mean(pp > 0.1);
            ratioBoth(idx) = mean(sci > 0.8 & pp > 0.1);
        end
    end

    subject = repmat(sub, nTotal, 1);
    experiment = repmat({['exp' num2str(expNo)]}, nTotal, 1);
    task = repelem(cfg.trl.task, nPairs, 1);
    trialNo = repelem(((trialIdx - nTrials):(trialIdx - 1))', nPairs, 1);
    channel = repmat(...
        join([probe.sourceLabels(mlActAuto{1}(1:nPairs, 1))' ...
              probe.detectorLabels(mlActAuto{1}(1:nPairs, 2))'], '-'), ...
        nTrials, 1);

    corrHb = nan(nTotal, 1);

    tbl = table(subject, experiment, task, trialNo, channel, ...
                ratioMA, ratioSCI, ratioPeakPower, ratioBoth, SCI, PeakPower, corrHb);
end

function [fs, s10d9, s19d16, times, tasks] = update_plotdata(input, s10d9, s19d16, times, tasks, cur_task)
    if ischar(input)
        loadedData = load(input, 'trialsDC');
        trialsDC = loadedData.trialsDC;
    elseif isstruct(input)
        trialsDC = input;
    end

    fs = trialsDC.fsample;
    tasks = [tasks, cur_task];

    for t = 1:length(cur_task)
        task = cur_task{t};
        trialIdx = find(strcmp(trialsDC.trialinfo.task, task));

        tTrial = 0;
        for j = 1:length(trialIdx)
            tTrial = [tTrial, tTrial(end) + size(trialsDC.trial{trialIdx(j)}, 2) / fs];
        end
        tTrial = tTrial(2:end);
        times = [times, {tTrial}];

        signal = cell2mat(trialsDC.trial(trialIdx));
        s10d9 = [s10d9, {signal(37:38, :)}];
        s19d16 = [s19d16, {signal(79:80, :)}];
    end
end

function plot_channels(s10d9, s19d16, times, tasks, fs, outPath)
    f = figure('Visible', 'off');
    for t = 1:length(tasks)
        subplot(length(tasks), 1, t);
        timeVec = (1:size(s10d9{t}, 2)) / fs;
        plot(timeVec, s10d9{t}(1, :), 'r-', timeVec, s10d9{t}(2, :), 'b-');

        if ~isempty(times{t})
            yLimits = [min(s10d9{t}(:)), max(s10d9{t}(:))];
            for i = 1:length(times{t})
                line([times{t}(i), times{t}(i)], yLimits, 'LineStyle', '--', 'Color', 'k');
            end
        end
        title(tasks{t});
    end
    print(f, fullfile(outPath, 'S10_D9_signals'), '-dpng');
    close(f);

    f = figure('Visible', 'off');
    for t = 1:length(tasks)
        subplot(length(tasks), 1, t);
        timeVec = (1:size(s19d16{t}, 2)) / fs;
        plot(timeVec, s19d16{t}(1, :), 'r-', timeVec, s19d16{t}(2, :), 'b-');

        if ~isempty(times{t})
            yLimits = [min(s19d16{t}(:)), max(s19d16{t}(:))];
            for i = 1:length(times{t})
                line([times{t}(i), times{t}(i)], yLimits, 'LineStyle', '--', 'Color', 'k');
            end
        end
        title(tasks{t});
    end
    print(f, fullfile(outPath, 'S19_D16_signals'), '-dpng');
    close(f);
end
