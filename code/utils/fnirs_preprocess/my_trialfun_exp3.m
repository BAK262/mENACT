function trl = my_trialfun_exp3(cfg)
% This function requires the following fields to be specified
%   cfg.dataset
%   cfg.trialdef.pre
%   cfg.trialdef.post
%   cfg.trialdef.startidx (optional)

% read the header information and the events from the data
hdr   = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset);

% read the trial information from the data
folder = cfg.trialdef.inpath;
files = dir([folder,'/','*_rating.csv']);
rating_file_idx = contains({files.name},'exp3');
if sum(rating_file_idx) == 1
    rating_file = files(rating_file_idx).name;
elseif sum(rating_file_idx) > 1
    error('Detected more than one rating file for [Experiment 3].')
elseif sum(rating_file_idx) == 0
    error('Detected no rating file for [Experiment 3].')
end
ratings = readtable([folder,'/',rating_file]);
ratings = ratings(ratings.actCompletedPerc > 0, :);

% determine the number of samples before and after the trigger
pretrial = round(cfg.trialdef.pre * hdr.Fs);
posttrial = round(cfg.trialdef.post * hdr.Fs);

% determine which session this recording started at
if isfield(cfg.trialdef,'startidx')
    if isnumeric(cfg.trialdef.startidx) && length(cfg.trialdef.startidx)==1
        session_start_idx = cfg.trialdef.startidx;
    else
        error('The field ''begintrial'' of ''cfg'' can only accept one number.')
    end
else
    session_start_idx = 1;
end

% if this recording started from the first session, check the trigger of
% 'experiment start'.
if session_start_idx == 1 && ~any(contains({event.type},[cfg.trialdef.prefix,'48']))
    warning('Trigger of ''Experiment Start'' no found in this recording.')
end

% go through all triggers and record valid trials
trial = [];
task = {};
in_trial = false;
for i=1:length(event)
    if strcmp(event(i).type, [cfg.trialdef.prefix,'55']) % Performance trial start (trigger 55)
        begsample       = event(i).sample - pretrial; % start of segment
        offset          = -pretrial; % how many samples before trial
        cur_task        = 'Performance';
        in_trial        = true;
    elseif in_trial
        % normal end of trial (Performance end trigger)
        if any( ...
            strcmp( ...
            event(i).type, ...
            { ...
            [cfg.trialdef.prefix,'50'], ...
            [cfg.trialdef.prefix,'56'] ...
            } ...
            ) ...
            )% Performance end
            endsample       = event(i).sample + posttrial; % end of segment
            trial(end+1,:)  = [begsample, endsample, offset];
            task{end+1,1}   = cur_task;
            in_trial        = false;
        else
            % abnormal end of trial
            % e.g., the program was terminated before the trigger [acting] end was detected.
            in_trial        = false;
        end
    end
end
n_sessions = size(trial,1);
session_end_idx = session_start_idx + n_sessions -1;

% if this recording ended with the final session, check the trigger of
% 'experiment end'.
if session_end_idx > height(ratings)
    error(['Found ' num2str(session_end_idx) ' valid trials in fNIRS ' ...
        'recordings, while only ' num2str(height(ratings)) ' trials have ratings.'])
elseif session_end_idx == height(ratings)
    if ~any(contains({event.type},[cfg.trialdef.prefix,'49']))
        warning('Trigger of ''Experiment End'' no found in those recordings.')
    end
end

% save additional trial informations
trl = ratings(session_start_idx:session_end_idx,:);
% trl = renamevars(trl,"actCompletedPerc","percCompleted");
trl.Properties.VariableNames{'actCompletedPerc'} = 'percCompleted';
ori_cols = width(trl);
trl.begsample   = trial(:,1);
trl.endsample   = trial(:,2);
trl.offset      = trial(:,3);
trl.task        = task;
trl = trl(:,[(ori_cols+1):(ori_cols+4) 1:ori_cols]);

end
