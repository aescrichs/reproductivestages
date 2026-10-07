% =========================================================================
% Ignition_SingleSubject.m
% Computes node-metastability (intrinsic ignition) for each subject
%
% Input:  timeseries_infosubj.mat — cell array with NregionsxTimepoints
%         per subject (HCP-A, 100 Schaefer parcels)
% Output: Ignition100_HCP_4runs.mat — stdevokedintegS, mevokedintegS,
%         varevokedintegS (NodesxSubjects matrices)
%
% =========================================================================

clear;

Allcell    = load('timeseries_infosubj.mat', 'ts_HCP');
timeSeries = Allcell.ts_HCP;

% =========================================================================
% Parameters
% =========================================================================
TR   = 0.8;   % Repetition time (seconds)
NSUB = 725;   % Total subjects
N    = 100;   % Total nodes (Schaefer 100-parcel parcellation)
nTRs = 5;     % TRs to compute ignition after spontaneous events

% Bandpass filter (0.01-0.1 Hz), 2nd order Butterworth
flp  = 0.01;
fhi  = 0.1;
delt = TR;
k    = 2;
fnq  = 1 / (2 * delt);
Wn   = [flp/fnq fhi/fnq];
[bfilt2, afilt2] = butter(k, Wn);

% =========================================================================
% Main loop — compute ignition for each subject
% =========================================================================
for nsub = 1:NSUB
    fprintf('Processing subject %d / %d\n', nsub, NSUB);

    % Some subjects have fewer timepoints due to incomplete runs (Tmax=956).
    % Use actual length for each subject to handle this automatically.
    Tmax = size(timeSeries{nsub}, 2);

    xs = timeSeries{nsub}(:, 1:Tmax);
    T  = 1:Tmax;

    clear x timeseriedata events IntegStim2 nevents2

    % -----------------------------------------------------------------
    % Detect spontaneous events for each node
    % -----------------------------------------------------------------
    for seed = 1:N
        x = demean(detrend(xs(seed, :)));
        timeseriedata(seed, :) = filtfilt(bfilt2, afilt2, x);
        tise  = detrend(demean(timeseriedata(seed, T)));
        ev1   = tise > (std(tise) + mean(tise));
        ev2   = [0 ev1(1:end-1)];
        events(seed, :) = (ev1 - ev2) > 0;
    end
    SubjEvents{nsub} = events;

    % -----------------------------------------------------------------
    % Compute global integration at each time point
    % -----------------------------------------------------------------
    for t = T
        phasematrix = zeros(N, N);
        for i = 1:N
            for j = 1:N
                phasematrix(i, j) = events(i, t) * events(j, t);
            end
        end
        cc = phasematrix - eye(N);
        [~, csize] = get_components(cc);
        integ(t)   = max(csize) / N;
    end

    % -----------------------------------------------------------------
    % Event-triggered integration (nTRs window per node)
    % -----------------------------------------------------------------
    nevents2   = zeros(1, N);
    IntegStim2 = zeros(N, nTRs, Tmax);

    for seed = 1:N
        flag = 0;
        for t = T
            if events(seed, t) == 1 && flag == 0
                flag = 1;
                nevents2(seed) = nevents2(seed) + 1;
            end
            if flag > 0
                IntegStim2(seed, flag, nevents2(seed)) = integ(t);
                flag = flag + 1;
            end
            if flag == nTRs
                flag = 0;
            end
        end
    end

    % -----------------------------------------------------------------
    % Node-metastability: std of max ignition across events
    % -----------------------------------------------------------------
    for seed = 1:N
        ev_max = max(squeeze(IntegStim2(seed, :, 1:nevents2(seed))));
        mevokedinteg2(seed)   = mean(ev_max);
        stdevokedinteg2(seed) = std(ev_max);
        varevokedinteg(seed)  = var(ev_max);
    end

    stdevokedintegS(:, nsub) = stdevokedinteg2;
    mevokedintegS(:, nsub)   = mevokedinteg2;
    varevokedintegS(:, nsub) = varevokedinteg;

end

save('Ignition100_HCP_4runs.mat', ...
     'stdevokedintegS', 'mevokedintegS', 'varevokedintegS');

fprintf('Done. Results saved to Ignition100_HCP_4runs.mat\n');
