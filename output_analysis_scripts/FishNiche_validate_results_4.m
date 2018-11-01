lakelistfile  = 'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\lakes\test.csv';%2018-5-15 MC 2017SwedenList_only_validation_lakes
outputdir = 'C:\Users\Marianne\Documents\Fish_niche\MDN_FishNiche_2017\output';%2018-5-15 MC

% The following are for scenario 2 only, it should be automated based on scenario choice.
m1 = 'MOHC-HadGEM2-ES';% 2018-05-25 MC model 4
m2 = 'r1i1p1_SMHI-RCA4_v1_day'% 2018-05-25 MC model 4
exA = 'historical';
y1A = 2001;
exB = 'rcp45';
y1B = 2006;
y2A = y1A + 4;
y2B = y1B + 4;

modelstart = datenum([y1A, 1, 1]);
modelend   = datenum([y2B, 12, 31]);
ntimesteps = modelend-modelstart+1;

lakes = readLakeListCsv(lakelistfile);

nlakes = length(lakes);

%lakeswehavedatafor = [698, 67035, 19167, 31985, 310, 32276, 31895, 59185, 4810];
lakeswehavedatafor = [698, 67035, 19167, 31895, 310, 32276, 99045, 99516, 6950];


subplotindex = 1;

fig_temperature = figure('name', 'Temperatures');
fig_oxygen      = figure('name', 'Oxygen');

for lakenum = 1:nlakes
    
    curlake = lakes(lakenum);
    
    if any(curlake.lake_id == lakeswehavedatafor)
    %if curlake.lake_id == 698
        outputdir2 = getOutputPathFromEbHex(outputdir, lakes(lakenum).ebhex);
        d4 = sprintf('EUR-11_%s_%s-%s_%s_%d0101-%d1231', m1, exA, exB, m2, y1A, y2B);

        lakedir = strcat(outputdir2, d4, '\');
        disp(lakedir);

        O2model   = csvread(strcat(lakedir, 'O2zt.csv'))';
        Tmodel    = csvread(strcat(lakedir, 'Tzt.csv'))';

        filename = 'C:/Users/Marianne/Documents/Fish_niche/MDN_FishNiche_2017/sweden_inflow_data/Validation_data_for_lookup.xlsx';%MC test
        worksheet = sprintf('%d', curlake.lake_id);

        [data_num, data_txt, data_raw] = xlsread(filename, worksheet);
        
        dates = datenum(data_txt(2:end, 1));
        depths = data_num(:, 1);
        O2raw = data_num(:, 2) * 1000;
        Traw= data_num(:, 3);

        lakedepth = ceil(curlake.depth);
        %O2data = zeros(lakedepth, ntimesteps) * NaN;
        %Tdata = zeros(lakedepth, ntimesteps) * NaN;
        
        figure(fig_temperature);
        ax_t = subplot(3, 3, subplotindex);
        
        figure(fig_oxygen);
        ax_o = subplot(3, 3, subplotindex);
        
        hold(ax_t, 'on');
        hold(ax_o, 'on');
        
        anydata_t = false;
        anydata_o = false;
        
        T_model_samples = [];
        T_data_samples = [];
        
        O2_model_samples = [];
        O2_data_samples = [];
        
        
        crashed = false;
        try
            for ii=1:length(dates)
                dnum = datenum(dates(ii));
                dateindex = dnum - modelstart + 1
                if dateindex >= 1 && dateindex <= ntimesteps
                    depth = ceil(depths(ii));
                    O2_data = O2raw(ii);
                    T_data = Traw(ii);

                    O2_model = O2model(depth, dateindex);
                    T_model = Tmodel(depth, dateindex);
                    
                    if ~isnan(T_data)
                        plot(ax_t, T_data, T_model, 'b.');
                        anydata_t = true;
                        
                        T_model_samples = [T_model_samples T_model];
                        T_data_samples = [T_data_samples T_data];
                    end
                    
                    if ~isnan(O2_data)
                        plot(ax_o, O2_data, O2_model, 'b.');
                        anydata_o = true;
                        
                        O2_model_samples = [O2_model_samples O2_model];
                        O2_data_samples = [O2_data_samples O2_data];
                    end
                end

            end
        catch err
            crashed = true;
            disp(sprintf('Error when handling lake %s (%d)\n', curlake.name, curlake.lake_id));
            disp(err);
        end
        
        label_t = sprintf('%s (%d)', curlake.name, curlake.lake_id);
        label_o = sprintf('%s (%d)', curlake.name, curlake.lake_id);
        if crashed
            label_t = sprintf('%s - Error in program, unreliable', label_t);
            label_o = sprintf('%s - Error in program, unreliable', label_o);
        else
            if ~anydata_t
                label_t = sprintf('%s - No data in timespan', label_t);
            else
                RMSE_T = sqrt(mean( (T_model_samples - T_data_samples).^2 ));
                label_t = sprintf('%s, RMSE: %.2f', label_t, RMSE_T);
            end
            if ~anydata_o
                label_o = sprintf('%s (No data in timespan)', label_o);
            else
                RMSE_O = sqrt(mean( (O2_model_samples - O2_data_samples).^2 ));
                label_o = sprintf('%s, RMSE: %.2f', label_o, RMSE_O);
            end
        end
        
         
        xlabel(ax_t, label_t);
        xlabel(ax_o, label_o);
        
        axi = axis(ax_t);
        tempmax = max(axi);
        plot(ax_t, [0, tempmax], [0, tempmax], 'r');
        
        axi = axis(ax_o);
        tempmax = max(axi);
        plot(ax_o, [0, tempmax], [0, tempmax], 'r');
        
        subplotindex = subplotindex + 1;
        hold(ax_t, 'off');
        hold(ax_o, 'off');
        
    end
end