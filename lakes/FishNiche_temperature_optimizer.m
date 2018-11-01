population_size = 40;  % Population size for each generation of the genetic algorithm
max_generations = 9;   % How many generations to run the genetic algorithm for

lakelistfile  = '2017SwedenList_temperature_validation.csv';
outputdir = '../output';
validationfile = '../sweden_inflow_data/Validation_data_for_lookup.xlsx';
main_scriptfile = 'runlakesGoran_par.py';
templatefile = 'mylake_param_template.txt';


% The following are for scenario 2 only, it should be automated based on scenario choice.
m1 = 'ICHEC-EC-EARTH';
m2 = 'r3i1p1_DMI-HIRHAM5_v1_day';
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

% Preparing validation data
for ii=1:nlakes
    curlake = lakes(ii);
    
    outputdir2 = getOutputPathFromEbHex(outputdir, curlake.ebhex);
    d4 = sprintf('EUR-11_%s_%s-%s_%s_%d0101-%d1231', m1, exA, exB, m2, y1A, y2B);
    lakes(ii).outputfile = strcat(outputdir2, d4, '/Tzt.csv');
    
    lakedepth = floor(curlake.depth);
    
    filename = validationfile;
    worksheet = sprintf('%d', curlake.lake_id);

    [data_num, data_txt, data_raw] = xlsread(filename, worksheet);
    dates = datenum(data_txt(2:end, 1));
    depths = data_num(:, 1);
    %O2raw = data_num(:, 2) * 1000; % Remember conversion factor for O2
    Traw= data_num(:, 3);
    
    Data = zeros(lakedepth, ntimesteps) * NaN;
    
    for jj=1:length(dates)
        dnum = datenum(dates(jj));
        dateindex = dnum - modelstart + 1;
        if dateindex >= 1 && dateindex <= ntimesteps
            depth = ceil(depths(jj));
            T_data = Traw(jj);

            if ~isnan(T_data)
                Data(depth, dateindex) = T_data;
            end
        end
    end
    
    lakes(ii).data_mask = ~isnan(Data);
    lakes(ii).Data = Data(lakes(ii).data_mask);
end

% For temperature
varyindexes = [2, 3, 5, 11, 16, 17, 24, 25];
initialparam = [0.004, 0.000898, 2.75, 0.45, 1.15, 0, 2.5, 1];

% Best so far: 0.004 0.003 0.950 1.626 1.290 0.985 8.031 2.403
% New best so far: 0.004 0.003 0.950 0.099 1.423 0.985 2.205 2.403

minparam = initialparam * 0.2;
maxparam = initialparam * 5;
minparam(5) = 0.5;
maxparam(5) = 1.5;
minparam(6) = -4;
maxparam(6) = 4;

% Test it
%error = FishNiche_single_run(main_scriptfile, lakelistfile, templatefile, lakes, varyindexes, initialparam);

runfun = @(varyparam) FishNiche_single_run(main_scriptfile, lakelistfile, templatefile, lakes, varyindexes, varyparam);

options = optimoptions('ga', 'MaxGenerations', max_generations, 'PopulationSize', population_size, 'InitialPopulationMatrix', initialparam);

% Running the genetic optimizer algorithm
[optimal_parameters, optimal_ss, exitflag, output, pop, scores] = ga(runfun, length(varyindexes),...
    [], [], [], [], minparam, maxparam,...
    [], [], options);


function error = FishNiche_single_run(main_scriptfile, lakelistfile, templatefile, lakedata, varyindexes, varyparam)
    
    % Write parameter template file that can be used by Fish Niche
    writeParamTemplate(varyindexes, varyparam, templatefile);

    %TODO: only run the lakes where we have relevant data for this time
    %period. Could be done manually by just using a different lake file.

    command = sprintf('python %s %d %d %s', main_scriptfile, 2, 2, lakelistfile);
    system(command);
    
    totaldatalength = 0;
    for ii=1:length(lakedata)
        totaldatalength = totaldatalength + length(lakedata(ii).Data);
    end
    
    error = 0;
    
    for ii=1:length(lakedata)
        len = length(lakedata(ii).Data);
        if len  > 0
            Model_all = csvread(lakedata(ii).outputfile)';
            %fprintf(1, '%d, model: %d x %d, data: %d x %d\n', ii, size(Model_all,1), size(Model_all, 2), size(lakedata(ii).data_mask, 1), size(lakedata(ii).data_mask, 2));
            Model = Model_all(lakedata(ii).data_mask);
            
            error = error + nansum( (Model - lakedata(ii).Data).^2 ) / len;
        end
    end
    
    %error = nansum( (data_total-model_total).^2 );
    
    % Debug output.
    nf = java.text.DecimalFormat;
    errstr = char(nf.format(error));
    fprintf(1, '\n');
    fprintf(1, '*******************************************************************************************\n');
    fprintf(1, '                Single model run finished. Error: %s\n', errstr);
    fprintf(1, 'Parameters in this run:');
    for ii = 1:length(varyparam)
    fprintf(1, ' %.3f', varyparam(ii));
    end
    fprintf('\n');
    fprintf(1, '*******************************************************************************************\n');
    fprintf(1, '\n');
end


function writeParamTemplate(varyindexes, varyparam, templatefile)
    fid = fopen(templatefile, 'w');
    fprintf(fid, '-999	"Mylake parameters"\n');
    fprintf(fid, 'Parameter\tValue\tMin\tMax\tUnit\n');

    Params = cell(60, 5);

    Params(1,:)  = {'dz', '1', '0.5', '2', 'm'};
    Params(2,:)  = {'Kz_ak', '0.004', 'NaN', 'NaN', '(-)'};
    Params(3,:)  = {'Kz_ak_ice', '0.003', 'NaN', 'NaN', '(-)'};
    Params(4,:)  = {'Kz_N0', '7.00E-05', 'NaN', 'NaN', 's-2'};
    Params(5,:)  = {'C_shelter', '0.950', 'NaN', 'NaN', '(-)'};
    Params(6,:)  = {'latitude', '%.5f', 'NaN', 'NaN', 'dec.deg'}; %Is later inserted for each individual lake by the FishNiche application.
    Params(7,:)  = {'longitude', '%.5f', 'NaN', 'NaN', 'dec.deg'}; %Is later inserted for each individual lake by the FishNiche application.
    Params(8,:)  = {'alb_melt_ice', '0.6', 'NaN', 'NaN', '(-)'};
    Params(9,:)  = {'alb_melt_snow', '0.9', 'NaN', 'NaN', '(-)'};
    Params(10,:) = {'PAR_sat', '3.00E-05', '1.00E-05', '1.00E-04', 'mol m-2 s-1'};
    Params(11,:) = {'f_par', '0.99', 'NaN', 'NaN', '(-)'};
    Params(12,:) = {'beta_chl', '0.015', '0.005', '0.045', 'm2 mg-1'};
    Params(13,:) = {'lambda_I', '5', 'NaN', 'NaN', 'm-1'};
    Params(14,:) = {'lambda_s', '15', 'NaN', 'NaN', 'm-1'};
    Params(15,:) = {'sed_sld', '0.36', 'NaN', 'NaN', 'm3/m3'};
    Params(16,:) = {'I_scV', '1.423', 'NaN', 'NaN', '(-)'};
    Params(17,:) = {'I_scT', '0.985', 'NaN', 'NaN', 'deg C'};
    Params(18,:) = {'I_scC', '1', 'NaN', 'NaN', '(-)'};
    Params(19,:) = {'I_scS', '1', 'NaN', 'NaN', '(-)'};
    Params(20,:) = {'I_scTP', '1', '0.4', '0.8', '(-)'};
    Params(21,:) = {'I_scDOP', '1', 'NaN', 'NaN', '(-)'};
    Params(22,:) = {'I_scChl', '1', 'NaN', 'NaN', '(-)'};
    Params(23,:) = {'I_scDOC', '1', 'NaN', 'NaN', '(-)'};
    Params(24,:) = {'swa_b0', '2.205', 'NaN', 'NaN', 'm-1'};
    Params(25,:) = {'swa_b1', '2.403', '0.8', '1.3', 'm-1'};
    Params(26,:) = {'S_res_epi', '3.30E-07', '7.30E-08', '1.82E-06', 'm d-1 (dry mass)'};
    Params(27,:) = {'S_res_hypo', '3.30E-08', 'NaN', 'NaN', 'm d-1 (dry mass)'};
    Params(28,:) = {'H_sed', '0.3', 'NaN', 'NaN', 'm'};
    Params(29,:) = {'Psat_Lang', '2500', 'NaN', 'NaN', 'mg m-3'};
    Params(30,:) = {'Fmax_Lang', '8000', '5000', '10000', 'mg kg-1'};
    Params(31,:) = {'Uz_Sz', '0.3', '0.1', '1', 'm d-1'};
    Params(32,:) = {'Uz_Chl', '0.16', '0.05', '0.5', 'm d-1'};
    Params(33,:) = {'Y_cp', '1', 'NaN', 'NaN', '(-)'};
    Params(34,:) = {'m_twty', '0.2', '0.1', '0.3', 'd-1'};
    Params(35,:) = {'g_twty', '1.5', '1', '1.5', 'd-1'};
    Params(36,:) = {'k_sed_twty', '2.00E-04', 'NaN', 'NaN', 'd-1'};
    Params(37,:) = {'k_dop_twty', '0', 'NaN', 'NaN', 'd-1'};
    Params(38,:) = {'P_half', '0.2', '0.2', '2', 'mg m-3'};
    Params(39,:) = {'PAR_sat2', '3.00E-05', 'NaN', 'NaN', 'mol m-2 s-1'};
    Params(40,:) = {'beta_chl2', '0.015', 'NaN', 'NaN', 'm2 mg-1'};
    Params(41,:) = {'Uz_Chl2', '0.16', 'NaN', 'NaN', 'm d-1'};
    Params(42,:) = {'m_twty2', '0.2', 'NaN', 'NaN', 'd-1'};
    Params(43,:) = {'g_twty2', '1.5', 'NaN', 'NaN', 'd-1'};
    Params(44,:) = {'P_half2', '0.2', 'NaN', 'NaN', 'mg m-3'};
    Params(45,:) = {'oc_DOC', '0.01', 'NaN', 'NaN', 'm2 mg-1'};
    Params(46,:) = {'qy_DOC', '0.1', 'NaN', 'NaN', 'mg mol-1'};
    Params(47,:) = {'k_BOD', '0.01', 'NaN', 'NaN', 'd-1'};
    Params(48,:) = {'k_SOD', '100', 'NaN', 'NaN', 'mg m-2'};
    Params(49,:) = {'theta_BOD', '1.047', 'NaN', 'NaN', '(-)'};
    Params(50,:) = {'theta_BOD_ice', '1.13', 'NaN', 'NaN', '(-)'};
    Params(51,:) = {'theta_SOD', '1', 'NaN', 'NaN', '(-)'};
    Params(52,:) = {'theta_SOD_ice', '1', 'NaN', 'NaN', '(-)'};
    Params(53,:) = {'theta_T', '4', 'NaN', 'NaN', 'deg C'};
    Params(54,:) = {'pH', '5.2', 'NaN', 'NaN', '(-)'};
    Params(55,:) = {'I_scDIC', '1', 'NaN', 'NaN', '(-)'};
    Params(56,:) = {'Mass_Ratio_C_Chl', '100', 'NaN', 'NaN', '(-)'};
    Params(57,:) = {'SS_C', '0.25', 'NaN', 'NaN', ' '};
    Params(58,:) = {'density_org_H_nc', '1.95', 'NaN', 'NaN', ' '};
    Params(59,:) = {'density_inorg_H_nc', '2.65', 'NaN', 'NaN', ' '};
    Params(60,:) = {'I_scO', '1', 'NaN', 'NaN', '(-)'};

    for ii=1:length(varyindexes)
        Params{varyindexes(ii), 2} = sprintf('%f', varyparam(ii));
    end

    for ii=1:length(Params)
        fprintf(fid, '%s\t%s\t%s\t%s\t%s\n', Params{ii,1}, Params{ii,2}, Params{ii,3}, Params{ii,4}, Params{ii,5} );
    end

    fclose(fid);
end
