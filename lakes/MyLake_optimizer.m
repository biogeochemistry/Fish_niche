function Result = MyLake_optimizer(m_start, m_stop,spin_up, parfile, inputfile, initfile,outdir, latitude, longitude,icedays)
    %% Project specific setup script

    % Population size for each generation of the genetic algorithm. If you use parallellization it will be
    % more efficient if this is a multiple of the number of cores on the machine.
    % It is generally more important to have a large
    % population_size than it is to have a large max_generations since it is
    % important to sample a large part of the parameter space before starting
    % to sample around the best fits.
    population_size = 48;  
    % Max generations to run. The maximal amount of runs is population_size*max_generations.
    max_generations = 48;
    paralellize     = true; % Run many lake processes in parallell (saves time if the computer has many cores).
    calibration = 1;
    % Loading a priori values for all parameters.
    if exist(parfile, 'file') == 2
        fid = readtable(parfile,'HeaderLines',1,'Format','%s %f %f %f %s');
        K_lake = fid(:,{'Value','Parameter'});
    else
        K_lake = load_params_lake(longitude, latitude);
    end
    
  
    % varyindexes is the indexes of the parameters in K_lake and K_sediments we want to vary during
    % optimization. 
    % Every value that has an index that belongs to the same column in
    % varyindexes is given the same value. indexes in the second row or later
    % can be NaN if you don't want the index in the first row of that column to
    % covary with any in the next row.

    %varyindexes = [23 25 47 48 ];%I_scDOC, Swa_b1, k_BOD, k_SOD
    %varyindexes = [4 5 8 9 16 17 24 25];%kz_N0, c_shelter, alb_melt_ice, alb_melt_snow,I_scV, i_scT swa_b0, swa_b1
    varyindexes = [4 5 8 9 16 17  24 25];%kz_N0, c_shelter, alb_melt_ice, alb_melt_snow,I_scV, i_scT swa_b0, swa_b1
%     varyindexes = [4 5 8 9 17  24 25];%kz_N0, c_shelter, alb_melt_ice, alb_melt_snow, i_scT swa_b0, swa_b1
    %varyindexes = [4 5 24 25];%kz_N0, c_shelter, alb_melt_ice, alb_melt_snow,I_scV, i_scT swa_b0, swa_b1

    % Setting up the min and max boundaries for each covarying set of parameters.
    %minparam = [ 0.01, 0.1, 5, 0.000001];
    %maxparam = [ 10, 2, 1500, 3.0];
%                 %kz_N0,    c_shelter, alb_melt_ice, alb_melt_snow, I_scV, i_scT swa_b0, swa_b1
%     minparam = [ 0.000001, 0.001,    0.0,          0.5,           0.01,  0.0, 0.01,   0.01];
%     maxparam = [ 0.001,    1,        0.4,          1,             100,   5,   10,     10];

                %kz_N0,    c_shelter, alb_melt_ice, alb_melt_snow, I_scV,i_scT  swa_b0
    minparam = [ 0.000001, 0.0001,    0.0,          0.4,           0.001,  0.00,    0.0, 0.05];
    maxparam = [ 0.01,          1,    0.4,            1,             2,    5,        20, 15];
%                  %kz_N0,    c_shelter, alb_melt_ice, alb_melt_snow, i_scT swa_b0 swab1
%     minparam = [ 0.000001,     0.0001,    0.0,          0.4,           0.00,    0.0, 0.05];
%     maxparam = [ 0.01,          1,    0.4,            1,                5,        20, 15];
%     
%                 %kz_N0,    c_shelter,I_scV  swa_b0 swab1
%     minparam = [ 0.000001, 0.0001,    0.1, 0.05];
%     maxparam = [ 0.01,    1,          20, 15];
    % The best initial guess for the values of each set of covarying parameters (can have
    % multiple rows for multiple initial guesses. (up to population_size rows)
    %initial_guess = [4.75, 0.42, 100, 0.001];
    dataS = readtable(sprintf("%s/Observed_Secchi.csv",outdir),'ReadVariableNames',0);
    column = dataS{2:end,2:end}(:);
    secchimean = nanmean(column);
    if isnan(secchimean)
        swa_b1 =0.8;
    else
        swa_b1 = 1.48/secchimean;
    end
%                     %kz_N0,   c_shelter, alb_melt_ice, alb_melt_snow, I_scV, i_scT swa_b0, swa_b1
%     initial_guess = [0.00007, 0.05,      0.3,          0.7,          1,     1,    2.5,    swa_b1];
%                     kz_N0,   c_shelter, alb_melt_ice, alb_melt_snow,I_scV, i_scT  swa_b0
%     initial_guess = [0.00007, 0.05,      0.3,          0.7,          1,     1,       2.5];
    try
        if isfile(sprintf("%s/Calibration_Complete.csv",outdir))
            bestvalue = readtable(sprintf("%s/Calibration_Complete.csv",outdir),'ReadVariableNames',0);
                                     %kz_N0,     c_shelter,   alb_melt_ice,  alb_melt_snow,         I_scV,         i_scT,       swa_b0
            initial_guess = [bestvalue{1,1},bestvalue{1,2}, bestvalue{1,3}, bestvalue{1,4},bestvalue{1,5},bestvalue{1,6},bestvalue{1,7},swa_b1];

        else
                                %kz_N0,c_shelter,alb_melt_ice,alb_melt_snow,I_scV, i_scT,swa_b0,swa_b1
            initial_guess = [0.00007,       0.05,         0.3,          0.7,    1,     0,   2.5,swa_b1];

        end
    catch
                               %kz_N0,c_shelter,alb_melt_ice,alb_melt_snow,I_scV, i_scT,swa_b0,swa_b1
        initial_guess = [0.00007,       0.05,         0.3,          0.7,    1,     0,   2.5,swa_b1];
    end
%         try
%         if isfile(sprintf("%s/Calibration_Complete.csv",outdir))
%             bestvalue = readtable(sprintf("%s/Calibration_Complete.csv",outdir),'ReadVariableNames',0);
%                                      %kz_N0,     c_shelter,   alb_melt_ice,  alb_melt_snow,        i_scT,       swa_b0
%             initial_guess = [bestvalue{1,1},bestvalue{1,2}, bestvalue{1,3}, bestvalue{1,4},bestvalue{1,6},bestvalue{1,7},swa_b1];
% 
%         else
%                                 %kz_N0,c_shelter,alb_melt_ice,alb_melt_snow, i_scT,swa_b0,swa_b1
%             initial_guess = [0.00007,       0.05,         0.3,          0.7,     0,   2.5,swa_b1];
% 
%         end
%     catch
%                                %kz_N0,c_shelter,alb_melt_ice,alb_melt_snow, i_scT,swa_b0,swa_b1
%         initial_guess = [0.00007,       0.05,         0.3,          0.7,        0,   2.5,swa_b1];
%     end
    
%     if isfile(sprintf("%s/Calibration_Complete.csv",outdir))
%         bestvalue = readtable(sprintf("%s/Calibration_Complete.csv",outdir),'ReadVariableNames',0);
%                                  %kz_N0,     c_shelter,       swa_b0
%         initial_guess = [bestvalue{1,1},bestvalue{1,2}, bestvalue{1,3},swa_b1];
%     
%     else
%                             %kz_N0,c_shelter,swa_b0,swa_b1
%         initial_guess = [0.00007,       0.05,    2.5,swa_b1];
%     
%     end

    modeleval      = @MyLake_model_evaluation;
    errfun         = @error_function;
    filenameprefix = "calibration_complete"; % Prefix for the .mat file where the optimal parameters are saved in the end.
    %try
        do_MyLake_optimization(m_start, m_stop,spin_up, K_lake, ...
            varyindexes, minparam, maxparam, initial_guess, modeleval, errfun,...
            population_size, max_generations, paralellize, filenameprefix);

        Result = "all done!";
    %catch
     %   Result = "problem";
    %end
    
    %% BEGIN project specific evaluation functions

    % Model evaluation function. It should take the parameters
    % (m_start, m_stop, K_sediment, K_lake) and return a ModelResult struct.
    % The ModelResult struct should contain whatever the error function needs
    % to compare the model result to data.
    function ModelResult = MyLake_model_evaluation(m_start, m_stop,spin_up, lake_params)
        %run_INCA = 0; % 1- MyLake will run INCA, 0- No run
        %use_INCA = 0; % 1- MyLake will take written INCA input, either written just now or saved before, and prepare inputs from them. 0- MyLake uses hand-made input files
        %save_initial_conditions = false; % save final concentrations as initial for the next run
        %run_ID = 0;
        %clim_ID = 0;
        tempfile = tempname(outdir);
        export_isimip_params_lake(lake_params, parfile, tempfile)
        ModelResult = mylakeGoran_optimize(initfile, tempfile, inputfile, m_start, m_stop,spin_up, outdir,icedays,calibration);
        %MyLake_results  = fn_MyL_application(m_start, m_stop, sediment_params, lake_params, use_INCA, run_INCA, run_ID, clim_ID, save_initial_conditions); % runs the model and outputs obs and sim
        
        %comparison
        %ModelResult = compare_model_result_data(outdir, m_start, m_stop);
        %csvread(sprintf("%s/tzt.csv",outdir));
        
        %Totalchl = MyLake_results.basin1.concentrations.Chl + MyLake_results.basin1.concentrations.C;
        %ModelResult.Chlintegratedepi = transpose(mean(Totalchl(1:4,:)));
    end

    % Error function. The error function takes a ModelResult
    % struct and a Data struct. Remember that these two structs are user
    % defined. They should contain whatever is needed for the error function to
    % compare the model result to measured data. It has to return a positive
    % number err, which is smaller the better fit the model is to the data.
    function [errA errT errO errS errI icemodel] = error_function(ModelResult)
        
        if size(ModelResult.Dates) == 0
           
            errT = 9999999;
            errO = 0;
            errS = 0;
            errI = 0;
            icemodel = 0;
 
        else
            errT = nansum((ModelResult.T_data-ModelResult.T_model).^2); %change from NRMSE to sum of squares 
            %errT = (sqrt(nanmean((ModelResult.T_data-ModelResult.T_model).^2)/size(ModelResult.T_data))/(nanmax(ModelResult.T_data)-nanmin(ModelResult.T_data));
            errI = ModelResult.ice;
            icemodel = ModelResult.icemodel;
            if size(ModelResult.DatesO) == 0
            errO = 0;
            else
                %Normalized root-mean-square deviation
                errO =0;
                %errO = (sqrt(nanmean(((ModelResult.O_data-ModelResult.O_model)*0.001).^2)))/((nanmax(ModelResult.O_data)- nanmin(ModelResult.O_data))*0.001); %*0.001 convert mg/m*-3 to mg/l

            end
            if size(ModelResult.S_data) == 0
                errS = 0;
            else
                %errS = nansum((ModelResult.S_data-ModelResult.S_model).^2);
                %errS = (sqrt(nanmean(((ModelResult.S_data-ModelResult.S_model)).^2)))/((nanmax(ModelResult.S_data)-nanmin(ModelResult.S_data)));
                errS = nansum((ModelResult.S_data-ModelResult.S_model).^2); %change from NRMSE to sum of squares 
            
                %errS=0;

            end
        
        end
        
        
         errA = errT+errO+errS;
            %R = corrcoef(ModelResult.T_data,ModelResult.T_model,'rows','complete');
            %R2 = (R.^2);
    end

    %% END project specific evaluation functions

    %% The following two functions are general and should not have to be modified in each project

    function do_MyLake_optimization(m_start, m_stop,spin_up, K_values_lake, varyindexes, minparam, maxparam, initial_guess, ...
        modeleval, errfun, max_generations, population_size, parallelize, filenameprefix)
        

    
        %%testtt = MyLake_optimizer_single_run(m_start, m_stop,  K_values_lake,  varyindexes, modeleval, errfun, [1,2.5,0.1,500]);
        runfunc = @(varyparam)(MyLake_optimizer_single_run(m_start, m_stop,spin_up,  K_values_lake,  varyindexes, modeleval, errfun, varyparam));

        options = optimoptions('ga', 'MaxGenerations', max_generations, 'PopulationSize', population_size, 'UseParallel', parallelize, 'InitialPopulationMatrix', initial_guess);
        
        % Running the genetic optimizer algorithm
        [optimal_parameters, fval,exitFlag,output,population,scores] = ga(runfunc, size(varyindexes,2),...
            [], [], [], [], minparam, maxparam,...
            [], [], options);

        cl = clock;
        for ii = 1:size(varyindexes,2)
            if varyindexes(ii) <= size(K_lake,1)
                K_lake{varyindexes(ii), 1} = optimal_parameters(ii);
                    
                end
            
            end
        export_isimip_params_lake(K_lake,parfile, fullfile(outdir,"Completepar"))
        calibration = 0;
        ModelResult = mylakeGoran_optimize(initfile,fullfile(outdir,"CompleteparOXY"), inputfile, m_start, m_stop,spin_up, outdir,icedays,calibration);
        [err errt erro errs erri icemodel] = errfun(ModelResult);
        scores_all = [err errt erro errs erri];
        %filename = sprintf('%s_optimized_parameters_%d_%d_%d.mat', filenameprefix, cl(3), cl(2), cl(1));
        %save(filename, 'optimal_parameters', 'varyindexes');
        filename2 = sprintf("%s/Calibration_Complete.csv",outdir);
        writematrix(optimal_parameters,filename2);
        filename2 = sprintf("%s/Calibration_Completefval.csv",outdir);
        writematrix(fval,filename2);
        filename2 = sprintf("%s/Calibration_Completepopulation.csv",outdir);
        writematrix(population,filename2);
        filename2 = sprintf("%s/Calibration_Completescore.csv",outdir);
        writematrix(scores_all,filename2);
        
        
       
        
    end

    function err = MyLake_optimizer_single_run(m_start, m_stop,spin_up, K_lake, varyindexes, modeleval, errfun, varyparam)

        % Inserting the varying parameters into K_lake and K_sediments
        for ii = 1:size(varyindexes,2)
            if varyindexes(ii) <= size(K_lake,1)
                K_lake{varyindexes(ii), 1} = varyparam(ii);
                    
                end
            
            end
        

        % Running the model
        ModelResult = modeleval(m_start, m_stop,spin_up,K_lake);

        % Evaluating the error
        [err errt erro errs erri icemodel] = errfun(ModelResult);

        % Debug output.
        nf = java.text.DecimalFormat;
        ssstr = char(nf.format(err));
        ssstrt = char(nf.format(errt));
        ssstro = char(nf.format(erro));
        ssstrs = char(nf.format(errs));
        ssstri = char(nf.format(erri));
        fprintf(1, '\n');
        fprintf(1, '*******************************************************************************************\n');
        fprintf(1, '                Single model run finished. Error: %s T: %s O: %s S: %s I: %s\n', ssstr,ssstrt,ssstro,ssstrs,ssstri);
        fprintf(1,'icemodel: %f icedata: %f\n' ,icemodel,icedays);
        fprintf(1, 'Parameters in this run:');
        for ii = 1:length(varyparam)
        fprintf(1, ' %.3g', varyparam(ii));
        end
        fprintf('\n');
        fprintf(1, '*******************************************************************************************\n');
        fprintf(1, '\n');
    end
    end