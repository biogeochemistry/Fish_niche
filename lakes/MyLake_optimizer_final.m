function Result = MyLake_optimizer_final(m_start, m_stop,spin_up, parfile, inputfile, initfile,outdir, latitude, longitude,icedays,variable_calibrated)
%MyLake_optimizer_final - Main Function to optimize the MyLake model for one lake with Genetic algorythm.
%This script is a modified version from Mylake_optimizer.m from Kateri Salk (Project MyLake Lake 227)
%
% Syntax:  Result = MyLake_optimizer_final(m_start, m_stop,spin_up, parfile, inputfile, initfile,outdir, latitude, longitude,icedays, variable_calibrated)
%
% Inputs:
%    m_start - Description
%    m_stop - Description
%    spin_up - Description
%    parfile- Description
%    inputfile- Description
%    initfile- Description
%    outdir- Description
%    latitude- Description
%    longitude- Description
%    icedays- Description
%    variable_calibrated- Description
%
% Outputs:
%    Result - Description
%
% Other m-files required: none
% Subfunctions: MyLake_model_evaluation, error_function, do_MyLake_optimization
% MAT-files required: none
%
% See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2
% Author: Marianne Cote
% Website: https://github.com/biogeochemistry/Fish_niche
% Nov 2018; Last revision: 25-May-2021
%------------- BEGIN CODE --------------    
   
    % Population size for each generation of the genetic algorithm. If you use parallellization it will be
    % more efficient if this is a multiple of the number of cores on the machine.
    % It is generally more important to have a large
    % population_size than it is to have a large max_generations since it is
    % important to sample a large part of the parameter space before starting
    % to sample around the best fits.
    population_size = 48;  
    % Max generations to run. The maximal amount of runs is population_size*max_generations.
    max_generations = 32;
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
    
    if (variable_calibrated == "temperature")
        varyindexes = [4 5 8 9 16 17  24 25];%kz_N0, c_shelter, alb_melt_ice, alb_melt_snow,I_scV, i_scT swa_b0, swa_b1

        % Setting up the min and max boundaries for each covarying set of parameters.
                       %kz_N0,  c_shelter, alb_melt_ice, alb_melt_snow, I_scV, i_scT, swa_b0, swa_b1,
        minparam = [ 0.000001,     0.0001,          0.0,           0.4, 0.001,   0.0,    0.1,    0.1];
        maxparam = [     0.01,          1,          0.4,             1,     2,     5,     10,     10];

        % The best initial guess for the values of each set of covarying parameters (can have
        % multiple rows for multiple initial guesses. (up to population_size rows))
        dataS = readtable(sprintf("%s/Observed_Secchi.csv",outdir),'ReadVariableNames',0);
        column = dataS{2:end,2:end}(:);
        secchimean = nanmean(column);
        if isnan(secchimean)
            swa_b1 =0.8;
        else
            swa_b1 = 1.48/secchimean;
        end
        
        if isfile(sprintf("%s/Calibration_Complete.csv",outdir))
            bestvalue = readtable(sprintf("%s/Calibration_Complete.csv",outdir),'ReadVariableNames',0);
                                     %kz_N0,     c_shelter,   alb_melt_ice,  alb_melt_snow,         I_scV,         i_scT,       swa_b0
            initial_guess = [bestvalue{1,1},bestvalue{1,2}, bestvalue{1,3}, bestvalue{1,4},bestvalue{1,5},bestvalue{1,6},bestvalue{1,7},swa_b1];
        
        else
                                %kz_N0,c_shelter,alb_melt_ice,alb_melt_snow,I_scV, i_scT,swa_b0,swa_b1
            initial_guess = [0.00007,       0.05,         0.3,          0.7,    1,    0,   2.5,swa_b1];
        
        end
    elseif(variable_calibrated == "oxygen")
        
        varyindexes = [23 47 48 60];%I_scDOC, k_BOD, k_SOD, I_scO
    
        % Setting up the min and max boundaries for each covarying set of parameters.
                    %I_scDOC,    k_BOD, k_SOD, I_scO
        minparam = [    0.01, 0.000001,     5, 0.001];
        maxparam = [      10,      3.0,  1500,   5];

        % The best initial guess for the values of each set of covarying parameters (can have
        % multiple rows for multiple initial guesses. (up to population_size rows)

        if isfile(sprintf("%s/Calibration_CompleteOXY.csv",outdir))
            bestvalue = readtable(sprintf("%s/Calibration_CompleteOXY.csv",outdir),'ReadVariableNames',0);
                                   %I_scDOC,       k_BOD,            k_SOD,         I_scO
            initial_guess = [bestvalue{1,1},bestvalue{1,2}, bestvalue{1,3}, bestvalue{1,4}];

        else
                         %I_scDOC, k_BOD, k_SOD,I_scO, 
            initial_guess = [4.75, 0.001,   100,    1];
        end
    
    else
        error("Error with the variable given (must be temperature or oxygen)");
    end

    modeleval      = @MyLake_model_evaluation;
    errfun         = @error_function;
    filenameprefix = "calibration_complete"; % Prefix for the .mat file where the optimal parameters are saved in the end.
    %try
        do_MyLake_optimization(m_start, m_stop,spin_up, K_lake, ...
            varyindexes, minparam, maxparam, initial_guess, modeleval, errfun,...
            population_size, max_generations, paralellize, filenameprefix);

        Result = "all done!";
%     catch
%        Result = "problem";
%     end

    %% BEGIN project specific evaluation functions

    function ModelResult = MyLake_model_evaluation(m_start, m_stop,spin_up, lake_params)
    %MyLake_model_evaluation - Nested Model evaluation function of MyLake_optimizer_final.
    %This script is a modified version from Mylake_optimizer.m from Kateri Salk (Project MyLake Lake 227)
    %
    % Syntax:  ModelResult = MyLake_model_evaluation(m_start, m_stop,spin_up, lake_params) 
    %
    % Inputs:
    %    m_start - Description
    %    m_stop - Description
    %    spin_up - Description
    %    lake_params- Description
    %
    % Outputs:
    %     ModelResult- this struct should contain whatever the error function needs to compare the model result to data.
    %
    % Other m-files required: none
    % Subfunctions: none
    % MAT-files required: none
    %
    % See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2
    % Author: Marianne Cote
    % Website: https://github.com/biogeochemistry/Fish_niche
    % Nov 2018; Last revision: 25-May-2021
    %------------- BEGIN CODE --------------
    % 
        tempfile = tempname(outdir);
        export_isimip_params_lake(lake_params, parfile, tempfile,variable_calibrated)
        ModelResult = mylakeGoran_optimizefinal(initfile, tempfile, inputfile, m_start, m_stop,spin_up, outdir,icedays,calibration,variable_calibrated);

    end

    % Error function. The error function takes a ModelResult
    % struct and a Data struct. Remember that these two structs are user
    % defined. They should contain whatever is needed for the error function to
    % compare the model result to measured data. It has to return a positive
    % number err, which is smaller the better fit the model is to the data.
    function [errA errT errO errS errI icemodel] = error_function(ModelResult)
        
        if (size(ModelResult.Dates) == 0) & (size(ModelResult.DatesO) == 0)
           
            errT = 9999999;
            errO = 0;
            errS = 0;
            
            errI = 0;
            icemodel = 0;
            
 
        else
            if(variable_calibrated == "temperature")
                errT = nansum((ModelResult.T_data-ModelResult.T_model).^2); %change from NRMSE to sum of squares 
                %errT = (sqrt(nanmean((ModelResult.T_data-ModelResult.T_model).^2)/size(ModelResult.T_data))/(nanmax(ModelResult.T_data)-nanmin(ModelResult.T_data));
                errI = ModelResult.ice;
                icemodel = ModelResult.icemodel;
                errO=0;
                if size(ModelResult.S_data) == 0
                    errS = 0;
                else
                    %errS = nansum((ModelResult.S_data-ModelResult.S_model).^2);
                    %errS = (sqrt(nanmean(((ModelResult.S_data-ModelResult.S_model)).^2)))/((nanmax(ModelResult.S_data)-nanmin(ModelResult.S_data)));
                    errS = nansum((ModelResult.S_data-ModelResult.S_model).^2); %change from NRMSE to sum of squares 

                    %errS=0;
                end
            else
                errT = 0;
                icemodel =ModelResult.icemodel;
                errS = 0;
            end
            if(variable_calibrated == "oxygen")
                if size(ModelResult.DatesO) == 0
                    errO = 99999999;
                    errS = 0;
                    errT = 0;
                    errI = 0;
                    icemodel = 0;
            
                else
                    %Normalized root-mean-square deviation
                    errO =nansum((ModelResult.O_data-ModelResult.O_model).^2);
                    errI = ModelResult.ice;
                    icemodel = ModelResult.icemodel;
                    errS = 0;
                    errT = 0;
                    %errO = (sqrt(nanmean(((ModelResult.O_data-ModelResult.O_model)*0.001).^2)))/((nanmax(ModelResult.O_data)- nanmin(ModelResult.O_data))*0.001); %*0.001 convert mg/m*-3 to mg/l

                end
            else
                errO=0;
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
        
        if (variable_calibrated == "temperature")
        
            export_isimip_params_lake(K_lake,parfile, fullfile(outdir,"Completepar"),variable_calibrated)
            ModelResult = mylakeGoran_optimizefinal(initfile,fullfile(outdir,"Completepar"), inputfile, m_start, m_stop,spin_up, outdir,icedays,calibration,variable_calibrated);
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
            
        elseif(variable_calibrated == "oxygen")
            export_isimip_params_lake(K_lake,parfile, fullfile(outdir,"CompleteparOXY"),variable_calibrated)
            
            ModelResult = mylakeGoran_optimizefinal(initfile,fullfile(outdir,"CompleteparOXY"), inputfile, m_start, m_stop,spin_up, outdir,icedays,calibration,variable_calibrated);
            [err errt erro errs] = errfun(ModelResult);
            scores_all = [err errt erro errs];
            %filename = sprintf('%s_optimized_parameters_%d_%d_%d.mat', filenameprefix, cl(3), cl(2), cl(1));
            %save(filename, 'optimal_parameters', 'varyindexes');
            filename2 = sprintf("%s/Calibration_CompleteOXY.csv",outdir);
            writematrix(optimal_parameters,filename2);
            filename2 = sprintf("%s/Calibration_CompletefvalOXY.csv",outdir);
            writematrix(fval,filename2);
            filename2 = sprintf("%s/Calibration_CompletepopulationOXY.csv",outdir);
            writematrix(population,filename2);
            filename2 = sprintf("%s/Calibration_CompletescoreOXY.csv",outdir);
            writematrix(scores_all,filename2);
        end
        
       
        
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