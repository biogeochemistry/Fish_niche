function Result = MyLake_optimizer(m_start, m_stop, parfile, inputfile, initfile,outdir, latitude, longitude)
    %% Project specific setup script

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

    
    varyindexes = [23 25 47 48 ];%I_scDOC, Swa_b1, k_BOD, k_SOD

    % Setting up the min and max boundaries for each covarying set of parameters.
    minparam = [ 0.00001, 0, 0.4, 0.4, 0.4, 0.4];
    maxparam = [ 0.0001, 1, 1, 1, 4, 2];

    % The best initial guess for the values of each set of covarying parameters (can have
    % multiple rows for multiple initial guesses. (up to population_size rows)
    initial_guess = [];

    modeleval      = @MyLake_model_evaluation;
    errfun         = @error_function;
    filenameprefix = 'T'; % Prefix for the .mat file where the optimal parameters are saved in the end.

    do_MyLake_optimization(m_start, m_stop, K_lake, ...
        varyindexes, minparam, maxparam, initial_guess, modeleval, errfun,...
        population_size, max_generations, paralellize, filenameprefix);

    Result = "all done!";
    %% BEGIN project specific evaluation functions

    % Model evaluation function. It should take the parameters
    % (m_start, m_stop, K_sediment, K_lake) and return a ModelResult struct.
    % The ModelResult struct should contain whatever the error function needs
    % to compare the model result to data.
    function ModelResult = MyLake_model_evaluation(m_start, m_stop,lake_params)
        %run_INCA = 0; % 1- MyLake will run INCA, 0- No run
        %use_INCA = 0; % 1- MyLake will take written INCA input, either written just now or saved before, and prepare inputs from them. 0- MyLake uses hand-made input files
        %save_initial_conditions = false; % save final concentrations as initial for the next run
        %run_ID = 0;
        %clim_ID = 0;

        mylakeGoran(initfile, parfile, inputfile, m_start, m_stop, outdir,lake_params)
        %MyLake_results  = fn_MyL_application(m_start, m_stop, sediment_params, lake_params, use_INCA, run_INCA, run_ID, clim_ID, save_initial_conditions); % runs the model and outputs obs and sim
        
        %comparison
        ModelResult = compare_model_result_data(outdir, m_start, m_stop);
        %csvread(sprintf("%s/tzt.csv",outdir));
        
        %Totalchl = MyLake_results.basin1.concentrations.Chl + MyLake_results.basin1.concentrations.C;
        %ModelResult.Chlintegratedepi = transpose(mean(Totalchl(1:4,:)));
    end

    % Error function. The error function takes a ModelResult
    % struct and a Data struct. Remember that these two structs are user
    % defined. They should contain whatever is needed for the error function to
    % compare the model result to measured data. It has to return a positive
    % number err, which is smaller the better fit the model is to the data.
    function err = error_function(ModelResult)
        err = sqrt(nanmean((ModelResult.T_data-ModelResult.T_model).^2));
        %R = corrcoef(ModelResult.T_data,ModelResult.T_model,'rows','complete');
        %R2 = (R.^2);
        
    end

    %% END project specific evaluation functions

    %% The following two functions are general and should not have to be modified in each project

    function do_MyLake_optimization(m_start, m_stop, K_values_lake, varyindexes, minparam, maxparam, initial_guess, ...
        modeleval, errfun, max_generations, population_size, parallelize, filenameprefix)

        runfunc = @(varyparam)(MyLake_optimizer_single_run(m_start, m_stop,  K_values_lake,  varyindexes, modeleval, errfun, varyparam));

        options = optimoptions('ga', 'MaxGenerations', max_generations, 'PopulationSize', population_size, 'UseParallel', parallelize, 'InitialPopulationMatrix', initial_guess);

        % Running the genetic optimizer algorithm
        [optimal_parameters, optimal_ss, exitflag, output, pop, scores] = ga(runfunc, size(varyindexes,2),...
            [], [], [], [], minparam, maxparam,...
            [], [], options);

        cl = clock;
        filename = sprintf('%s_optimized_parameters_%d_%d_%d.mat', filenameprefix, cl(3), cl(2), cl(1));
        save(filename, 'optimal_parameters', 'varyindexes');
    end

    function err = MyLake_optimizer_single_run(m_start, m_stop, K_lake, varyindexes, modeleval, errfun, varyparam)

        % Inserting the varying parameters into K_lake and K_sediments
        for ii = 1:size(varyindexes,2)
            if varyindexes(1,ii) <= length(K_lake)
                for jj = 1:size(varyindexes, 1)
                    if ~isnan(varyindexes(jj, ii))
                        K_lake{varyindexes(jj,ii), 1} = varyparam(ii);
                    end
                end
            
            end
        end

        % Running the model
        ModelResult = modeleval(m_start, m_stop,K_lake);

        % Evaluating the error
        err = errfun(ModelResult);

        % Debug output.
        nf = java.text.DecimalFormat;
        ssstr = char(nf.format(err));
        fprintf(1, '\n');
        fprintf(1, '*******************************************************************************************\n');
        fprintf(1, '                Single model run finished. Error: %s\n', ssstr);
        fprintf(1, 'Parameters in this run:');
        for ii = 1:length(varyparam)
        fprintf(1, ' %.3g', varyparam(ii));
        end
        fprintf('\n');
        fprintf(1, '*******************************************************************************************\n');
        fprintf(1, '\n');
    end
    end