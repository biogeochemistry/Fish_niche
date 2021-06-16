function ModelComparison = compare_model_result_data_final(outdir, m_start2, m_stop2,Tzt,O2zt,lambdazt,His,icedays,calibration,variable_calibrated)
    %model = readtable(sprintf("%s/Tzt.csv",outdir));
    directory = split(outdir,"EUR-11");
    d4 = sprintf('EUR-11_ICHEC-EC-EARTH_historical-rcp45_r3i1p1_DMI-HIRHAM5_v1_day_20010101-20101231');
    lakedir = strcat(directory{1,1},d4,"/calibration_result");
    
    m_start = datetime(sprintf('%d-01-01',m_start2),'Format','yyyy-MM-dd')-1; 
    m_stop =  datetime(sprintf('%d-12-31',m_stop2),'Format','yyyy-MM-dd'); 
    model = Tzt.';
    modelS = lambdazt.';
    data = readtable(sprintf("%s/Observed_Temperature.csv",lakedir));
    dataS = readtable(sprintf("%s/Observed_Secchi.csv",lakedir));
    depthsmodel = 0.5:size(model,2);
    depths = data{1,2:size(data,2)};
    initial_date = datetime(data{2,1},'Format','yyyy-MM-dd');
    
    %ice comparison
    
    his2 = His(7,:);
    sum2 = nansum(his2)/10;
    ModelComparison.ice = abs(sum2 - icedays);
    ModelComparison.icedata =  icedays;
    ModelComparison.icemodel = sum2 ;
    ModelComparison.Dates = 0 ;
    
    if (variable_calibrated == "temperature")
        %temperature treatment
        model = Tzt.';
        modelS = lambdazt.';
        data = readtable(sprintf("%s/Observed_Temperature.csv",lakedir));
        dataS = readtable(sprintf("%s/Observed_Secchi.csv",lakedir));
        depthsmodel = 0.5:size(model,2);
        depths = data{1,2:size(data,2)};
        initial_date = datetime(data{2,1},'Format','yyyy-MM-dd');
        if m_start < initial_date
            start_row = days(initial_date-m_start)-1;
        else
            start_row = 0;
        end
        
        z = 0;
        zz=0;
        dates = [];
        %try
            maxdepth = size(data,2);
            depths = data{1,2:size(data,2)};
            if(depths(1)< 0.5)
                min_depth_correct = 0;
                for i =1:size(depths,2)
                    if min_depth_correct == 1
                        break
                    else
                        depths(1) = [];
                        if (depths(1) < 0.5)
                            min_depth_correct = 1;
                        end
                    end
                end
            end
            if (depths(end) > depthsmodel(end))
                max_depth_correct = 0;
                for i = 1:size(depths,2)
                    if max_depth_correct == 1
                        break
                    else
                        depths(end) = [];
                        if (depths(end) < depthsmodel(end))
                            max_depth_correct = 1;
                        end
                    end
                end
            end
            temperature_model = [];
            temperature_data = [];
            temperature_depth =[];
            for i = start_row:size(model,1)
                date = m_start + i;

                for ii = 2+zz:height(data)
                    datedata =  datetime(data{ii,1},'Format','yyyy-MM-dd');
                    if date < datedata
                        break
                    elseif date > datedata
                        zz = zz+1;
                    else 
                        for jj = 1+z:size(depths,2)
                            if depths(jj) < 0.5
                                z = z+1;
                            elseif depths(jj) > depthsmodel(end)
                                    break
                            else
                                dates = [dates; datestr(date)];
                                temperature_data = [temperature_data data{ii,jj+1}];
                                temperature_depth = [temperature_depth depths(jj)];
                                depth = depths(jj);
                                if (ismember([depth],depthsmodel))==1
                                    temperature_model = [temperature_model model(i,round(depth))];
                                else
                                    ttt = depths(jj);
                                    tttt = (ceil(depths(jj))-0.5);
                                    if depths(jj) > (ceil(depths(jj))-0.5)
                                        test2 = ceil(depth);
                                        aa =(model(i,ceil(depth)));
                                        bb = (model(i,(ceil(depth)+1)));
                                        cc =(depthsmodel(ceil(depth)));
                                        ccc = depthsmodel(ceil(depth)+1);
                                        m = (model(i,ceil(depth)) - model(i,ceil(depth)+1))/(depthsmodel(ceil(depth))- depthsmodel(ceil(depth)+1));
                                        yc = (ceil(depth) - (depthsmodel(ceil(depth)+1))) * m + model(i,ceil(depth)+1);
                                        temperature_model = [temperature_model yc];
                                    else
                                        test2 = ceil(depth);
                                        aa =(model(i,(ceil(depth)-1)));
                                        bb = (model(i,(ceil(depth))));
                                        cc =(depthsmodel(ceil(depth)-1));
                                        ccc = depthsmodel(ceil(depth));
                                        m = (aa - bb)/(cc- ccc);
                                        yc = (ceil(depth) - depthsmodel(ceil(depth))) * m + (model(i,(ceil(depth))));
                                        temperature_model = [temperature_model yc];
                                    end

                                end
                            end
                        end
                    end
                end
            end
            formatOut = 'yyyy-mm-dd';
            datetest = datestr(dates,formatOut);
            datetest = string(datetest);

            A(:,1) = datetest;
            A(:,2) = temperature_depth;
            A(:,3) = temperature_data;
            A(:,4) = temperature_model;
            A = rmmissing(A);
            f1_name = (strcat(outdir, '\Tztcompare.csv')); % b = binary mode, z = archived file
            writematrix(A,f1_name);
            dates = transpose(datetest);
        %catch
        %    dates=[];
        %    temperature_model = [];
        %    temperature_data = [];
        %    temperature_depth =[];
        %end
        ModelComparison.Dates = dates;
        ModelComparison.Depth = temperature_depth;
        ModelComparison.T_data = temperature_data;
        ModelComparison.T_model = temperature_model;
        
        %secchi treatment
        z = 0;
        zz=0;
        dates = [];

        secchi_model = [];
        secchi_data = [];
        secchi_depth =[];
       % try
            ttt =size(modelS,1);
            if ttt > 0
                lambda = mean(modelS, 2);
                for i = start_row:size(modelS,1)
                    date = m_start + i;
                    aaa= height(dataS);
                    for ii = 2+zz:height(dataS)
                       datedataS =  datetime(dataS{ii,1},'Format','yyyy-MM-dd');
                       test = days(date - datedataS);
                       if date < datedataS
                           break
                       end
                       if date > datedataS
                           zz = zz+1;

                       elseif (date - datedataS) == 0
                           for jj = 2+z:size(dataS,2)
                               cc = size(dataS,2);
                               c = dataS{ii,jj};
                               try
                                   if isnan(dataS{ii,jj})
                                       continue
                                   else
                                       secchimodel = 1.48/lambda(ii);
                                       dates = [dates; datestr(date)];
                                       secchi_data = [secchi_data dataS{ii,jj}];
                                       secchi_depth = [secchi_depth 0.5];
                                       secchi_model = [secchi_model secchimodel];
                                   end
                               catch
                                   disp("gg");
                               end
                           end

                       else
                           continue
                       end
                    end
                end

            else
                datetest = [];
                secchi_depth = [];
                secchi_data = [];
                secchi_model = [];
            end
    %     catch
    %         disp("ee")
    %     end
        formatOut = 'yyyy-mm-dd';
        datetest = datestr(dates,formatOut);
        datetest = string(datetest);
        C(:,1) = datetest;
        C(:,2) = secchi_depth;
        C(:,3) = secchi_data;
        C(:,4) = secchi_model;
        C = rmmissing(C);

        f1_name = (strcat(outdir, '\Secchicompare.csv')); % b = binary mode, z = archived file
        writematrix(C,f1_name);

        ModelComparison.S_data = secchi_data;
        ModelComparison.S_model = secchi_model;
        %ModelComparison.S_data = [];
        %ModelComparison.S_model = [];
    %     error = (temperature_data-temperature_model);
    %     errorsqrt= error.^2;
    %     mean1 = nanmean(errorsqrt);
    %     rmse = sqrt(mean1);
    %     R = corrcoef(temperature_data,temperature_model,'rows','complete');
    %     R2 = (R.^2);
    %     evaluator = [rmse R2];


    elseif (variable_calibrated == "oxygen")
        modelO = O2zt.';
        modelO = modelO*0.001; %convert from mg/m*-3 to mg/l to have same unit as observed oxygen.
        dataO = readtable(sprintf("%s/Observed_Oxygen.csv",lakedir));
        depthsmodel = 0.5:size(modelO,2);
%         depths = dataO{1,2:size(dataO,2)};
%         initial_date = datetime(dataO{2,1},'Format','yyyy-MM-dd');
        if m_start < initial_date
            start_row = days(initial_date-m_start)-1;
        else
            start_row = 0;
        end
        %treatment oxygen
        z = 0;
        zz=0;
        datesO = [];
        try
            maxdepth = size(dataO,2);
            depthsO = dataO{1,2:size(dataO,2)};
            if(depthsO(1)< 0.5)
                min_depth_correct = 0;
                for i =1:size(depthsO,2)
                    if min_depth_correct == 1
                        break
                    else
                        depthsO(1) = [];
                        if (depthsO(1) < 0.5)
                            min_depth_correct = 1;
                        end
                    end
                end
            end
            if (depthsO(end) > depthsmodel(end))
                max_depth_correct = 0;
                for i = 1:size(depthsO,2)
                    if max_depth_correct == 1
                        break
                    else
                        depthsO(end) = [];
                        if (depthsO(end) < depthsmodel(end))
                            max_depth_correct = 1;
                        end
                    end
                end
            end
            oxygen_model = [];
            oxygen_data = [];
            oxygen_depth =[];
            for i = start_row:size(modelO,1)
                date = m_start + i;
                for ii = 2+zz:height(dataO)
                    datedata =  datetime(dataO{ii,1},'Format','yyyy-MM-dd');
                    if date < datedata 
                        break
                    elseif date > datedata
                        zz = zz+ 1;
                    else
                        for jj = 1 + z:size(depthsO,2)
                            if depthsO(jj) < 0.5
                                z = z+1;
                            elseif depthsO(jj) > depthsmodel(end)
                                      break

                            else
                                datesO = [datesO; datenum(date)];
                                oxygen_data = [oxygen_data dataO{ii,jj+1}];
                                oxygen_depth = [oxygen_depth depths(jj)];
                                depth = depths(jj); 
                                if (ismember([depth],depthsmodel))==1
                                    oxygen_model = [oxygen_model modelO(i,round(depth))];
                                else
                                    ttt = depths(jj);
                                    tttt = (ceil(depths(jj))-0.5);
                                    if depths(jj) > (ceil(depths(jj))-0.5)
                                        test2 = ceil(depth);
                                        aa =(modelO(i,ceil(depth)));
                                        bb = (modelO(i,(ceil(depth)+1)));
                                        cc =(depthsmodel(ceil(depth)));
                                        ccc = depthsmodel(ceil(depth)+1);
                                        m = (modelO(i,ceil(depth)) - modelO(i,ceil(depth)+1))/(depthsmodel(ceil(depth))- depthsmodel(ceil(depth)+1));
                                        yc = (ceil(depth) - (depthsmodel(ceil(depth)+1))) * m + modelO(i,ceil(depth)+1);
                                        oxygen_model = [oxygen_model yc];
                                    else
                                        test2 = ceil(depth);
                                        aa =(modelO(i,(ceil(depth)-1)));
                                        bb = (modelO(i,(ceil(depth))));
                                        cc =(depthsmodel(ceil(depth)-1));
                                        ccc = depthsmodel(ceil(depth));
                                        m = (modelO(i,(ceil(depth)-1))) - (modelO(i,(ceil(depth))))/((depthsmodel(ceil(depth)-1))- depthsmodel(ceil(depth)));
                                        yc = (ceil(depth) - depthsmodel(ceil(depth))) * m + (modelO(i,(ceil(depth))));
                                        oxygen_model = [oxygen_model yc];
                                    end
                                end               
                            end
                        end
                    end
                end

            end
%             datesO = [];
            formatOut = 'yyyy-mm-dd';
            datetestO = datestr(datesO,formatOut);
            datetestO = string(datetestO);
            B(:,1) = datetestO;
            B(:,2) = oxygen_depth;
            B(:,3) = oxygen_data;
            B(:,4) = oxygen_model;
            B = rmmissing(B);
            f1_name = (strcat(outdir, '\O2ztcompare.csv')); % b = binary mode, z = archived file
            writematrix(B,f1_name);
            datesO = transpose(datetestO);
        catch
            datesO = [];
            oxygen_depth = [];
            oxygen_data = [];
            oxygen_model = [];
        end    
        ModelComparison.DatesO = datesO;
        ModelComparison.DepthsO = oxygen_depth;
        ModelComparison.O_data = oxygen_data;
        ModelComparison.O_model = oxygen_model;
    end
 
end