function ModelComparison = compare_model_result_data(outdir, m_start2, m_stop2,Tzt,O2zt)
    %model = readtable(sprintf("%s/Tzt.csv",outdir));
    model = Tzt.';
    modelO = O2zt.';
    modelO = modelO*0.001; %convert from mg/m*-3 to mg/l to have same unit as observed oxygen.
    data = readtable(sprintf("%s/Observed_Temperature.csv",outdir));
    dataO = readtable(sprintf("%s/Observed_Oxygen.csv",outdir));
    depthsmodel = 0.5:size(model,2);
    depths = data{1,2:size(data,2)};
    m_start = datetime(sprintf('%d-01-01',m_start2),'Format','yyyy-MM-dd'); 
    m_stop =  datetime(sprintf('%d-12-31',m_stop2),'Format','yyyy-MM-dd'); 
    initial_date = datetime(data{2,1},'Format','yyyy-MM-dd');
    if m_start < initial_date
        start_row = days(initial_date-m_start)-1;
    else
        start_row = 0;
    end
    z = 0;
    zz=0;
    dates = [];
    
    temperature_model = [];
    temperature_data = [];
    temperature_depth =[];
    ttt =size(model,1);
    for i = start_row:size(model,1)
        date = m_start + i;
        aaa= height(data);
        for ii = 2+zz:height(data)
           datedata =  datetime(data{ii,1},'Format','yyyy-MM-dd');
           test = days(date - datedata);
           if date < datedata
               break
           end
           if date > datedata
               zz = zz+1;
               
           elseif (date - datedata) == 0
               ss = 1+z:size(depths,2);
               for jj = 1+z:size(depths,2)
                  d = depths(jj) ;
                  dd = depthsmodel(end);
                  if depths(jj) < 0.5
                    z = z+1;
                  elseif depths(jj) > depthsmodel(end)
                          break
                  else
                    dates = [dates; datenum(date)];
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
                            m = aa - bb/(cc- ccc);
                            yc = (ceil(depth) - depthsmodel(ceil(depth))) * m + (model(i,(ceil(depth))));
                            temperature_model = [temperature_model yc];
                        end
                        
                    end
                  end
               end
               zz = zz+1;
           else
               continue
           end
        end
    end
    datetest = string(dates);
   
    A(:,1) = datetest;
    A(:,2) = temperature_depth;
    A(:,3) = temperature_data;
    A(:,4) = temperature_model;
    dates = transpose(datetest);
    ModelComparison.Dates = dates;
    ModelComparison.Depth = temperature_depth;
    ModelComparison.T_data = temperature_data;
    ModelComparison.T_model = temperature_model;
    
    %treatment oxygen
    z = 0;
    zz=0;
    datesO = [];
    
    oxygen_model = [];
    oxygen_data = [];
    oxygen_depth =[];
    ttt =size(modelO,1);
    for i = start_row:size(modelO,1)
        date = m_start + i;
        aaa= height(dataO);
        for ii = 2+zz:height(dataO)
           datedata =  datetime(dataO{ii,1},'Format','yyyy-MM-dd');
           test = days(date - datedata);
           if date < datedata
               break
           end
           if date > datedata
               zz = zz+1;
              
           elseif (date - datedata) == 0
               ss = 1+z:size(depths,2);
               for jj = 1+z:size(depths,2)
                  d = depths(jj) ;
                  dd = depthsmodel(end);
                  if depths(jj) < 0.5
                    z = z+1;
                  elseif depths(jj) > depthsmodel(end)
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
                            m = aa - bb/(cc- ccc);
                            yc = (ceil(depth) - depthsmodel(ceil(depth))) * m + (modelO(i,(ceil(depth))));
                            oxygen_model = [oxygen_model yc];
                        end
                        
                    end
                  end
               end
               zz = zz+1;
           else
               continue
           end
        end
    end
    
    datetestO = string(datesO);
    
    B(:,1) = datetestO;
    B(:,2) = oxygen_depth;
    B(:,3) = oxygen_data;
    B(:,4) = oxygen_model;
    datesO = transpose(datetestO);
    ModelComparison.DatesO = datesO;
    ModelComparison.DepthO = oxygen_depth;
    ModelComparison.O_data = oxygen_data;
    ModelComparison.O_model = oxygen_model;

%     error = (temperature_data-temperature_model);
%     errorsqrt= error.^2;
%     mean1 = nanmean(errorsqrt);
%     rmse = sqrt(mean1);
%     R = corrcoef(temperature_data,temperature_model,'rows','complete');
%     R2 = (R.^2);
%     evaluator = [rmse R2];
    
    
    
end