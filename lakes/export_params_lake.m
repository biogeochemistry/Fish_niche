function export_params_lake(lake_params, parfile)
    basicfile = 'C:\Users\macot620\Documents\GitHub\Fish_niche\lakes\2017par.txt';    
    if exist(parfile, 'file') == 2
                % Read txt into cell A
        fid = fopen(parfile,'r');
        i = 1;
        tline = fgetl(fid);
        A{i} = tline;
        while ischar(tline)
            i = i+1;
            if( i == 25 || i== 27 || i == 49|| i == 50)
                j = i-2;
                tline = fgets(fid);
                sc = split(tline);
                cell = lake_params(j,1);
                tline = [sc{1},num2str(cell{1}),num2str(sc{3}),num2str(sc{4}),sc{5}];
            else
            tline = fgetl(fid);
            end
            A{i} = tline;
        end
        fclose(fid);
        fid = fopen(parfile, 'w');
        for i = 1:numel(A)
            if A{i+1} == -1
                fprintf(fid,'%s', A{i});
                break
            else
                fprintf(fid,'%s\n', A{i});
            end
        end 
    else
                % Read txt into cell A
        fid = fopen(basicfile,'r');
        i = 1;
        tline = fgetl(fid);
        A{i} = tline;
        while ischar(tline)
            i = i+1;
            if i == 25 | i== 27 | i == 49| i == 50
                j = i-2;
                tline = fgets(fid);
                sc = split(tline);
                cell = lake_params(j,1);
                tline = [sc{1},num2str(cell{1}),num2str(sc{3}),num2str(sc{4}),sc{5}];
            else
            tline = fgetl(fid);
            end
            A{i} = tline;
        end
        fclose(fid);
        fid = fopen(parfile, 'w');
        for i = 1:numel(A)
            if A{i+1} == -1
                fprintf(fid,'%s', A{i});
                break
            else
                fprintf(fid,'%s\n', A{i});
            end
        end 
         
    end
end
