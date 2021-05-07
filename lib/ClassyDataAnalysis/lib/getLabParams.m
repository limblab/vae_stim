function [fhcal,rotcal,Fy_invert, forceOffsets]=getLabParams(labnum,dateTime,rothandle)
    %wrapper function to contain the logic that selects load cell calibrations
    %and rotation data. Intended for use when converting robot handle load cell
    %data from raw form into room coordinates. Provides load cell calibration
    %matrices, rotation matrices, and a variable used to invert the y axis in
    %the case of data where the load cell was installed upside down. This
    %function is called when preprocessing data as it is loaded into
    %matlab. It is not intended for end user use. Measured force offsets
    %for lab 6 are included as an output, but not handled in calling
    %functions.
    %
    %Each lab with a robot should have a block dedicated to it
    %every time the lab 
    forceOffsets = [];
    if labnum==3 %If lab3 was used for data collection
        % Check date of recording to see if it's before or after the
        % change to force handle mounting.
        if datenum(dateTime) < datenum('5/27/2010')            
            fhcal = [ 0.1019 -3.4543 -0.0527 -3.2162 -0.1124  6.6517; ...
                     -0.1589  5.6843 -0.0913 -5.8614  0.0059  0.1503]';
            rotcal = [0.8540 -0.5202; 0.5202 0.8540];                
            Fy_invert = -1; % old force setup was left hand coordnates.
        elseif datenum(dateTime) < datenum('6/28/2011')
            fhcal = [0.0039 0.0070 -0.0925 -5.7945 -0.1015  5.7592; ...
                    -0.1895 6.6519 -0.0505 -3.3328  0.0687 -3.3321]';
            rotcal = [1 0; 0 1];                
            Fy_invert = 1;
        else
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 3\FT7520.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
            fhcal = [-0.0129 0.0254 -0.1018 -6.2876 -0.1127 6.2163;...
                    -0.2059 7.1801 -0.0804 -3.5910 0.0641 -3.6077]'./1000;
            
            Fy_invert = 1;
            if rothandle
                rotcal = [-1 0; 0 1];  
            else
                rotcal = [1 0; 0 1];  
            end
        end
    elseif labnum==2 %if lab2 was used for data collection
        warning('calc_from_raw_script:Lab2LoadCellCalibration','No one noted what the calibration for the Lab2 robot was, so this processing assumes the same parameters as the original LAB3 values. THE FORCE VALUES RESULTING FROM THIS ANALYSIS MAY BE WRONG!!!!!!!!!!!!!!')
        if datenum(dateTime) < datenum('5/27/2010')            
            fhcal = [ 0.1019 -3.4543 -0.0527 -3.2162 -0.1124  6.6517; ...
                     -0.1589  5.6843 -0.0913 -5.8614  0.0059  0.1503]';
            rotcal = [0.8540 -0.5202; 0.5202 0.8540];                
            Fy_invert = -1; % old force setup was left hand coordnates.
        elseif datenum(out_struct.meta.datetime) < datenum('6/28/2011')
            fhcal = [0.0039 0.0070 -0.0925 -5.7945 -0.1015  5.7592; ...
                    -0.1895 6.6519 -0.0505 -3.3328  0.0687 -3.3321]';
            rotcal = [1 0; 0 1];                
            Fy_invert = 1;
        elseif rothandle
            %included this section for consistency. Old Lab2 files 
            %would never have used a rotated handle
            error('calc_from_raw_script:Lab2RotHandle','the rotate handle option was never used in Lab2. If lab2 has been updated with a loadcell and you are using the handle in a rotated position you need to modify raw2handleforce to reflect this')
        end
    elseif labnum==6 %If lab6 was used for data collection
        if datenum(dateTime) < datenum('5/27/2010')            
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 6\FT16018.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
        elseif datenum(dateTime) < datenum('07-Mar-2016')
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 6\FT16018.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
            fhcal = [0.02653 0.02045 -0.10720 5.94762 0.20011 -6.12048;...
                    0.15156 -7.60870 0.05471 3.55688 -0.09915 3.44508;...
                    10.01343 0.36172 10.30551 0.39552 10.46860 0.38238;...
                    -0.00146 -0.04159 0.14436 0.02302 -0.14942 0.01492;...
                    -0.16542 -0.00272 0.08192 -0.03109 0.08426 0.03519;...
                    0.00377 -0.09455 0.00105 -0.08402 0.00203 -0.08578]'./1000;
            Fy_invert = 1;
            rotcal = eye(6);
            forceOffsets = [];
        elseif datenum(dateTime) < datenum('17-Jul-2017')
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 6\FT16018.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
            fhcal = [0.02653 0.02045 -0.10720 5.94762 0.20011 -6.12048;...
                    0.15156 -7.60870 0.05471 3.55688 -0.09915 3.44508;...
                    10.01343 0.36172 10.30551 0.39552 10.46860 0.38238;...
                    -0.00146 -0.04159 0.14436 0.02302 -0.14942 0.01492;...
                    -0.16542 -0.00272 0.08192 -0.03109 0.08426 0.03519;...
                    0.00377 -0.09455 0.00105 -0.08402 0.00203 -0.08578]'./1000;
                
            Fy_invert = 1;
            % rotation of the load cell to match forearm frame
            % (load cell is upside down and slightly rotated)
            theta_off = atan2(3,27); %angle offset of load cell to forearm frame- 3 and 27 are the empirircal measures used to generate the angle
%                         theta_off = 0;
            rotcal = [-cos(theta_off) -sin(theta_off) 0    0             0              0;...
                      -sin(theta_off) cos(theta_off)  0    0             0              0;...
                      0                 0             1    0             0              0;...
                      0                 0             0 -cos(theta_off) -sin(theta_off) 0;...
                      0                 0             0 -sin(theta_off) cos(theta_off)  0;...
                      0                 0             0    0             0              1]'; 
            forceOffsets = [-240.5144  245.3220 -103.0073 -567.6240  332.3762 -591.9336]; %measured 3/17/16
%                         force_offsets = [];
        elseif datenum(dateTime) < datenum('08-Sep-2017')
            % replaced lab 6 load cell + handle with lab 3 load cell + handle
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 3\FT7520.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.

            fhcal = [-0.06745   0.13235  -0.53124 -32.81043  -0.58791  32.43832;...
                    -1.07432  37.46745  -0.41935 -18.73869   0.33458 -18.82582;...
                    -18.56153   1.24337 -18.54582   0.85789 -18.70268   0.63662;...
                    -0.14634   0.36156 -31.67889   0.77952  32.39412  -0.81438;...
                    36.65668  -1.99599 -19.00259   0.79078 -18.87751   0.31411;...
                    -0.31486  18.88139   0.09343  18.96202  -0.46413  18.94001]'./...
                    repmat([5.218 5.218 1.772 217.518 217.518 217.669],6,1)./1000;
            
            Fy_invert = 1;
            if rothandle
                rotcal = diag([-1 1 1 -1 1 1]);  
            else
                rotcal = eye(6);  
            end
        elseif datenum(dateTime) < datenum('09-Mar-2018')
            % replaced handle with old lab 6 handle and lab 3 load cell
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 3\FT7520.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.

            fhcal = [-0.06745   0.13235  -0.53124 -32.81043  -0.58791  32.43832;...
                    -1.07432  37.46745  -0.41935 -18.73869   0.33458 -18.82582;...
                    -18.56153   1.24337 -18.54582   0.85789 -18.70268   0.63662;...
                    -0.14634   0.36156 -31.67889   0.77952  32.39412  -0.81438;...
                    36.65668  -1.99599 -19.00259   0.79078 -18.87751   0.31411;...
                    -0.31486  18.88139   0.09343  18.96202  -0.46413  18.94001]'./...
                    repmat([5.218 5.218 1.772 217.518 217.518 217.669],6,1)./1000;
            
            Fy_invert = 1;
            % rotation of the load cell to match forearm frame
            % (load cell is upside down and slightly rotated)
            theta_off = atan2(3,27); %angle offset of load cell to forearm frame- 3 and 27 are the empirircal measures used to generate the angle
%                         theta_off = 0;
            rotcal = [-cos(theta_off) -sin(theta_off) 0    0             0              0;...
                      -sin(theta_off) cos(theta_off)  0    0             0              0;...
                      0                 0             1    0             0              0;...
                      0                 0             0 -cos(theta_off) -sin(theta_off) 0;...
                      0                 0             0 -sin(theta_off) cos(theta_off)  0;...
                      0                 0             0    0             0              1]'; 

        elseif datenum(dateTime)<datenum('10-Jun-2018')
            % replaced load cell with new lab 6 load cell and amp
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 6\New load cell (20180309)\FT23102.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.

            fhcal = [  0.09275   0.07076  -0.38368   5.92817   0.18265  -6.27042;...
                     -0.01335  -7.47013  -0.09448   3.46810  -0.14599   3.55623;...
                     10.21848   0.37747  10.47247   0.35487  10.47848  -0.03876;...
                      0.00337  -0.04040   0.15030   0.02426  -0.14815   0.01946;...
                     -0.16856  -0.00689   0.08967  -0.02890   0.08321   0.03386;...
                      0.00221  -0.09029   0.00410  -0.08202   0.00206  -0.08720]'./1000;
            
            Fy_invert = 1;
            % rotation of the load cell to match forearm frame
            % (load cell is upside down and slightly rotated)
            theta_off = atan2(3,27); %angle offset of load cell to forearm frame- 3 and 27 are the empirircal measures used to generate the angle
%                         theta_off = 0;
            rotcal = [-cos(theta_off) -sin(theta_off) 0    0             0              0;...
                      -sin(theta_off) cos(theta_off)  0    0             0              0;...
                      0                 0             1    0             0              0;...
                      0                 0             0 -cos(theta_off) -sin(theta_off) 0;...
                      0                 0             0 -sin(theta_off) cos(theta_off)  0;...
                      0                 0             0    0             0              1]'; 

        elseif datenum(dateTime)<datenum('01-Nov-2018')
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 3\FT7520.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
            fhcal = [-0.06745   0.13235  -0.53124 -32.81043  -0.58791  32.43832;...
                    -1.07432  37.46745  -0.41935 -18.73869   0.33458 -18.82582;...
                    -18.56153   1.24337 -18.54582   0.85789 -18.70268   0.63662;...
                    -0.14634   0.36156 -31.67889   0.77952  32.39412  -0.81438;...
                    36.65668  -1.99599 -19.00259   0.79078 -18.87751   0.31411;...
                    -0.31486  18.88139   0.09343  18.96202  -0.46413  18.94001]'./...
                    repmat([5.218 5.218 1.772 217.518 217.518 217.669],6,1)./1000;
            Fy_invert = 1;
            if rothandle
                rotcal = diag([-1 1 1 -1 1 1]);  
            else
                rotcal = eye(6);  
            end
        elseif datenum(datetime)< datenum('01-Nov-2019')
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab
            % 6\FT25831.cal (this was sent back for recalibration and
            % received 20180901)
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
           
            fhcal = [  0.54394   0.05569  -0.95964   33.11326   0.40244  -33.79878;...
                     -1.22429 -39.12317  0.02792   18.80600  -0.33459   19.53948;...
                     17.87880   -0.32145  19.05080   -0.1956  18.23279  0.16308;...
                      0.35868  -0.10436   32.59968   0.24722  -32.92496   -0.76043;...
                     -36.11741  0.12455   18.86971  -0.16965   18.26433   0.66712;...
                      -0.30954 -19.51322   0.18811  -19.05310   0.48341  -19.31874]'./...
                      repmat([10.4365219 10.4365219 3.54463927 435.0360510 435.0360510 435.3391356],6,1)./1000;
            
            Fy_invert = -1;
            % rotation of the load cell to match forearm frame
            % (load cell is upside down and slightly rotated)
            theta_off = deg2rad(30); %angle offset of load cell to forearm frame- 3 and 27 are the empirircal measures used to generate the angle
%                         theta_off = 0;
            rotcal = [-cos(theta_off) -sin(theta_off) 0    0             0              0;...
                      -sin(theta_off) cos(theta_off)  0    0             0              0;...
                      0                 0             1    0             0              0;...
                      0                 0             0 -cos(theta_off) -sin(theta_off) 0;...
                      0                 0             0 -sin(theta_off) cos(theta_off)  0;...
                      0                 0             0    0             0              1]'; 
            
        else
            % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab
            % 6\FT25831.cal (this was sent back for recalibration and
            % received 20180901)
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
           
           % Fx,Fy,scaleX,scaleY from ATI calibration file:
            % \\citadel\limblab\Software\ATI FT\Calibration\Lab 6\FT16018.cal
            % fhcal = [Fx;Fy]./[scaleX;scaleY]
            % force_offsets acquired empirically by recording static
            % handle.
            fhcal = [0.02653 0.02045 -0.10720 5.94762 0.20011 -6.12048;...
                    0.15156 -7.60870 0.05471 3.55688 -0.09915 3.44508;...
                    10.01343 0.36172 10.30551 0.39552 10.46860 0.38238;...
                    -0.00146 -0.04159 0.14436 0.02302 -0.14942 0.01492;...
                    -0.16542 -0.00272 0.08192 -0.03109 0.08426 0.03519;...
                    0.00377 -0.09455 0.00105 -0.08402 0.00203 -0.08578]'./1000;
            Fy_invert = 1;
            rotcal = eye(6);
            forceOffsets = [];
        end   
        if rothandle
            error('getLabParams:HandleRotated','Handle rotation not implemented for lab 6')  
        end
    else
        error('getLabParams:BadLab',['lab: ',labnum,' is not configured properly']);
    end
end
