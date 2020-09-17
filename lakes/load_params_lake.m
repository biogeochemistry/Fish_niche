%% load_params_lake: load parameters for MyLake 
function [lake_params] = load_params_lake(longitude,latitude)

lake_params = {
    % PhysPar
    0.5, 'dz',                 % 1
    NaN, 'Kz_ak',              % 2     open water diffusion parameter (-)
    0.000898, 'Kz_ak_ice',     % 3     under ice diffusion parameter (-)
    7E-05, 'Kz_N0',            % 4     min. stability frequency (s-2)
    NaN, 'C_shelter',          % 5     wind shelter parameter (-)
    latitude, 'latitude',      % 6     latitude (decimal degrees)
    longitude, 'longitude',    % 7     longitude (decimal degrees)
    0.3, 'alb_melt_ice',       % 8     albedo of melting ice (-)
    0.4, 'alb_melt_snow',      % 9     albedo of melting snow (-)
    2.7834E-04, 'PAR_sat',     % 10    PAR saturation level for phytoplankton growth (mol(quanta) m-2 s-1)
    0.45, 'f_par',             % 11    Fraction of PAR in incoming solar radiation (-)
    0.015, 'beta_chl',         % 12    Optical cross_section of chlorophyll (m2 mg-1)
    5, 'lambda_I',             % 13    PAR light attenuation coefficient for ice (m-1)
    15, 'lambda_s',            % 14    PAR light attenuation coefficient for snow (m-1)
    0.05, 'F_sed_sld',         % 15    volume fraction of solids in sediment (= 1-porosity)
    1, 'I_scV',                % 16    scaling factor for inflow volume (-)
    0, 'I_scT',                % 17    adjusting delta for inflow temperature (-)
    1, 'I_scC',                % 18    scaling factor for inflow concentration of C (-)
    1, 'I_scS',                % 19
    1, 'I_scTP',               % 20    scaling factor for inflow concentration of total P (-)
    1, 'I_scDOP',              % 21    scaling factor for inflow concentration of diss. organic P (-)
    1, 'I_scChl',              % 22    scaling factor for inflow concentration of Chl a (-)
    2, 'I_scDOC',              % 23    scaling factor for inflow concentration of DOC  (-)
    2.5, 'swa_b0',             % 24     non-PAR light attenuation coeff. (m-1)
    1.05, 'swa_b1',            % 25     PAR light attenuation coeff. (m-1)
    3.30E-07, 'S_res_epi',     % 26     Particle resuspension mass transfer coefficient, epilimnion (m day-1, dry)
    3.30E-08, 'S_res_hypo',    % 27     Particle resuspension mass transfer coefficient, hypolimnion (m day-1, dry)
    0.03, 'H_sed',             % 28     height of active sediment layer (m, wet mass)
    2500, 'Psat_Lang',         % 29     NOTE: NOT USED: Half saturation parameter for Langmuir isotherm
    8000, 'Fmax_Lang',         % 30     NOTE: NOT USED: Scaling parameter for Langmuir isotherm !!!!!!!!!!!!
    0.1, 'Uz_Sz',              % 31     settling velocity for S (m day-1)
    0.0133, 'Uz_chl',          % 32     settling velocity for Chl a (m day-1)
    1, 'Y_cp',                 % 33     NOTE: NOT USED:  yield coefficient (chlorophyll to carbon) * (carbon to phosphorus) ratio (-)   1/55*112/1 = 1
    0.0342, 'm_twty',          % 34    loss rate (1/day) at 20 deg C
    1.4476, 'g_twty',          % 35    specific growth rate (1/day) at 20 deg C
    2.00E-04, 'k_sed_twty',    % 36    NOTE: NOT USED: specific Chl a to P transformation rate (1/day) at 20 deg C
    0, 'k_dop_twty',           % 37    NOTE: NOT USED: specific DOP to P transformation rate (day-1) at 20 deg C
    1.0594, 'P_half',          % 38    Half saturation growth P level (mg/m3)
    2.7834E-04, 'PAR_sat2',    % 39    PAR saturation level for phytoplankton growth (mol(quanta) m-2 s-1)
    0.015, 'beta_chl2',        % 40    Optical cross_section of chlorophyll (m2 mg-1)
    0.1939, 'Uz_Chl2',         % 41    Settling velocity for Chl a (m day-1)
    0.0342, 'm_twty2',         % 42    Loss rate (1/day) at 20 deg C
    0.8736, 'g_twty2',         % 43    Specific growth rate (1/day) at 20 deg C
    0.0062, 'P_half2',         % 44    Half saturation growth P level (mg/m3)
    0.01, 'oc_DOC',            % 45    Optical cross-section of DOC (m2/mg DOC)
    0.1, 'qy_DOC',             % 46    Quantum yield (mg DOC degraded/mol quanta)
    0.01, 'k_BOD',             % 47    NOTE: NOT USED: Organic decomposition rate (1/d)
    1, 'k_SOD',                % 48
    1.047, 'theta_BOD',        % 49    NOTE: NOT USED: Temperature adjustment coefficient for BOD, T ? 10 °C
    1.13, 'theta_BOD_ice',     % 50    NOTE: NOT USED: Temperature adjustment coefficient for BOD, T < 10 °C
    1, 'theta_SOD',            % 51    NOTE: NOT USED: Temperature adjustment coefficient for SOD, T ? 10 °C
    1, 'theta_SOD_ice',        % 52    NOTE: NOT USED: Temperature adjustment coefficient for SOD, T < 10 °C
    4, 'theta_T',              % 53    NOTE: NOT USED: Threshold for bod or bod_ice °C
    7.5, 'pH',                 % 54    Lake water pH
    1,'I_scDIC',               % 55
    100,'Mass_Ratio_C_Chl',    % 56
    0.25,'SS_C',               % 57
    1.95,'density_org_H_nc',   % 58
    2.65,'density_inorg_H_nc', % 59
    1,'I_scO',                 % 60
    }; 
end