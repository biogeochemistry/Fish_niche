function [ O2z, Pz ] = update_wc( O2z, Pz, MyLake_params, sediment_diffusion_fluxes, sediment_bioirrigation_fluxes )
%UPDATE_WC Summary of this function goes here
    % Fluxes:
    O2z = update_C_due_to_flux(O2z, sediment_diffusion_fluxes{1}, MyLake_params);
    Pz = update_C_due_to_flux(Pz, sediment_diffusion_fluxes{4}, MyLake_params);

    % Bioirrigation:
    O2z = update_C_due_to_flux(O2z, sediment_bioirrigation_fluxes{1}, MyLake_params);
    % O2z = (O2z>0).*O2z;
    if O2z(end) < 0
        % disp('warning:Bioirrigation in sediments produce the negative concentration of O2z(end) in WC');
        O2z(end) = 0;
    end
end

%% update_C_due_to_flux: Update concentration of the WC due to flux from/to sediment
function [C] = update_C_due_to_flux(C, flux, MyLake_params)
    Az_end = MyLake_params('Az(end)');
    Vz_end = MyLake_params('Vz(end)');
    dt     = MyLake_params('dt');

    C(end) = C(end) - flux * dt * Az_end / Vz_end;
end


