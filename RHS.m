%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DFBAlab: Dynamic Flux Balance Analysis laboratory                       %
% Process Systems Engineering Laboratory, Cambridge, MA, USA              %
% July 2014                                                               %
% Written by Jose A. Gomez and Kai H�ffner                                %
%                                                                         % 
% This code can only be used for academic purposes. When using this code  %
% please cite:                                                            %
%                                                                         %
% Gomez, J.A., H�ffner, K. and Barton, P. I.                              %
% DFBAlab: A fast and reliable MATLAB code for Dynamic Flux Balance       %
% Analysis. Submitted.                                                    %
%                                                                         %
% COPYRIGHT (C) 2014 MASSACHUSETTS INSTITUTE OF TECHNOLOGY                %
%                                                                         %
% Read the LICENSE.txt file for more details.                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [lb,ub] = RHS(t,y,INFO )

% This subroutine updates the upper and lower bounds for the fluxes in the
% exID arrays in main. The output should be two matrices, lb and ub. The lb matrix
% contains the lower bounds for exID{i} in the ith row in the same order as
% exID. The same is true for the upper bounds in the ub matrix.
% Infinity can be used for unconstrained variables, however, it should be 
% fixed for all time. 

% Assign value

nmodel = INFO.nmodel;
ns = INFO.ns;
N = INFO.N;
param = INFO.param;

v_cm = param(1);
Kc   = param(2);
v_c2m = param(3);
Kc2   = param(4);
Kic = param(5);
Emax = 60;
Amax = 20;

j=1:nmodel;
cl(j) = y(3+(j-1)*ns);
c2l(j) = y(4+(j-1)*ns);

for j = 1:nmodel
    
    % Biomass
    lb(j,1) = 0;
    ub(j,1) = Inf;
    
    % CO
    lb(j,2) = -v_cm*max([cl(j) 0])/(Kc + cl(j) + cl(j)^2/Kic)*max([(1 - y(2+ns*nmodel)/Amax) 0])*max([(1 - y(3+ns*nmodel)/Emax) 0]);;
    ub(j,2) = 0;

    % CO2 
    lb(j,3) = -v_c2m*max([c2l(j) 0])/(Kc2 + c2l(j))*max([(1 - y(2+ns*nmodel)/Amax) 0])*max([(1 - y(3+ns*nmodel)/Emax) 0]);
    ub(j,3) = Inf;
    
    % Acetate
    lb(j,4) = 0;
    ub(j,4) = Inf;
    
    % Ethanol
    lb(j,5) = 0;
    ub(j,5) = Inf;
    
    % 23BDO
    lb(j,6) = 0;
    ub(j,6) = Inf;
    
    % Lactate
    lb(j,7) = 0;
    ub(j,7) = Inf;
    
    % Proton
    lb(j,8) = 0;
    ub(j,8) = 3.3;
    
end

end

