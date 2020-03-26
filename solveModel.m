%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DFBAlab: Dynamic Flux Balance Analysis laboratory                       %
% Process Systems Engineering Laboratory, Cambridge, MA, USA              %
% November 2014                                                           %
% Written by Jose A. Gomez                                                %
%                                                                         % 
% This code can only be used for academic purposes. When using this code  %
% please cite:                                                            %
%                                                                         %
% Gomez, J.A., Höffner, K. and Barton, P. I. (2014).                      %
% DFBAlab: A fast and reliable MATLAB code for Dynamic Flux Balance       %
% Analysis. BMC Bioinformatics, 15:409                                    % 
%                                                                         %
% COPYRIGHT (C) 2014 MASSACHUSETTS INSTITUTE OF TECHNOLOGY                %
%                                                                         %
% Read the LICENSE.txt file for more details.                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modifed by Jin Chen 20150119

function [flux,penalty] = solveModel(t,y,INFO)
[lbx,ubx]=RHS(t,y,INFO);
ncost = INFO.ncost;
lexID = INFO.lexID;
nmodel = INFO.nmodel;
tolPh1 = INFO.tolPh1;
bmodel = INFO.b;
sense = INFO.sense;
lbct = INFO.lbct;
plambda = INFO.plambda;
indlb = INFO.indlb;
indub = INFO.indub;

penalty=(zeros(nmodel,1));
flux = zeros(nmodel,max(ncost)-1);
for i=1:nmodel
    b = bmodel{i};
    lb = lbx(i,1:lexID(i));
    ub = ubx(i,1:lexID(i));
    lb(indlb{i}) = [];
    ub(indub{i})=[];
    b(1:length(lb)) = lb;
    b(length(lb)+lbct(i)+1:length(lb)+lbct(i)+length(ub)) = ub;
    
    j=2:ncost(i);
    flux(i,j-1) = sense{i}(j).*(b'*plambda{i}(:,j));
    
    penalty(i) = plambda{i}(:,1)'*b;
    if penalty(i)<tolPh1
       penalty(i) = 0;
    end
end


end

