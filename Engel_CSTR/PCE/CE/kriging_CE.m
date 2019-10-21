%% Author: Farshud Sorourifar 9/7/19
%% load packages and data
clearvars
uqlab

%Note: Yx and Yx_val are for future surrogates

%load training data
X = csvread('CE_training_Input.csv');
%X=X(1:300,:);
%X=[X(:,1:2),X(:,2+10*5:1+10*6)];
Y = csvread('CE_training_T.csv');
%Y=Y(1:300,:);

%load validation set
Xval =csvread('CE_val_Input.csv');
Xval = Xval(1:300,:);
%Xval=[Xval(:,1:2),Xval(:,2+10*5:1+10*6)];
Yval=csvread('CE_val_T.csv');
Yval = Yval(1:300,:);


%% input model

%theta 1
InputOpts.Marginals(1).Type = 'Gaussian';
InputOpts.Marginals(1).Moments = [2.5889*10^(-7)*120 ((2.5889*10^(-7)*120*.10))^2];

%theta 2
InputOpts.Marginals(2).Type = 'Gaussian';
InputOpts.Marginals(2).Moments = [-274.5925 (-274.5925*.10)^2];

%state measurment noise
V = [10^-17, 5, 5, 5, .2, .2, .2];
%measurment noise for 7 states at 10 sample times
for i = [1:10]
    for j = [1:7]+2
        %noise for xj
        InputOpts.Marginals(j+7*(i-1)).Type = 'Gaussian';
        InputOpts.Marginals(j+7*(i-1)).Moments = [0 (V(j-2)^2)];
        %InputOpts.Marginals(i+2).Type = 'Gaussian';
        %InputOpts.Marginals(i+2).Moments = [0 (V(5))^2];
                
    end
end 

%create input object
myInput = uq_createInput(InputOpts);

%% PCE Metamodel

% PCE modeling tool
MetaOpts.Type = 'Metamodel';
MetaOpts.MetaType = 'PCE';
%MetaOpts.Mode = 'sequential';
%MetaOpts.Kriging.Optim.Method = 'CMAES';

% load experiement data
MetaOpts.ExpDesign.X = X;
MetaOpts.ExpDesign.Y = Y;
% maximum polynomial degree
MetaOpts.Degree =1;
% load validation data for validation error
MetaOpts.ValidationSet.X = Xval;
MetaOpts.ValidationSet.Y = Yval;
%MetaOpts.Rank = 1;

%create metamodel object and add to UQLab and return results
myPCE = uq_createModel(MetaOpts);
uq_print(myPCE)

%% Validation
% evaluate at validation setpoints
YPCE = uq_evalModel(myPCE,Xval);

% plot Histogram of true outputs and PCE predictions 

uq_figure('Position', [50 50 500 400])
myColors = ['r','b']; % use the UQLab colormap
li = [0:2:18]; % use normalized positions

title('Kriging')
hold on
uq_plot(NaN,NaN,'Color',myColors(1));
uq_plot(NaN,NaN,'Color',myColors(2));
% Loop over the realizations
for ii = 1:size(Yval,1)
    p1=uq_plot(li, Yval(ii,:),'-', 'LineWidth', 1, 'Color', myColors(1)); 
    p2=uq_plot(li, YPCE(ii,:),'-', 'LineWidth', 1, 'Color', myColors(2)); 
    p1.Color(4) = 0.1;
    p2.Color(4) = 0.1;

end
hold off
ylabel('Temperature [K]')
xlabel('Time [min]')
legend('True', 'Surrogate')

uq_setInterpreters(gca)

% Print  validation and leave-one-out(LOO) cross-validation errors:
fprintf('PCE metamodel validation error: %5.4e\n', myPCE.Error.Val)
fprintf('PCE metamodel LOO error:        %5.4e\n', myPCE.Error.LOO)

