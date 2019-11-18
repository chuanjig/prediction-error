function [AIC_RW, BIC_RW, AIC_PH, BIC_PH, alph_est, alph0_est, gamm_est, ...
    cost_RW, cost_PH] = pesd_fits_scaling(data,con, ntrials, Nsub)

SD = repmat([5 10 15 5 10 15],1,Nsub); EV = repmat([35 35 35 65 65 65],1,Nsub); 

% Data
Nc = size(con,2);
mu0 = EV(con);
sd0 = SD(con);
P = reshape(data(:,1,con),size(data,1),Nc); % predictions
R = reshape(data(:,2,con),size(data,1),Nc); % rewards
fprintf('SD = %4.1f EV = %2.1f\n',[sd0; mu0])

P = P(1:ntrials,:);
R = R(1:ntrials,:);

% RW fit
mu_0    = P(1,:);
s2_RW_N = 100;
alph_RW = .5;
x    = R(1:end-1,:);
data = P(2:end,:)';
aux  = [sd0; mu_0];
cf   = @(F,x,aux,data) -cf_RW(F,x,aux,data);          % -log(likelihood) cost function
F = [alph_RW s2_RW_N];                                % Initialize
lb  = [0  1];                                         % Lower bounds
ub  = [1 20];                                         % Upper bounds
opt = optimset('Display','off','TolX',1e-6,'TolFun',1e-9,'algorithm','sqp','FunValCheck','on');
warning off 
[F_fit, cost] = fmincon(cf,F,[],[],[],[],lb,ub,[],opt,x,aux,data); warning on 
cf(F_fit,x,aux,data); % check for zero likelihood
alph_est    = F_fit(1);
sigma_N_est = F_fit(2);
k = numel(F_fit) - 1;
M = numel(P); % number of data
cost_RW = cost;
AIC_RW = 2*(k + cost);
BIC_RW = k*log(M) + 2*cost;
fprintf('alph_est  = %4.1f - sigma_N_est = %4.1f - AIC_RW    = %4.1f\n',alph_est,sigma_N_est,AIC_RW)

% PH fit
mu_0  = P(1,:);
sN_PH = 10;
alph0 = alph_est;
gamm  = 0.0;
x     = R(1:end-1,:);
data  = P(2:end,:)';
aux  = [sd0; mu_0];
cf    = @(F,x,aux,data) -cf_PH(F,x,aux,data);         % -log(likelihood) cost function
F     = [alph0 gamm sN_PH];                           % Initialize
lb    = [    0    0     1];                           % Lower bounds
ub    = [    1    1    15];                           % Upper bounds
opt = optimset('Display','off','TolX',1e-6,'TolFun',1e-9,'algorithm','sqp','FunValCheck','on');
warning off 
[F_fit, cost] = fmincon(cf,F,[],[],[],[],lb,ub,[],opt,x,aux,data); warning on 
cf(F_fit,x,aux,data); % check for zero likelihood
alph0_est     = F_fit(1);
gamm_est      = F_fit(2);
sigma_N_est   = F_fit(3);
k = numel(F_fit) - 1;
M = numel(P); % number of data
cost_PH = cost;
AIC_PH = 2*(k + cost);
BIC_PH = k*log(M) + 2*cost;
fprintf('alph0_est = %4.1f - gamm_est    = %4.2f - sigma_N_est = %4.1f - AIC_PH = %4.1f\n',alph0_est,gamm_est,sqrt(sigma_N_est),AIC_PH)

end

%%

function [mu_n, l] = gm_RW(P,x,aux,data)

% Init
Nc = size(x,2);                 % Number of conditions
alph     = P(1);                % Learning rate
sigma2_n = P(2)^2;              % Variance of posterior mean
l        = nan(size(x));        % Likelihood
s2       = nan(size(x));        % Variance of likelihood
mu_n     = nan(size(x));        % Posterior mean
SD       = aux(1,:);            % The actual SDs

for c = 1:Nc
    
    % Process
    mu_pri = aux(2,c);          % Prior
    keep = find(~isnan(x(:,c)));
    Ni   = numel(keep);
    subsd    = log(SD(c))/log(25);
     
    for ind = 1:Ni
        
        % Index
        n = keep(ind);
        
        % Observation
        mu_ML = x(n,c);
        PE    = (mu_ML - mu_pri); % Prediction error
        
        % Posterior estimate
        mu_n(n,c)  = mu_pri + alph*(PE/subsd);
        
        % Update prior
        mu_pri     = mu_n(n,c);
        
        % Posterior likelihood of the data
        if exist('data','var')
            if ind == 1
                s2(n,c) = sigma2_n;
            else
                %                 n_1 = keep(ind-1);
                %                 s2(n,c) = sigma2_n + ((1-alph).^2)*s2(n_1,c);
                s2(n,c) = sigma2_n;
            end
            l(n,c) = normpdf( data(c,n)', mu_n(n,c), sqrt(s2(n,c)) );
        end
        
    end
    
end

end

function ll = cf_RW(P,x,aux,data)

[~, l] = gm_RW(P,x,aux,data);
bad = find(l==0);
Nb  = numel(bad);
wst = min(min(l(l~=0)));
if Nb > 0
    warning('Replacing %d zero-likelihood value(s) with %2.1e',Nb,wst)
    l(bad) = wst;
end
ll = numel(data)*nanmean(nanmean(log(l)),2); % Total log-likelihood of the data

end

%%
function [mu_n, l] = gm_PH(P,x,aux,data)

% Init
Nc = size(x,2);                 % Number of conditions
alph_0   = P(1);                % Initial learning rate
gamm     = P(2);                % Learning decay
sigma2_n = P(3)^2;              % Variance of posterior mean
l        = nan(size(x));        % Likelihood
s2       = nan(size(x));        % Variance of likelihood
mu_n     = nan(size(x));        % Posterior mean
alph_PH  = nan(size(x));        % PH learning rate
SD       = aux(1,:);            % The actual SDs

for c = 1:Nc
    
    % Process
    mu_pri = aux(2,c);          % Prior
    keep = find(~isnan(x(:,c)));
    Ni   = numel(keep);
    subsd    = log(SD(c))/log(25);
    
    % Init
    n1 = keep(1);
    PE = x(n1,c) - mu_pri;      % Prediction error
    alph_PH(n1,c) = gamm*abs(((PE/100)/subsd)) + (1-gamm)*(alph_0);
    mu_n(n1,c)    = mu_pri + alph_PH(n1,c)*(PE/subsd);
    mu_pri        = mu_n(n1,c);  % Update prior
    if exist('data','var')
        s2(n1,c) = sigma2_n;
        l(n1,c)  = normpdf(data(c,n1), mu_n(n1,c), sqrt(s2(n1,c)));
    else
        l = [];
    end
    
    for ind = 2:Ni
        % Index
        n = keep(ind);
        n_1 = keep(ind-1);
        
        % Observation
        mu_ML = x(n,c);
        PE    = mu_ML - mu_pri;  % Prediction error
        
        % Posterior estimate
        alph_PH(n,c) = gamm*abs(((PE/100)/subsd)) + (1-gamm)*(alph_PH(n_1,c));
        gain = alph_PH(n,c);
        mu_n(n,c)    = mu_pri + gain*PE;
        
        % Update prior
        mu_pri     = mu_n(n,c);
        
        % Posterior likelihood of the data
        if exist('data','var')
            %             s2(n,c) = sigma2_n + ((1-gain).^2)*s2(n_1,c);
            s2(n,c) = sigma2_n;
            l(n,c)  = normpdf( data(c,n)', mu_n(n,c), sqrt(s2(n,c)) );
        end
        
    end
    
end

end

function ll = cf_PH(P,x,aux,data)

[~, l] = gm_PH(P,x,aux,data);
bad = find(l==0);
Nb  = numel(bad);
wst = min(min(l(l~=0)));
if Nb > 0
    warning('Replacing %d zero-likelihood value(s) with %2.1e',Nb,wst)
    l(bad) = wst;
end
ll = numel(data)*nanmean(nanmean(log(l)),2); % Total log-likelihood of the data

end
