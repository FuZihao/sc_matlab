

% load in the digit database (only needs to be done once per session)
if ~(exist('train_data')&exist('label_train'))
   load digit_100_train_easy;
%   load digit_100_train_hard;
end

% choose which two digits to compare:
mm=26;
nn=27;

display_flag = 1;
affine_start_flag = 1;
polarity_flag = 1;
nsamp = 100; % sample number
eps_dum = 0.25; % outlier threhold
ndum_frac = 0.25;        
mean_dist_global = [];
ori_weight = 0.1;
nbins_theta = 12;
nbins_r = 5;
r_inner = 1/8;
r_outer = 2;
tan_eps = 1.0;
n_iter = 3;
beta_init = 1;
r = 1; % annealing rate
w = 4;
sf = 2.5;

% score weight
W_ac = 1.6;
W_dc = 1;
W_be = 0.3;

%%%
%%% image loading
%%%
model = reshape(train_data(mm,:),28,28)';
model = imresize(model,sf,'bil');
[N1, N2] = size(model);


[mx, my, tm] = bdry_extract_3(model);
nsamp_model = length(mx);
if nsamp_model >= nsamp
    [mx,my,tm] = get_samples_1(mx, my, tm, nsamp);
else
    error('transformed model doesn''t have enough samples')
end

% input from demo_2
M = X;
tm = t1;

target = reshape(train_data(nn,:),28,28)';
target = imresize(target,sf,'bil');
[tx, ty, tt] = bdry_extract_3(target);
nsamp_target = length(tx);
if nsamp_target >= nsamp
    [tx, ty, tt] = get_samples_1(tx, ty, tt, nsamp);
else
    error('target model doesn''t have enough samples');
end

% input from demo_2
T = Y;
tt = t2;

% compute correspondences
M_k = M;
tm_k = tm;
k = 1;
s = 1;
ndum = round(ndum_frac * nsamp);
out_vec_model = zeros(1, nsamp);
out_vec_target = zeros(1, nsamp);

costs = 0;
while s
    disp(['iter=' int2str(k)]);
    disp('computing shape contexts for transformed model...');
    [BH_model,mean_dist_model] = sc_compute(M_k', zeros(1,nsamp), mean_dist_global, nbins_theta, ...
        nbins_r, r_inner, r_outer, out_vec_model);
    % debug
    debug_BH_model = BH_model;
    debug_mean_dist_model = mean_dist_model;
    
    disp('done.');
    
    disp('computing shape contexts for target...');
    [BH_target,mean_dist_target] = sc_compute(T', zeros(1,nsamp), mean_dist_global, nbins_theta, ...
        nbins_r,r_inner,r_outer,out_vec_target);
    % debug
    debug_BH_target = BH_target;
    debug_mean_dist_target = mean_dist_target;
    
    disp('done.');
    
    costmat_shape = hist_cost_2(BH_model, BH_target);
    theta_diff = repmat(tm_k, 1, nsamp) - repmat(tt', nsamp, 1);
    
    % debug
    debug_costmat_shape_d = costmat_shape;
    debug_theta_diff_d = theta_diff;
    
    % costmat_theta=abs(atan2(sin(theta_diff),cos(theta_diff)))/pi;
    
    if polarity_flag
        % use edge polarity
        costmat_theta = 0.5 * (1 - cos(theta_diff));
    else
        % ignore edge polarity
        costmat_theta = 0.5 * (1 - cos(2 * theta_diff));
    end
    costmat=(1 - ori_weight) * costmat_shape + ori_weight * costmat_theta;
    
    % pad the cost matrix with costs for dummies
    nptsd = nsamp + ndum;
    costmat2 = eps_dum * ones(nptsd, nptsd);
    costmat2(1:nsamp, 1:nsamp) = costmat;
    cvec = hungarian(costmat2);
    
    % debug
    debug_cvec_d = cvec;
    
    % update outlier indicator vectors
    [a, cvec2] = sort(cvec);
    out_vec_model = cvec2(1 : nsamp) > nsamp;
    out_vec_target = cvec(1 : nsamp) > nsamp;
    
    %M_nan = NaN * ones(nptsd, 2);
    %M_nan(1 : nsamp, :) = M_k;
    %M_nan = M_nan(cvec, :);
    M_nan = NaN * ones(nptsd, 2);
    M_nan(1 : nsamp, :) = M;
    M_nan = M_nan(cvec, :);
    T_nan = NaN * ones(nptsd, 2);
    T_nan(1 : nsamp, :) = T;
    
    % debug
    debug_cvec2_d = cvec2;
    debug_M_nan = M_nan;
    debug_T_nan = T_nan;
    
    % extract coordinates of non-dummy correspondences and use them
    % to estimate transformation
    index_good = find(~isnan(M_nan(1 : nsamp, 1)));
    n_good = length(index_good);
    M_non = M_nan(index_good, :);
    T_non = T_nan(index_good, :);
    
    % debug
    debug_index_good = index_good;
    debug_M_non = M_non;
    debug_T_non = T_non;
    
    % may be not required
    if affine_start_flag
        if k == 1
            % use huge regularization to get affine behavior
            lambda_o = 1000;
        else
            lambda_o = beta_init * r^(k-2);
        end
    else
        lambda_o = beta_init * r^(k-1);
    end
    beta_k = (mean_dist_target^2) * lambda_o;
    
    [cx,cy,E]=bookstein(M_non, T_non, beta_k);
    
    % debug
    debug_cx_d = cx;
    debug_cy_d = cy;
    
    % calculate affine cost
    A = [cx(n_good+2 : n_good+3, :) cy(n_good+2 : n_good+3, :)];
    s = svd(A);
    aff_cost = log(s(1)/s(2));
    
    % calculate shape context cost : doubtable!
    [a1, b1] = min(costmat, [], 1);
    [a2, b2] = min(costmat, [], 2);
    sc_cost = max(mean(a1), mean(a2));
    
    % warp each coordinate
    fx_aff = cx(n_good+1 : n_good+3)' * [ones(1, nsamp); M'];
    d2 = max(dist2(M_non, M), 0);
    U = d2.*log(d2 + eps);
    fx_wrp = cx(1 : n_good)' * U;
    fx = fx_aff + fx_wrp;
    fy_aff = cy(n_good+1 : n_good+3)' * [ones(1, nsamp); M'];
    fy_wrp = cy(1 : n_good)' * U;
    fy = fy_aff + fy_wrp;
    
    Z = [fx; fy]';
    
    % debug
    %debug_Z_d = Z;
    
    % apply the warp to the tangent vectors to get the new angles
    Xtan = M + tan_eps * [cos(tm) sin(tm)];
    fx_aff = cx(n_good+1 : n_good+3)' * [ones(1,nsamp); Xtan'];
    d2 = max(dist2(M_non, Xtan), 0);
    U = d2.*log(d2 + eps);
    fx_wrp = cx(1 : n_good)' * U;
    fx = fx_aff + fx_wrp;
    fy_aff = cy(n_good+1 : n_good+3)' * [ones(1,nsamp); Xtan'];
    fy_wrp = cy(1 : n_good)' * U;
    fy = fy_aff + fy_wrp;
    
    Ztan = [fx; fy]';
    tm_k = atan2(Ztan(:,2)-Z(:,2), Ztan(:,1)-Z(:,1));
    
    % debug
    %debug_tm_k_update = tm_k;
    
    % update M_k for the next iteration
    M_k=Z;
    
    if k==n_iter
        s=0;
    else
        k=k+1;
    end
    
end

% compute the score

% global ssd
[x, y] = meshgrid(1 : N2, 1 : N1);
x = x(:);
y = y(:);
col =length(x);
fx_aff = cx(n_good+1 : n_good+3)' * [ones(1, col); x'; y'];
d2 = dist2(M_non, [x y]);
fx_wrp = cx(1 : n_good)' * (d2.*log(d2 + eps));
fx = fx_aff + fx_wrp;
fy_aff = cy(n_good+1 : n_good+3)' * [ones(1, col); x'; y'];
fy_wrp = cy(1 : n_good)' * (d2.*log(d2 + eps));
fy = fy_aff + fy_wrp;
disp('computing warped image...')
V1w = griddata(reshape(fx, N1, N2), reshape(fy, N1, N2), model, reshape(x, N1, N2), reshape(y, N1, N2));
fz = find(isnan(V1w));
V1w(fz)=0;
ssd = (target - V1w).^2;
ssd_global = sum(ssd(:));

% local ssd
wd=2*w+1; % window size
win_fun=gaussker(wd); % gaussian weights

% extract sets of blocks around each coordinate
% first do 1st shape; need to use transformed coords.
win_list_1=zeros(nsamp,wd^2);
for qq=1:nsamp
    row_qq=round(M_k(qq,2));
    col_qq=round(M_k(qq,1));
    row_qq=max(w+1,min(N1-w,row_qq));
    col_qq=max(w+1,min(N2-w,col_qq));
    tmp=V1w(row_qq-w:row_qq+w,col_qq-w:col_qq+w);
    tmp=win_fun.*tmp;
    win_list_1(qq,:)=tmp(:)';
end
% now do 2nd shape
win_list_2=zeros(nsamp,wd^2);
for qq=1:nsamp
    row_qq=round(T(qq,2));
    col_qq=round(T(qq,1));
    row_qq=max(w+1,min(N1-w,row_qq));
    col_qq=max(w+1,min(N2-w,col_qq));
    tmp=target(row_qq-w:row_qq+w,col_qq-w:col_qq+w);
    tmp=win_fun.*tmp;
    win_list_2(qq,:)=tmp(:)';
end
ssd_all=sqrt(dist2(win_list_1,win_list_2));

cost_1=0;
cost_2=0;
for qq=1:nsamp
   cost_1=cost_1+ssd_all(qq,b2(qq));
   cost_2=cost_2+ssd_all(b1(qq),qq);
end
ssd_local=(1/nsamp)*max(mean(cost_1),mean(cost_2));
ssd_local_avg=(1/nsamp)*0.5*(mean(cost_1)+mean(cost_2));

% debug
debug_ssd_local_d =  ssd_local;
debug_ssd_local_avg_d = ssd_local_avg;


