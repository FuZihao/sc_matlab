clear;
load test;
load test_names;
load x;

%Define flags and parameters:
display_flag = 1;
affine_start_flag = 1;
polarity_flag = 1;
nsamp = 150; % sample number
eps_dum = 0.25; % outlier threhold
ndum_frac = 0.25;        
mean_dist_global = [];
ori_weight = 0.1;
nbins_theta = 12;
nbins_r = 5;
r_inner = 1/8;
r_outer = 2;
tan_eps = 1.0;
n_iter = 4;
beta_init = 1;
r = 1; % annealing rate
w = 4;
sf = 2.5;

% score weight
W_ac = x(1);
W_sc = x(2);
W_be = x(3);
bias = x(4);

pic_num = size(data, 1);
pic_num = pic_num - 1; % substract the model picture

scores = zeros(pic_num, 1); % score of each picture
scores_component = zeros(pic_num, 3);

% image size
N1 = 100;
N2 = 100;

model_index = 45; % the last picture is transformed model
model = reshape(data(model_index, :), N1, N2); 
[mx, my, tm] = bdry_extract_3(model);
nsamp_model = length(mx);
if nsamp_model >= nsamp
    [mx,my,tm] = get_samples_1(mx, my, tm, nsamp);
else
    error('transformed model doesn''t have enough samples')
end
M = [mx ,my];

% debug
 ii = 1;
 %pic_num = 15;

for i = ii : pic_num
    disp(['precess data number ',num2str(i)]);
    target = reshape(data(i, :), N1, N2);
    % imshow(target);
    [tx, ty, tt] = bdry_extract_3(target);
    nsamp_target = length(tx);
    if nsamp_target >= nsamp
        [tx, ty, tt] = get_samples_1(tx, ty, tt, nsamp);
    else 
        error('target model doesn''t have enough samples');
    end
    T = [tx, ty];
    
    % compute correspondences
    M_k = M;
    tm_k = tm;
    k = 1;
    s = 1;
    ndum = round(ndum_frac * nsamp);
    out_vec_model = zeros(1, nsamp);
    out_vec_target = zeros(1, nsamp);
    
    while s
        disp(['iter=' int2str(k)]);
        %disp('computing shape contexts for transformed model...');
        [BH_model,mean_dist_model] = sc_compute(M_k', zeros(1,nsamp), mean_dist_global, nbins_theta, ...
                                                                            nbins_r, r_inner, r_outer, out_vec_model);
       % disp('done.');
   
        %disp('computing shape contexts for target...');
        [BH_target,mean_dist_target] = sc_compute(T', zeros(1,nsamp), mean_dist_global, nbins_theta, ...
                                                                            nbins_r,r_inner,r_outer,out_vec_target);
      %  disp('done.');
              
        costmat_shape = hist_cost_2(BH_model, BH_target);
        theta_diff = repmat(tm_k, 1, nsamp) - repmat(tt', nsamp, 1);
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
        
        % update outlier indicator vectors
        [a, cvec2] = sort(cvec);
        out_vec_model = cvec2(1 : nsamp) > nsamp;
        out_vec_target = cvec(1 : nsamp) > nsamp;

%         M_nan = NaN * ones(nptsd, 2);
%         M_nan(1 : nsamp, :) = M_k;
%         M_nan = M_nan(cvec, :);
        M_nan = NaN * ones(nptsd, 2);
        M_nan(1 : nsamp, :) = M;
        M_nan = M_nan(cvec, :);
        T_nan = NaN * ones(nptsd, 2);
        T_nan(1 : nsamp, :) = T;
   
        % extract coordinates of non-dummy correspondences and use them
        % to estimate transformation
        index_good = find(~isnan(M_nan(1 : nsamp, 1)));
        n_good = length(index_good);
        M_non = M_nan(index_good, :);
        T_non = T_nan(index_good, :);
                
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
        
        % calculate affine cost
        %A = [cx(n_good+2 : n_good+3, :) cy(n_good+2 : n_good+3, :)];
        %s = svd(A);
        %aff_cost = log(s(1)/s(2));
   
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
        
           % update Xk for the next iteration
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
  %  disp('computing warped image...')
    V1w = griddata(reshape(fx, N1, N2), reshape(fy, N1, N2), model, reshape(x, N1, N2), reshape(y, N1, N2));
    fz = find(isnan(V1w)); 
    V1w(fz)=0;
    ssd = (target - V1w).^2;
    ssd_global = sum(ssd(:));
    
    % local ssd
    wd = 2 * w + 1; % window size
    win_fun = gaussker(wd); % gaussian weights
    
    % extract sets of blocks around each coordinate
    % first do 1st shape; need to use transformed coords.
    win_list_1 = zeros(nsamp,wd^2);
    for qq = 1:nsamp
        row_qq = round(M_k(qq,2));
        col_qq = round(M_k(qq,1));
        row_qq = max(w+1,min(N1-w,row_qq));
        col_qq = max(w+1,min(N2-w,col_qq));
        tmp = V1w(row_qq-w : row_qq+w, col_qq-w : col_qq + w);
        tmp = win_fun.*tmp;
        win_list_1(qq, :) = tmp(:)';
    end
    % now do 2nd shape
    win_list_2 = zeros(nsamp, wd^2);
    for qq = 1:nsamp
        row_qq = round(T(qq,2));
        col_qq = round(T(qq,1));
        row_qq = max(w+1, min(N1-w,row_qq));
        col_qq = max(w+1, min(N2-w,col_qq));
        tmp = target(row_qq-w : row_qq+w, col_qq-w : col_qq+w);
        tmp = win_fun.*tmp;
        win_list_2(qq, :) = tmp(:)';
    end
    ssd_all = sqrt(dist2(win_list_1, win_list_2));
    
    cost_1 = 0;
    cost_2 = 0;
    for qq = 1:nsamp
        cost_1 = cost_1+ssd_all(qq,b2(qq));
        cost_2 = cost_2+ssd_all(b1(qq),qq);
    end
    ssd_local = (1/nsamp)*max(mean(cost_1),mean(cost_2));
    ssd_local_avg = (1/nsamp)*0.5*(mean(cost_1)+mean(cost_2));
    
    scores_component(i, 1) = ssd_local_avg;
    scores_component(i, 2) = sc_cost;
    scores_component(i, 3) = E;
    score = W_ac*ssd_local_avg + W_sc*sc_cost + W_be*E + bias;
    scores(i) = score;
end
    [rank, index] = sort(scores);
%     save('feature_point_1','scores_component')
    
    % display template image
%     if display_flag
%         figure(1);
%         imshow(model);
%     end
%     
%     % display ranked images
% 
%     if display_flag
%        figure(2)
%        for i = 1:pic_num
%           subplot(2, 5, i);
%           %figure(i+1);
%           target = reshape(data(index(i), :), N1, N2);
%           imshow(target);
%        end
% %        target = reshape(data(index(1), :), N1, N2);
% %        imshow(target);
%     end
%     
        
        
