% figure(1);
% subplot(2,2,1);
% plot(rating(:,6), rating(:,10),'b.');
% 
% subplot(2,2,2);
% plot(rating(:,7), rating(:, 10),'r.');
% 
% subplot(2,2,3);
% plot(rating(:,8), rating(:, 10),'g.');

% [X,Y,Z] = sphere(16);
% x = [0.5*X(:); 0.75*X(:); X(:)];
% y = [0.5*Y(:); 0.75*Y(:); Y(:)];
% z = [0.5*Z(:); 0.75*Z(:); Z(:)];
% 
% S = repmat([20,20,20],numel(X),1);
% C = repmat([1,2,5],numel(X),1);
% s = S(:);
% c = C(:);
% 
% figure
% scatter3(x,y,z,s,c,'filled')
% view(40,35)

figure;

s = repmat(20, numel(rating(:,6)), 1);
c = rating(:, 5)';
scatter3(rating(:,6)', rating(:,7)', rating(:,8)', s, c, 'filled')
xlabel('x')
ylabel('y')
zlabel('z')
view(40,35)