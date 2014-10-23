load digit_100_train_easy;
V_t=reshape(train_data(1,:),28,28)';
sf=2.5;
V_t=imresize(V_t,sf,'bil');

%c=contourc(V_t,[.5 .5]);
figure(1);
%subplot(2,2,1);
imshow(V_t);
%subplot(2,2,2);
V_t=flipud(V_t);
%b=contour(V_t,[0.5,0.5]);
%c=contourc(V_t,[0.5,0.5]);

% [x1,y1,t1]=bdry_extract_3(V_t);
% x=[x1 y1];
% figure(2);
% %subplot(2,2,3);
% plot(x(:,1),x(:,2),'ro');

[x2,y2,t2]=boundry_extract(V_t);
y=[x2 y2];
figure(3);
%subplot(2,2,2);
plot(y(:,1),y(:,2),'ro');