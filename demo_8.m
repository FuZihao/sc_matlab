clear;
load data;

%pic_num = size(data, 1);
pic_num = 3;

% image size
N1 = 100;
N2 = 100;

figure(1);
pic = reshape(data(2, :), N1, N2);
imshow(pic);

pic = flipud(pic);
[c, h] = contour(pic);
clabel(c, h);
colorbar;


% pic = flipud(pic);
% [x1, y1, t1] = bdry_extract_3(pic);
% x = [x1 y1];
% figure(2);
% %subplot(2,2,3);
% plot(x(:,1),x(:,2),'ro');