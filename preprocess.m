imgs = dir('./data/archery/');
img_num = length(imgs);
data = zeros(11,4900);

for i = 4:img_num;
    img = imread(strcat('./data/archery/', imgs(i).name));
    disp(['processing the number ',num2str(i-3),' pic: ', imgs(i).name]);
    img_gray = rgb2gray(img);
    img_gray = double(img_gray);
    img_n = imresize(img_gray, [70,70]);
    img_n = mat2gray(img_n,[0 1]);
    tmp = reshape(img_n,1,4900);
    data(i-3,:) = tmp;
end
disp('done!');
save('data', 'data');

