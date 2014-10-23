imgs = dir('./data/boxing/');
img_num = length(imgs);

N1 = 100;
N2 = 100;
data = zeros(img_num-3, N1*N2);
test_names = cell(img_num-3, 1);

for i = 4:img_num;
    img = imread(strcat('./data/boxing/', imgs(i).name));
    disp(['processing the number ',num2str(i-3),' pic: ', imgs(i).name]);
    img_gray = rgb2gray(img);

    img_n = imresize(img_gray, [N1,N2], 'bilinear');
    img_gray = double(img_gray);
    img_n = mat2gray(img_n,[0 1]);
    tmp = reshape(img_n,1,N1*N2);
    data(i-3,:) = tmp;
    test_names{i-3, 1} = imgs(i).name;
end
disp('done!');
save('test', 'data');
save('test_names','test_names');