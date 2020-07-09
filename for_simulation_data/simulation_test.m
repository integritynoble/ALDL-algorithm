clc;clear;close all
address='.\boatman_image\';
start_num=860;
end_num=875;
pic_video=8;
ver_num=256;
hor_num=256;
picture_num=end_num-start_num+1;
video_num=double(picture_num/pic_video);
orig1=zeros(ver_num,hor_num,picture_num);
%orig=orig1;
orig=zeros(ver_num,hor_num,video_num*8);
for i=1:picture_num
    add2=strcat(address,int2str(start_num+i-1));
    add3=strcat(add2,'.bmp');
    A1=norm1(double(imread(add3)));
    A=imresize(A1,[ver_num hor_num]);
    orig1(:,:,i)=norm1(A);
end

load('.\train\boatman\Code.mat')
mask=code;
for i=1:video_num
    for j=1:pic_video
        orig(:,:,(i-1)*pic_video+j)=circshift(orig1(:,:,(i-1)*pic_video+j),[j 0]);
    end
   
    mask1=double(mask);
    B=orig(:,:,(i-1)*pic_video+1:i*pic_video).*mask1;
    meas(:,:,i)=sum(B,3);    
end
%code=uint8(mask);
E=meas;
truth=orig;
stepsize=1;
save('.\test\boatman\Data.mat','E','truth','stepsize')
save('.\test\boatman\Code.mat','code')




