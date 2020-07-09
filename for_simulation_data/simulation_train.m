clc;clear;close all
address='.\boatman_image\';
start_num=345;
end_num=856;
pic_video=8;
ver_num=256;
hor_num=256;
picture_num=end_num-start_num+1;
video_num=picture_num-pic_video+1;
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
code1=rand(ver_num,hor_num)-0.5;
code1(code1<0)=0;
code1(code1>0)=1;
mask=zeros(ver_num,hor_num,pic_video);
C=code1;
for j=1:pic_video
    mask(:,:,j)=circshift(C,[j 0]);
end
for i=1:video_num
    for j=1:pic_video
        orig(:,:,(i-1)*pic_video+j)=circshift(orig1(:,:,i+j-1),[j 0]);
    end
   
    mask1=double(mask);
    B=orig(:,:,(i-1)*pic_video+1:i*pic_video).*mask1;
    meas(:,:,i)=sum(B,3);    
end
code=uint8(mask);
E=meas;
truth=orig;
stepsize=1;
save('.\train\boatman\Data.mat','E','truth','stepsize')
save('.\train\boatman\Code.mat','code')




