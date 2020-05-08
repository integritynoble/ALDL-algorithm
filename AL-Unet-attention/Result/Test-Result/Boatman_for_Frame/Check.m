clear;%clc;
%close all
load('Data_Visualization_0.mat')
[batch,height,width,frame] = size(truth);
psnr_frame = zeros([1 batch*frame]);
ssim_frame = zeros([1 batch*frame]);
 
for b = 1:batch
   figure;
    for f = 1:frame
        bf=(b-1)*frame+f; pf=double(pred(b,:,:,f)); of=double(truth(b,:,:,f));
        psnr_frame(bf) = psnr(pf,of,max(of(:)));
        ssim_frame(bf) = ssim(pf,of);
        subplot(4,5,f)
        imagesc(squeeze(pf),[0 0.9]);colormap(hot)
    end
end
mean(psnr_frame) 
mean(ssim_frame)
%figure;imshow(squeeze(pf))
%figure;imshow(squeeze(of))