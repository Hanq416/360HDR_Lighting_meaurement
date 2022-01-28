% show HDR image
function hdrGammaShow(imHDR,gamma)
fr = imHDR(:,:,1);fg = imHDR(:,:,2);fb = imHDR(:,:,3);
fr = single(fr).^gamma;fg = single(fg).^gamma;fb = single(fb).^gamma;
fig1 = cat(3,fr,fg,fb);imshow(fig1);
end