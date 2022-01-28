%illuminance calculation
function [lx] = FASTequisolid(hdr,luxmask,CF, perception)
if perception == "eml"
    lmap = (hdr(:,:,1).*0.0013 + hdr(:,:,2).*0.3812 + hdr(:,:,3).*0.6475).*179.*CF;
else
    lmap = (hdr(:,:,1).*0.265 + hdr(:,:,2).*0.670 + hdr(:,:,3).*0.065).*179.*CF;
end
lx = sum(sum(lmap.*luxmask));
end