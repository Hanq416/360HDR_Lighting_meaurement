function plot3d_CV(imap,ts,as,ta,tb,aa,ab,figName,fcode) %#ok<*INUSL>
%%
imap(:,1) = (imap(:,1) + max(imap(:,1)))./ts;
imap(:,2) = imap(:,2)./as;
imap(:,1) = imap(:,1)+1; imap(:,2) = imap(:,2)+1;
map = zeros(max(imap(:,1)),max(imap(:,2)));
for w = 1: size(imap,1)
    map(imap(w,1),imap(w,2)) = imap(w,4);
end
map(1,:) = mean(map(1,:));
map(size(map,1),:) = mean(map(size(map,1),:));
coordinate = []; color_ref = []; ct = 1;
for h = 1 : size(map,2)
    for v = 1 : size(map,1)
        hd = (h-1)*as; vd = 90 - (v-1)*ts;
        coordinate(ct,1) = map(v,h)*cosd(vd)*sind(hd); % x
        coordinate(ct,2) = map(v,h)*cosd(vd)*cosd(hd); % y
        coordinate(ct,3) = map(v,h)*sind(vd); % z
        color_ref(ct,1) = map(v,h);
        ct = ct + 1;
    end
end
%% plot
figure(fcode); title(['\fontsize{16}\color[rgb]{0 .5 .5}',figName]);
scatter3(coordinate(:,1),coordinate(:,2),coordinate(:,3), 40, color_ref, '*');
hold on; scatter3(0,0,0,100,'filled'); hold off;
colormap(jet); hcb = colorbar; title(hcb,'CV');
xlabel('X');
ylabel('Y');
zlabel('Z');
end