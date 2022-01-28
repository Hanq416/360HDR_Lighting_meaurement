%data analysis function
function staReport(imap,cv_flg,cr_flg)
%%
th = -30; % statisic starting angle range [-90 to 90], 0: defualt
%lux results
lux_max = max(imap(:,3)); %max lux number
%average lux of direction with positive tilting angle
row_pos = (-1.*imap(:,1)>= th); 
lux_avg = mean(nonzeros(imap(:,3).*row_pos));
lux_std = std(nonzeros(imap(:,3).*row_pos));
%CV results
if cv_flg
    cv_max = max(imap(:,4)); %most unnuniform
    ycv = find(imap(:,4) == cv_max);
end
%CR results
if cr_flg
    crln_max = max(imap(:,5)); %highest near background contrast point
    crlf_max = max(imap(:,6)); %highest far background contrast point
    crrn_max = max(imap(:,7)); %highest near background contrast point
    crrf_max = max(imap(:,8)); %highest far background contrast point
    ycrln = find(imap(:,5) == crln_max);
    ycrlf = find(imap(:,6) == crlf_max);
    ycrrn = find(imap(:,7) == crrn_max);
    ycrrf = find(imap(:,8) == crrf_max);
end

%find aiming direction of max lux
ylux = find(imap(:,3) == lux_max);
%calculating EV and Eh
yev = find(imap(:,1) == 0);
yev(:,2) = imap(yev(:,1),3);
yev(:,3) = find(imap(:,1) == -90);
yev(:,4) = imap(yev(:,3),3);
%generate a brief report:
fprintf('Brief Uniformity Data Evaluation: \n');
fprintf('\n[1] Aiming direction of max illuminance\n');
fprintf('Horizontal : %d(deg); Vertical : %d(deg); Max illuminance: %.2f(lux)\n',...
    imap(ylux(1),2)-180,-1.*imap(ylux(1),1),lux_max);
fprintf('\n[2] Average illuminance of veritcal aiming direction >= %d (deg)\n',th);
fprintf('Average illuminance: %.2f(lux); Standard Deviation: %.2f(lux)\n', lux_avg, lux_std);
if cv_flg
    fprintf('\n[3] Aiming Direction with <MOST Ununiform> lighting condition\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); Coefficient of Variance: %.3f\n',...
        imap(ycv,2)-180,-1.*imap(ycv,1),cv_max);
end
if cr_flg
    fprintf('\n[4-1] Highest near background luminance contrast point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (NEAR)contrast-ratio: %.3f\n',...
        imap(ycrln(1),2)-180,-1.*imap(ycrln(1),1),crln_max);
    fprintf('\n[4-2] Highest far background luminance contrast point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (FAR)contrast-ratio: %.3f\n',...
        imap(ycrlf(1),2)-180,-1.*imap(ycrlf(1),1),crlf_max);
    fprintf('\n[4-3] Highest near background luminance ratio point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (NEAR)contrast-ratio: %.3f\n',...
        imap(ycrrn(1),2)-180,-1.*imap(ycrrn(1),1),crrn_max);
    fprintf('\n[4-4] Highest far background luminance ratio point\n');
    fprintf('Horizontal : %d(deg); Vertical : %d(deg); (FAR)contrast-ratio: %.3f\n',...
        imap(ycrrf(1),2)-180,-1.*imap(ycrrf(1),1),crrf_max);
end
fprintf('\n[5] Veritcal illuminance (Ev) = %.2f lux \n',mean(yev(:,2)));
fprintf('    Horizontal illuminance (Eh) = %.2f lux \n',mean(yev(:,4)));
end