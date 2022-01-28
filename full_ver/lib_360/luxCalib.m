%illuminance calibration
%5 lux method
%with four reference lux value
function lux_cal = luxCalib(lux,aa,ta,cfh,cfv0,cfv90,cfv180,cfv270)
if cfv0 == 0
    cfv0 = (cfv90 + cfv270)/2;
end
if cfv180 == 0
    cfv180 = (cfv90 + cfv270)/2;
end
if aa < 0
    aa = 360 + aa;
end
if aa >=0 && aa < 90
    cfv = (90-aa)/90*cfv0 + (aa)/90*cfv90;
elseif aa>=90 && aa <180
    cfv = (180-aa)/90*cfv90 + (aa-90)/90*cfv180;
elseif aa>=180 && aa<270
    cfv = (270-aa)/90*cfv180 + (aa-180)/90*cfv270;
elseif aa>=270 && aa<=360
    cfv = (360-aa)/90*cfv270 + (aa-270)/90*cfv0;
end
if ta <= 0
    lux_cal = lux*cfv;
else
    lux_cal = lux*((90-ta)/90.*cfv + (ta)/90*cfh);
end
end