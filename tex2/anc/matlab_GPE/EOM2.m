function dy = EOM2(t,y,~,H,H_ramp,U)

dy = -1i.*( (H + H_ramp(t))*y + U.*abs(y).^2.*y );