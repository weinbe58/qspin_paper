function y = EOM(t,y0,H,H_ramp,U,absrel,abstol)

options = odeset('RelTol',absrel,'AbsTol',abstol);
%start solving the ODE%
[~,y] = ode45('EOM2',t,y0,options,H,H_ramp,U);