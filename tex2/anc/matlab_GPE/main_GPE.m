close all;
clear;

L=300; %300; % system size
% define middle of the chain
i_CM = L/2+0.5; % centre of chain

J=1.0;% # hopping
U=1.0;% # Bose-Hubbard interaction strength

mu_i=0.02;% # initial chemical potential
mu_f=0.0002;% # final chemical potential
t_f=30.0/J;

t=linspace(0.0,t_f,21);

sites=1:1:L;

H_ramp = @(t) (mu_f - mu_i).*t/t_f*sparse( diag((sites-i_CM).^2 ) );

H=full(gallery('tridiag',L,-J,0,-J));
%H(1,end)=-J;
%H(end,1)=-J;
H = sparse( H + mu_i.*diag( (sites-i_CM).^2 ) );


[V,E]=eig(full(H));
y0=V(:,1)*sqrt(L);


absrel=1E-15;
abstol=1E-15;
y = EOM(t,y0,H,H_ramp,U,absrel,abstol);

%{
for j =1:length(t)
    plot(sites, abs(y(j,:)).^2,'.', 'markersize',20 );hold all
    plot(sites, abs(y(j,:)).^2,'.', 'markersize',20 );hold all
end
%}
plot(sites, abs(y(20,:)).^2,'.-', 'markersize',26 );hold all
