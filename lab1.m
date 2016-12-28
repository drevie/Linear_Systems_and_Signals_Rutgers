% Daniel Revie 
% 9/28/2016
% Lab 1 


% Question 1 
ep = .05;
ep2 = .1;
t = linspace(-1, 1, 1001);
u = @(t) (t>=0); 

P1 = (1/ep)*((u(t+(ep./2))-u(t-(ep./2))));
P2 = (1/ep2)*((u(t+(ep2./2))-u(t-(ep2./2))));
Q1 = (1/(sqrt(2*pi*ep)))*exp((-t.^2)/(2*ep));
Q2 = (1/(sqrt(2*pi*ep2)))*exp((-t.^2)/(2*ep2));
R1 = (1/pi)*(ep./((ep.^2) + t.^2));
R2 = (1/pi)*(ep2./((ep2.^2) + t.^2));
S1 = sinc(t/ep)./(pi*t);
S2 = sinc(t/ep2)./(pi*t);

plot(t, P1, 'color', 'b'); hold on;
plot(t, P2, 'color', 'r');
legend('e = .1', 'e = .01');
figure;

plot(t, Q1, 'color', 'b'); hold on;
plot(t, Q2, 'color', 'r');
legend('e = .1', 'e = .01');
figure;

plot(t, R1, 'color', 'b'); hold on;
plot(t, R2, 'color', 'r');
legend('e = .1', 'e = .01');
figure;


plot(t, S1, 'color', 'b'); hold on;
plot(t, S2, 'color', 'r');
legend('e = .1', 'e = .01');


