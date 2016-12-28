%% Daniel Revie
% Lab 4
% 11.19.2016
clear;
close all;

%% (a)

% declare parameter values
a = 2; kp = 10; ki = 5; kd = 3;

% Construct transfer function objects for the following:
%   1) System G(S)
%   2) Controller G_c(s)
%   3) Closed-loop feedback system H(s)
%   4) Tracking error H_err(s)
s = tf('s');
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s;
H = minreal(Gc*G/(1+Gc*G));
Herr = minreal(1/(1+Gc*G));
Hdist = minreal(G/(1+Gc*G));

% Calculate Poles
p = roots(H.den{1});    % H.dem{1} is the vector of the denominator of coefficients of H(s)
p = pzmap(H);

% Calculate the 40-db time constant from the pole lying closest to the
% imaginary axis on the s-plane.
tau = -log10(100)/abs(real(p(3)))
% Define t as a vector of 1001 equally spaced samples spanning the
% intervale 0<=t<=20
t = linspace(0,20,1001);

% Define the unit step
u = @(t) double(t>=0);

% define lsim
y = lsim(H, u(t), t);

% Plot
figure
plot(t, y, 'b-');
title('step response, kp = 10, ki = 5, kd = 3');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 1.4])
grid on;

% Repeat by doubling PID paramters one at a time


% Double p
a = 2; ki = 5; kd = 3;
kp = 20;
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s;
H = minreal(Gc*G/(1+Gc*G));
% define lsim
y = lsim(H, u(t), t);
% Plot
figure
plot(t, y, 'b-');
title('step response, a = 2, kp = 20, ki = 5, kd = 3');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 1.4])
grid on;
% Double i
a = 2; kp = 10; kd = 3;
ki = 10;
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s;
H = minreal(Gc*G/(1+Gc*G));
% define lsim
y = lsim(H, u(t), t);
% Plot
figure
plot(t, y, 'b-');
title('step response, a = 2, kp = 10, ki = 10, kd = 3');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 1.4])
grid on;
% Double d
a = 2; kp = 10; ki = 5;
kd = 6;
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s;
H = minreal(Gc*G/(1+Gc*G));
% define lsim
y = lsim(H, u(t), t);
% Plot
figure
plot(t, y, 'b-');
title('step response, a = 2 kp = 10, ki = 5, kd = 6');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 1.4])
grid on;

% Doubling dp increases the maximum peak of the output. 
% Doubling ki extends the peak value of the output.
% Doubling kd reduces the peak value of the output.


% Double p and i 
%a = 2; kd = 3;
%kp = 20; ki = 10; 
%G = 1/(s*(s+a));
%Gc = kp + ki/s + kd*s;
%H = minreal(Gc*G/(1+Gc*G));

% Double p and d 
%a = 2; ki = 5;
%kp = 20; kd = 6;
%G = 1/(s*(s+a));
%Gc = kp + ki/s + kd*s;
%H = minreal(Gc*G/(1+Gc*G));

%% (b)
% declare parameter values
a = 2; kp = 10; ki = 5; kd = 3;
s = tf('s');
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s;
H = minreal(Gc*G/(1+Gc*G));
Herr = minreal(1/(1+Gc*G));
Hdist = minreal(G/(1+Gc*G));
t = linspace(0,20,1001);
u = @(t) double(t>=0);


r1 = u(t) + u(t-10);
r2 = 0.1.*t.*u(t);
r3 = atan(0.1*t).*u(t);

r4_1 = .04.*t.*u(t) - .04.*t.*u(t-10);
r4_2 = (-2 + 0.69.*t - 0.07.*t.^2 + 0.0025.*t.^3).*u(t-10) - (-2 + 0.69.*t - 0.07.*t.^2 + 0.0025.*t.^3).*u(t-14);
r4_3 = (0.8 + 0.2.*(t - 14)).*u(t-14) - (0.8 + 0.2.*(t - 14)).*u(t-20);
r4 = r4_1 + r4_2 + r4_3;

y1 = lsim(H,r1,t);
y2 = lsim(H,r2,t);
y3 = lsim(H,r3,t);
y4 = lsim(H,r4,t);

ye1 = lsim(Herr, r1, t);
ye2 = lsim(Herr, r2, t);
ye3 = lsim(Herr, r3, t);
ye4 = lsim(Herr, r4, t);


figure;
plot(t, y1, 'b-', t, r1, 'r--');
title('tracking step changes');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');

figure;
plot(t, ye1, 'b-');
title('tracking error');
xlabel('t');
ylabel('e(t)');
axis([0 20 -.2 1.4])
grid on;

figure;
plot(t, y2, 'b-', t, r2, 'r--');
title('ramp tracking');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');

figure;
plot(t, ye2, 'b-');
title('tracking error');
xlabel('t');
ylabel('e(t)');
axis([0 20 -.1 .1])
grid on;

figure;
plot(t, y3, 'b-', t, r3, 'r--');
title('ramp tracking with correct angle');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');

figure;
plot(t, ye3, 'b-');
title('tracking error');
xlabel('t');
ylabel('e(t)');
axis([0 20 -.1 .1])
grid on;

figure;
plot(t, y4, 'b-', t, r4, 'r--');
title('accelerating case');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');

figure;
plot(t, ye4, 'b-');
title('tracking error');
xlabel('t');
ylabel('e(t)');
axis([0 20 -.1 .1])
grid on;

% Particular case
ki = 0;
s = tf('s');
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s;
H = minreal(Gc*G/(1+Gc*G));
Herr = minreal(1/(1+Gc*G));
Hdist = minreal(G/(1+Gc*G));
t = linspace(0,20,1001);
u = @(t) double(t>=0);

r2 = 0.1.*t.*u(t);
y2 = lsim(H,r2,t);
ye2 = lsim(Herr, r2, t);

figure;
plot(t, y2, 'b-', t, r2, 'r--');
title('ramp tracking, ki = 0');
xlabel('t');
ylabel('y(t)');
axis([0 20 0 2.5])
legend('y(t)', 'r(t)');

figure;
plot(t, ye2, 'b-');
title('tracking errorn ki = 0');
xlabel('t');
ylabel('e(t)');
axis([0 20 -.1 .1])
grid on;

%% (c)
% declare parameter values
a = 2; kp = 10; ki = 5; kd = 3; tau = 0.05;
s = tf('s');
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s/(tau*s + 1);
H = minreal(Gc*G/(1+Gc*G));
Herr = minreal(1/(1+Gc*G));
Hdist = minreal(G/(1+Gc*G));
t = linspace(0,20,1001);
u = @(t) double(t>=0);


r1 = u(t) + u(t-10);
r2 = 0.1.*t.*u(t);
r3 = atan(0.1*t).*u(t);

r4_1 = .04.*t.*u(t) - .04.*t.*u(t-10);
r4_2 = (-2 + 0.69.*t - 0.07.*t.^2 + 0.0025.*t.^3).*u(t-10) - (-2 + 0.69.*t - 0.07.*t.^2 + 0.0025.*t.^3).*u(t-14);
r4_3 = (0.8 + 0.2.*(t - 14)).*u(t-14) - (0.8 + 0.2.*(t - 14)).*u(t-20);
r4 = r4_1 + r4_2 + r4_3;


H = Herr*Gc;
f1 = lsim(H, r1, t);
f2 = lsim(H, r2, t);
f3 = lsim(H, r3, t);
f4 = lsim(H, r4, t);


figure;
plot(t, f1);
title('torque f(t) -- step changes');
xlabel('t');
ylabel('f(t)');
axis([0 20 -20 80])
grid on;

figure;
plot(t, f2);
title('torque f(t) -- ramp tracking');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, .5])
grid on;

figure;
plot(t, f3);
title('torque f(t) -- ramp with correct angle');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, .5])
grid on;

figure;
plot(t, f4);
title('torque f(t) -- accelerating case');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, .7])
grid on;

%% (d)
a = 2; kp = 10; ki = 5; kd = 3; tau = 0.05;
s = tf('s');
G = 1/(s*(s+a));
Gc = kp + ki/s + kd*s/(tau*s + 1);
H = minreal(Gc*G/(1+Gc*G));
Herr = minreal(1/(1+Gc*G));
Hdist = minreal(G/(1+Gc*G));
t = linspace(0,20,1001);
u = @(t) double(t>=0);


r1 = u(t) + u(t-10);
r2 = 0.1.*t.*u(t);
r3 = atan(0.1*t).*u(t);

r4_1 = .04.*t.*u(t) - .04.*t.*u(t-10);
r4_2 = (-2 + 0.69.*t - 0.07.*t.^2 + 0.0025.*t.^3).*u(t-10) - (-2 + 0.69.*t - 0.07.*t.^2 + 0.0025.*t.^3).*u(t-14);
r4_3 = (0.8 + 0.2.*(t - 14)).*u(t-14) - (0.8 + 0.2.*(t - 14)).*u(t-20);
r4 = r4_1 + r4_2 + r4_3;

fdist = 2*(u(t-4)-u(t-6)); % wind gust

ydist = lsim(Hdist,fdist,t);

y1 = lsim(H,r1,t);
ytot1 = y1 + ydist;

y2 = lsim(H,r2,t);
ytot2 = y2 + ydist;

y3 = lsim(H,r3,t);
ytot3 = y3 + ydist;

y4 = lsim(H,r4,t);
ytot4 = y4 + ydist;

figure;
plot(t, ytot1, 'b', t, r1, 'r--', t, fdist, 'k:');
title('wind gust -- step changes');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'fdist', 'Location', 'SouthEast');

figure;
plot(t, ytot2, 'b', t, r2, 'r--', t, fdist, 'k:');
title('wind gust -- ramp');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'fdist', 'Location', 'SouthEast');

figure;
plot(t, ytot3, 'b', t, r3, 'r--', t, fdist, 'k:');
title('wind gust -- ramp with correct angle');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'fdist', 'Location', 'NorthEast');


figure;
plot(t, ytot4, 'b', t, r4, 'r--', t, fdist, 'k:');
title('wind gust -- accelerating');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'fdist', 'Location', 'SouthEast');

seed=2016; rng(seed);              % initialize random number generator
fdist = randn(size(t));            % zero-mean, unit-variance noise
ydist = lsim(Hdist,fdist,t);

y1 = lsim(H,r1,t);
ytot1 = y1 + ydist;

y2 = lsim(H,r2,t);
ytot2 = y2 + ydist;

y3 = lsim(H,r3,t);
ytot3 = y3 + ydist;

y4 = lsim(H,r4,t);
ytot4 = y4 + ydist;

figure;
plot(t, ytot1, 'b', t, r1, 'r--');
title('wind noise -- step changes');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');


figure;
plot(t, ytot2, 'b', t, r2, 'r--');
title('wind noise -- ramp');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');


figure;
plot(t, ytot3, 'b', t, r3, 'r--');
title('wind noise -- ramp with correct angle');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');


figure;
plot(t, ytot4, 'b', t, r4, 'r--');
title('wind noise -- accelerating');
xlabel('t');
ylabel('f(t)');
axis([0 20 0, 2.5])
legend('y(t)', 'r(t)', 'Location', 'NorthWest');