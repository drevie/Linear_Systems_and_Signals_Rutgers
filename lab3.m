% Daniel Revie 
% 10/26/2016
% Lab 3
close all;
clear all;

%% (a) 
syms s z;
w0 = 4; 

p = z^2 + z + 1.25;

% define constants 
H = (s + 3)/ (s^2 + s + 1.25); 
% Define  t
t = linspace(0, 10, 1001);

% define function
u = @(t) (t>=1);
x = sin(w0*t); 

% solve poles
real_pole = solve(p, z)

% Determine analytical expression for the impulse response h(t) 
H = partfrac(H); 
h = ilaplace(H)
hn = subs(h) ;

% solve for 40dB 
a = (reallog(100))
b = (abs(real(real_pole(1))))
a / b %% FOR SOME REASON IT COMES OUT AS A BIG NUMBER OVER A BIG NUMBER RATHER THAN A DECIMAL

%figure;
plot(t, hn, 'b-');
xlabel('t (sec)'); 
title('impluse response, h(t)');
axis([0 10 -2 2])
grid on;

%% (B)
% substitute for jw0
s = 1i*w0; 

% Define H(jw)
H_jw = (s + 3)/ (s^2 + s + 1.25);
% Define |H(jw)|^2
H_jw_2 = (H_jw)^2;
% Define phase
phase = angle(H_jw);

% Define y_st
y_st = abs(H_jw)*sin(w0*t + phase);


%% (C)
b = [1 3]; a = [1 (1-1i*w0) (1.25 - 1i*w0) -1.25*1i*w0];
[R, P] = residue(b, a);

y_c = (H_jw*exp(1i*w0*t) + R(2)*exp(P(2)*t) + R(3)*exp(P(3)*t));
y_imag  = imag(y_c);

q = x(8:10); 
q = q * 10;

%% (D) 

% Theoretical 
tph_theoretical =  -1*(phase / w0)

% Determine Actual Values
% Determine time interval 8-9 
t_ph = 8:0.01:9; 
% Redfine functions in new interval
xph = x(8:9);
y_c = (H_jw*exp(1i*w0*t) + R(2)*exp(P(2)*t) + R(3)*exp(P(3)*t));
yph = imag(y_c);
% Capture maximum values 
[t1_y, t1_x] = max(x(8:9))
[t2_y, t2_x] = max(yph)

tph_actual = t2_y - t1_y

figure;
plot(t, y_imag, 'blue', t, y_st, 'r--', t, x, 'black--', t1_x, t1_y, 'r.', t2_x, t2_y, 'r.')
xlabel('t (sec)'); 
title(' x(t), y(t), y_st(t)'); 
legend('exact', 'steady', 'input', 'max', 'Location', 'SouthWest');
axis([0 10 -1.3 1.3])
grid on;

%% (E)
% Define ytr 
ytr = y_imag - y_st; 

figure;
plot(t, ytr, 'b-');
xlabel('t (sec)'); 
title('transients, t40 = 9.21 sec'); 
grid on;

axis([0 10 -1.3 1.3])

disp('yes the result is consistent with the 40-db time constant that was calculated earlier in the code');

%% 2 
clear all;
syms s;
w0 = 5; a = .2; w = 0:0.01:10; 
H = (a*s)/(s^2 + a*s + w0^2); 


%% (A)

s = 1i*w; 
Hjw = (a*s)./(s.^2 + a*s + w0^2); 
Hjw2 = (a^2.*w.^2)./((w.^2 - w0^2).^2 + a^2.*w.^2);
phase_angle = angle(Hjw) / pi;


% Calculate the 3db frequencies 
w_p = sqrt(w0^2 + (a^2)/2 + a*sqrt(w0^2 + (a^2)/4))
w_n = sqrt(w0^2 + (a^2)/2 - a*sqrt(w0^2 + (a^2)/4))

line = w_n:0.05:w_p;
mag_half = 0.5;

figure;
plot(w, Hjw2, 'b-', line, ones(size(line))*mag_half, 'black');
xlabel('w'); 
title('|H(jw)|^2, w0 = 5, a = .2');
txt3 = '  delta(w) = a'; 
text(w_p, 1/2, txt3);
axis([0 10 0 1])

figure
plot(w, phase_angle, 'b-');
xlabel('w'); 
title('Phase Response, ArgH(jw)/ pi, a = 0.2');
axis([0 10 -1 1])
grid on;


%% (B)
Tmax = 40; T = Tmax/2000;
t = 0:T:Tmax; 
seed = 2016; rng(seed); 
v = randn(size(t));
x = sin(w0*t) + v; 

s = tf('s');
H = (a*s)/(s^2 + a*s + w0^2); 
%num = a; den = [1, a, w0^2];
% compute the filter output samples y(t) using the function lsim 
y = lsim(H, x, t, [0;0], 'zoh'); 

% Plot x(t) 
figure; 
plot(t, x, 'black')
xlabel('t');
title('noisy input sinusoid, x(t)'); 
axis([0 40 -4 4])
grid on;

% Plot y(t)
figure;
plot(t, y, 'b-');
xlabel('t');
title('filtered output, y(t), a = .2'); 
axis([0 40 -4 4])
grid on;

%% (C)
figure;
plot(t, v, 'black');
title('input noise, v(t)');
xlabel('t'); 
grid on;

y_v = lsim(H, v, t, [0;0], 'zoh'); 

figure
plot(t, y_v, 'b-');
title('filtered noise, yv(t), a = .2'); 
xlabel('t')
axis([0 40 -4 4]);
grid on; 

%% (D) 
% a = .5
disp('When chaging the values of alpha there are tradeoffs between noise reduction, speef of response, and quality of the resulting desired signal. The filtered noise graph is most erratic in the case of a = 1. The lower the value of a the higher the frequency of the response. The quality of the desired signal is best in the case of a smaller alpha. ')
clear all;
s = tf('s');
w0 = 5; a = .5; w = 0:0.01:10; 
H = (a*s)/(s^2 + a*s + w0^2); 

% (A) 

s = 1i*w; 
Hjw = (a*s)./(s.^2 + a*s + w0^2); 
Hjw2 = (a^2.*w.^2)./((w.^2 - w0^2).^2 + a^2.*w.^2);
phase_angle = angle(Hjw) / pi;

% Calculate the 3db frequencies 
w_p = sqrt(w0^2 + (a^2)/2 + a*sqrt(w0^2 + (a^2)/4))
w_n = sqrt(w0^2 + (a^2)/2 - a*sqrt(w0^2 + (a^2)/4))

line = w_n:0.05:w_p;
mag_half = 0.5;

figure;
plot(w, Hjw2, 'b-', line, ones(size(line))*mag_half, 'black');
xlabel('w'); 
title('|H(jw)|^2, w0 = 5, a = .5');
txt3 = '  delta(w) = a'; 
text(w_p, 1/2, txt3);
axis([0 10 0 1])

figure
plot(w, phase_angle, 'b-');
xlabel('w'); 
title('Phase Response, ArgH(jw)/ pi, a = 0.5');
axis([0 10 -1 1])
grid on;


% (B) 
Tmax = 40; T = Tmax/2000;
t = 0:T:Tmax; 
seed = 2016; rng(seed); 
v = randn(size(t));
x = sin(w0*t) + v; 

s = tf('s');
H = (a*s)/(s^2 + a*s + w0^2); 
%num = a; den = [1, a, w0^2];
% compute the filter output samples y(t) using the function lsim 
y = lsim(H, x, t, [0;0], 'zoh'); 

% Plot x(t) 
figure; 
plot(t, x, 'black')
xlabel('t');
title('noisy input sinusoid, x(t)'); 
axis([0 40 -4 4])
grid on;

% Plot y(t)
figure;
plot(t, y, 'b-');
xlabel('t');
title('filtered output, y(t), a = .5'); 
axis([0 40 -4 4])
grid on;

% (C) 

figure;
plot(t, v, 'black');
title('input noise, v(t)');
xlabel('t'); 
grid on;

y_v = lsim(H, v, t, [0;0], 'zoh'); 

figure
plot(t, y_v, 'b-');
title('filtered noise, yv(t), a = .5'); 
xlabel('t')
axis([0 40 -4 4]);
grid on; 

%---------------------------------------
% a = 1
% (A) 

clear all;
s = tf('s');
w0 = 5; a = 1; w = 0:0.01:10; 
H = (a*s)/(s^2 + a*s + w0^2); 


s = 1i*w; 
Hjw = (a*s)./(s.^2 + a*s + w0^2); 
Hjw2 = (a^2.*w.^2)./((w.^2 - w0^2).^2 + a^2.*w.^2);
phase_angle = angle(Hjw) / pi;

% Calculate the 3db frequencies 
w_p = sqrt(w0^2 + (a^2)/2 + a*sqrt(w0^2 + (a^2)/4))
w_n = sqrt(w0^2 + (a^2)/2 - a*sqrt(w0^2 + (a^2)/4))

line = w_n:0.05:w_p;
mag_half = 0.5;

figure;
plot(w, Hjw2, 'b-', line, ones(size(line))*mag_half, 'black');
xlabel('w'); 
title('|H(jw)|^2, w0 = 5, a = 1');
txt3 = '  delta(w) = a'; 
text(w_p, 1/2, txt3);
axis([0 10 0 1])

figure
plot(w, phase_angle, 'b-');
xlabel('w'); 
title('Phase Response, ArgH(jw)/ pi, a = 1');
axis([0 10 -1 1])
grid on;


% (B) 
Tmax = 40; T = Tmax/2000;
t = 0:T:Tmax; 
seed = 2016; rng(seed); 
v = randn(size(t));
x = sin(w0*t) + v; 

s = tf('s');
H = (a*s)/(s^2 + a*s + w0^2); 
%num = a; den = [1, a, w0^2];
% compute the filter output samples y(t) using the function lsim 
y = lsim(H, x, t, [0;0], 'zoh'); 

% Plot x(t) 
figure; 
plot(t, x, 'black')
xlabel('t');
title('noisy input sinusoid, x(t)'); 
axis([0 40 -4 4])
grid on;

% Plot y(t)
figure;
plot(t, y, 'b-');
xlabel('t');
title('filtered output, y(t), a = 1'); 
axis([0 40 -4 4])
grid on;

% (C) 

figure;
plot(t, v, 'black');
title('input noise, v(t)');
xlabel('t'); 
grid on;

y_v = lsim(H, v, t, [0;0], 'zoh'); 

figure
plot(t, y_v, 'b-');
title('filtered noise, yv(t), a = 1'); 
xlabel('t')
axis([0 40 -4 4]);
grid on; 

%% (E) 
clear all; 
w0 = 5; a = .2; Tmax = 40; T = Tmax/2000;
tn = 0:T:Tmax; 

% Define the input
seed = 2016; rng(seed); 
v = randn(size(tn));
x = sin(w0*tn) + v; 
xn = sin(w0*tn*T) + v;


% Define wr
wr = sqrt(w0^2 - (a^2)/4);

% Define functions
G = (a/wr)*exp((-a*T)/2)*sin(wr*T);
a1 = -2*exp((-a*T)/2)*cos(wr*T);
a2 = exp(-a*T);

% Compute using the iteration of Eq. (12) 
v1 = 0; v2 = 0; 
N = length(tn)
for n = 0:N-1
   y(n+1) = v1;
   v1 = v2 + G*x(n+1) - a1*y(n+1);
   v2 = -G*x(n+1) - a2*y(n+1);
end

% Compute using filter function
num = [0 G -G];
den = [1 a1 a2];
yfilter = filter(num, den, x);

% calculate lsim
s = tf('s');
H = (a*s)/(s^2 + a*s + w0^2); 

% compute the filter output samples y(t) using the function lsim 
y_lsim = lsim(H, x, tn, [0;0], 'zoh'); 

% Compute Euclidean norms
E_lsim = norm(y_lsim) - norm(y)
E_iter = norm(y) - norm(yfilter)

%% (F)
g = (a/wr).*exp((-a.*tn.*T)/2).*sin(wr.*tn.*T);
hd = g(n) - g(n-1);


