% Daniel Revie
% 10/18/2016 
% Lab 2 

%% 1 
close all
close all hidden

%% (A)
% Define constants
a = 0.9;
T = 0.05; td = 1; Tmax = 25; 
c1 = 1; c2 = 2; c3 = 1.5; 
t1 = 0; t2 = 10; t3 = 15; 

% Range of t 
t = 0:T:Tmax;

% Function Declarations
u = @(t) (t>=1);
F = @(t,td) (t>=0) - (t>=td); 
G = @(t,td) exp(-a*t).*(exp(a*min(t,td))-1) .* (t>=0); 

% Define x(t) and y(t) exact outputs
x = c1*F(t-t1, td) + c2*F(t-t2, td) + c3*F(t-t3, td);
y_exact = c1*G(t-t1, td) + c2*G(t-t2, td) + c3*G(t-t3, td);

% Define h(t) 
h = a*exp(-a*t);

% Define the convolution
y = T*conv(h,x);

% trim y to length of t to fit on plot against t
y = y(1:length(t)); 

% Plot
figure(1)
plot(t, x, ':', t, y, 'b', t, y_exact, 'r--'); 
title('td = 1, T = 0.05');
xlabel('t');
legend('input', 'conv', 'exact');
axis([0 25 0 3])

%% (B)
% Update td
td = 3; 


% Define x(t) and y(t) exact outputs
x = c1*F(t-t1, td) + c2*F(t-t2, td) + c3*F(t-t3, td);
y_exact = c1*G(t-t1, td) + c2*G(t-t2, td) + c3*G(t-t3, td);

% Define h(t) 
h = a*exp(-a*t);

% Define the convolution
y = T*conv(h,x);

% trim y to length of t to fit on plot against t
y = y(1:length(t)); 

figure(2)
plot(t, x, ':', t, y, 'b', t, y_exact, 'r--'); 
title('td = 3, T = 0.05');
xlabel('t');
legend('input', 'conv', 'exact');
axis([0 25 0 3])

%% (C)
% Update td
td = 5; 

% Define x(t) and y(t) exact outputs
x = c1*F(t-t1, td) + c2*F(t-t2, td) + c3*F(t-t3, td);
y_exact = c1*G(t-t1, td) + c2*G(t-t2, td) + c3*G(t-t3, td);

% Define h(t) 
h = a*exp(-a*t);

% Define the convolution
y = T*conv(h,x);

% trim y to length of t to fit on plot against t
y = y(1:length(t)); 

figure(3)
plot(t, x, ':', t, y, 'b', t, y_exact, 'r--'); 
title('td = 3, T = 0.05');
xlabel('t');
legend('input', 'conv', 'exact');
axis([0 25 0 3])

%% 2

%% (A)

% Define Constants 
w0 = 2; w1 = 3; a = 0.3; 

% Set time range 
Tmax = 100; T = Tmax/2000; t = 0:T:Tmax;

% Define function x 
x = sin(w1*t) .* F(t,30) + ...
    sin(w0*t) .* F(t-30, 40) + ... 
    sin(w1*t) .* F(t-70, 30); 

% Define constant wr 
wr = sqrt(w0^2 - (a^2)/4);

% Define g(t) 
g = a*exp(-a*t/2).*(cos(wr*t) - (a/(2*wr))*sin(wr*t));

% y temp 
y_temp = conv(g,x); 

% trim the convolution
y_temp = y_temp(1:length(t)); 
y_conv = x - T*y_temp;

% Plot the input signal
figure(4) 
plot(t, x, 'b'); 
title('input signal, x(t)');
xlabel('t'); 
axis([0 100 -2 2])

% Construct Transfer Function 
syms s; 
j = sqrt(-1);
H_ = abs((w1^2 - w0^2)/(w0^2 - w1^2 + a*j*w1))


% Allocate Vectors to represent transfer function
A = 0:T:30; 
A = zeros(size(A));
A(A == 0) = H_;

B = 0:T:40; 
B = zeros(size(B));
B(B == 0) = NaN;

C = [A B A];
C = C(1:length(t));

A = 0:T:30; 
A = zeros(size(A));
A(A == 0) = -H_;

D = [A B A];
D = D(1:length(t));

% Plot output signal via the convolutional method and the magnitudes
figure(5)
plot(t, y_conv, 'r', t, C, 'b', t, D, 'b');
title('output signal, y(t), conv method, a = 0.3');
legend('y(t)', '|H(jw1)|');
xlabel('t'); 
axis([0 100 -2 2])

%% (B) 
% Define Stable Pole
p = -a/(2+j*wr); 
tau = log(100)/abs((real(p)))

disp('The time constant tau at 40-dB is consistent with the transients observed in the plot of y(t). The y(t) levels off to zero at that value.');
% talk about if the value of t is consistent with the transients of plot of
% y(t)

%% (C)

% Set up tf for lsim
s = tf('s'); 
H = (s^2 + w0^2)/(s^2 + a*s + w0^2);

% Perform lsim
y_lsim = lsim(H, x, t);

% Plot lsim
figure(6);
plot(t, y_lsim, 'r', t, C, 'b', t, D, 'b');
title('output signal, y(t), lsim method, a = 0.3');
legend('y(t)', '|H(jw1)|');
xlabel('t'); 
axis([0 100 -2 2])

% Calculate Error
lnorm = norm(y_lsim);
convnorm = norm (y_conv);

error = abs(100*(convnorm - lnorm)/convnorm)

%% (D) 

% range of w 
w = 0:.05:5;

% Create function
mag_squared = (((w.^2 - w0^2).^2)./((w.^2-w0^2).^2 + (a^2)*(w.^2)));



% Point values
mag_squared_w0 = (((w0.^2 - w0^2).^2)./((w0.^2-w0^2).^2 + (a^2)*(w0.^2)));
mag_squared_w1 = (((w1.^2 - w0^2).^2)./((w1.^2-w0^2).^2 + (a^2)*(w1.^2)));

% Calculate the 3db frequencies 
w_p = sqrt(w0^2 + (a^2)/2 + a*sqrt(w0^2 + (a^2)/4))
w_n = sqrt(w0^2 + (a^2)/2 - a*sqrt(w0^2 + (a^2)/4))

line = w_n:0.05:w_p; 
mag_half = 0.5;

% Plot
figure(7); 
plot(w, mag_squared, 'b', w0, mag_squared_w0, 'r.', ... 
     w1, mag_squared_w1, 'r.', line, ones(size(line))*mag_half, 'r');
txt2 = '  w1';
text(w1, mag_squared_w1, txt2);
txt3 = '  3-dB width'; 
text(w_p, 1/2, txt3);
title('notch filter, |H(w)|^2, a = 0.3');
axis([0 5 0 1.1]);
xlabel('w');


%% (E)

% (A)

% Define Constants 
w0 = 2; w1 = 3; a = 0.9; 

% Set time range 
Tmax = 100; T = Tmax/2000; t = 0:T:Tmax;

% Define function x 
x = sin(w1*t) .* F(t,30) + ...
    sin(w0*t) .* F(t-30, 40) + ... 
    sin(w1*t) .* F(t-70, 30); 

% Define constant wr 
wr = sqrt(w0^2 - (a^2)/4);

% Define g(t) 
g = a*exp(-a*t/2).*(cos(wr*t) - (a/(2*wr))*sin(wr*t));

% y temp 
y_temp = conv(g,x); 

% trim the convolution
y_temp = y_temp(1:length(t)); 
y_conv = x - T*y_temp;

% Plot the input signal
figure
plot(t, x, 'b'); 
title('input signal, x(t)');
xlabel('t'); 
axis([0 100 -2 2])

% Construct Transfer Function 
syms s; 
j = sqrt(-1);
H_ = abs((w1^2 - w0^2)/(w0^2 - w1^2 + a*j*w1))


% Allocate Vectors to represent transfer function
A = 0:T:30; 
A = zeros(size(A));
A(A == 0) = H_;

B = 0:T:40; 
B = zeros(size(B));
B(B == 0) = NaN;

C = [A B A];
C = C(1:length(t));

A = 0:T:30; 
A = zeros(size(A));
A(A == 0) = -H_;

D = [A B A];
D = D(1:length(t));

% Plot output signal via the convolutional method and the magnitudes
figure
plot(t, y_conv, 'r', t, C, 'b', t, D, 'b');
title('output signal, y(t), conv method, a = 0.9');
legend('y(t)', '|H(jw1)|');
xlabel('t'); 
axis([0 100 -2 2])

% (B)

% Define Stable Pole
p = -a/(2+j*wr); 
tau = log(100)/abs((real(p)))

disp('The time constant tau at 40-dB is not consistent with the transients observed in the plot of y(t). At 20 the output does not approach 0.');


% (C)

% Set up tf for lsim
s = tf('s'); 
H = (s^2 + w0^2)/(s^2 + a*s + w0^2);

% Perform lsim
y_lsim = lsim(H, x, t);

% Plot lsim
figure;
plot(t, y_lsim, 'r', t, C, 'b', t, D, 'b');
title('output signal, y(t), lsim method, a = 0.9');
legend('y(t)', '|H(jw1)|');
xlabel('t'); 
axis([0 100 -2 2])

% Calculate Error
lnorm = norm(y_lsim);
convnorm = norm (y_conv);

error = abs(100*(convnorm - lnorm)/convnorm)


% (D)

% range of w 
w = 0:.05:5;

% Create function
mag_squared = (((w.^2 - w0^2).^2)./((w.^2-w0^2).^2 + (a^2)*(w.^2)));

% Point values
mag_squared_w0 = (((w0.^2 - w0^2).^2)./((w0.^2-w0^2).^2 + (a^2)*(w0.^2)));
mag_squared_w1 = (((w1.^2 - w0^2).^2)./((w1.^2-w0^2).^2 + (a^2)*(w1.^2)));

% Calculate the 3db frequencies 
w_p = sqrt(w0^2 + (a^2)/2 + a*sqrt(w0^2 + (a^2)/4))
w_n = sqrt(w0^2 + (a^2)/2 - a*sqrt(w0^2 + (a^2)/4))

w_p - w_n

line = w_n:0.05:w_p; 
mag_half = 0.5;

% Plot
figure;
plot(w, mag_squared, 'b', w0, mag_squared_w0, 'r.', ... 
     w1, mag_squared_w1, 'r.', line, ones(size(line))*mag_half, 'r');
txt2 = '  w1';
text(w1, mag_squared_w1, txt2);
txt3 = '  3-dB width'; 
text(w_p, 1/2, txt3);
title('notch filter, |H(w)|^2, a = 0.9');
axis([0 5 0 1.1]);
xlabel('w');



%% 3 
clear;

%% (A)
% Load Data File into Y
Y = load('lab2.dat'); 

% Convert notch frequency and alpha to radians/sec
w0 = 60*2*pi;
a = 2*pi*1.5;

% Calculate 3-dB frequencies
w_p = sqrt(w0^2 + (a^2)/2 + a*sqrt(w0^2 + (a^2)/4))
w_n = sqrt(w0^2 + (a^2)/2 - a*sqrt(w0^2 + (a^2)/4))

% Calculate Q
Q = w0/a 

% Calculate Tau
wr = sqrt(w0^2 - (a^2)/4);
p = -a/2 + j*wr; 
tau = log(100)/(abs(real(p)))

%% (B)

% Select appropriate column vectors
t = Y(:, 1);
st = Y(:, 2);
x = Y(:, 3);

% plot s(t)
figure;
plot(t, st);
title('noise-free ECG, s(t)');
xlabel('t (sec)');
axis([0 1.5 -1.5, 1.5]);

% plot x(t)
figure
plot(t, x, 'b');
title('ECG + 60 Hz interference, x(t)');
xlabel('t (sec)');
axis([0 1.5 -1.5, 1.5]);


%% (C)
% Set up tf for lsim
s = tf('s'); 
H = (s^2 + w0^2)/(s^2 + a*s + w0^2);
A = lsim(H, st, t);
B = lsim(H, x, t);

% Processed Plot
plot(t, A, 'black', t, B, 'r');
title('processed ECG, y(t)');
xlabel('t (sec)');
axis([0 1.5 -1.5, 1.5]);
legend('y(t)', 's(t)');

%% (D) 
% convert frequency limits to radians

f = 0:0.05:120;

h_mag = (((2*pi*f).^2 - w0^2).^2)./(((2*pi*f).^2 - w0^2).^2 + (a^2)*(2*pi*f).^2);

figure;
plot(f, h_mag, 'b');
title('notch filter magnitude response, |H(f)|^2');
xlabel('f (Hz)');
axis([0 120 0, 1.2]);


