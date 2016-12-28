% Daniel Revie
% 11/30/2016
% Lab 5
close all;
clear;
%% 1

%% (A)
% -----------------------------------
% Define constants
T = 1;
M = 10;
tau = T;


w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = 0:.003:3*T;
fm = 0; % zeroes length t
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end

c1 = 1/(4*1i);
fmc1 = 2*real(c1.*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;

for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


figure
plot(t/T, fm, 'b-', t/T, ymst,'r-');
grid on;
title('half-wave rectifier, M = 10, t = 1T');
xlabel('t/T')
axis([0 3 -0.2 1.2])
legend('half-wave f(t)', 'steady output')


% -----------------------------------
M = 10;
tau = 5*T;

% Define constants
T = 1;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = 0:.003:3*T;
fm = 0; % zeroes length t
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end

c1 = 1/(4*1i);
fmc1 = 2*real(c1.*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;

for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


figure
plot(t/T, fm, 'b-', t/T, ymst,'r-');
grid on;
title('half-wave rectifier, M = 10, t = 5T');
xlabel('t/T')
axis([0 3 -0.2 1.2])
legend('half-wave f(t)', 'steady output')

% -----------------------------------
M = 10;
tau = 10*T;

% Define constants
T = 1;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = 0:.003:3*T;
fm = 0; % zeroes length t
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end

c1 = 1/(4*1i);
fmc1 = 2*real(c1.*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;

for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


figure
plot(t/T, fm, 'b-', t/T, ymst,'r-');
grid on;
title('half-wave rectifier, M = 10, t = 10*T');
xlabel('t/T')
axis([0 3 -0.2 1.2])
legend('half-wave f(t)', 'steady output')

% -----------------------------------
M = 30;
tau = T;

% Define constants
T = 1;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = 0:.003:3*T;
fm = 0; % zeroes length t
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end

c1 = 1/(4*1i);
fmc1 = 2*real(c1.*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;

for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


figure
plot(t/T, fm, 'b-', t/T, ymst,'r-');
grid on;
title('half-wave rectifier, M = 30, t = 1T');
xlabel('t/T')
axis([0 3 -0.2 1.2])
legend('half-wave f(t)', 'steady output')

% -----------------------------------
M = 30;
tau = 5*T;

% Define constants
T = 1;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = 0:.003:3*T;
fm = 0; % zeroes length t
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end

c1 = 1/(4*1i);
fmc1 = 2*real(c1.*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;

for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


figure
plot(t/T, fm, 'b-', t/T, ymst,'r-');
grid on;
title('half-wave rectifier, M = 30, t = 5T');
xlabel('t/T')
axis([0 3 -0.2 1.2])
legend('half-wave f(t)', 'steady output')

% -----------------------------------
M = 30;
tau = 10*T;

% Define constants
T = 1;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = 0:.003:3*T;
fm = 0; % zeroes length t
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end

c1 = 1/(4*1i);
fmc1 = 2*real(c1.*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;

for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


figure
plot(t/T, fm, 'b-', t/T, ymst,'r-');
grid on;
title('half-wave rectifier, M = 30, t = 10T');
xlabel('t/T')
axis([0 3 -0.2 1.2])
legend('half-wave f(t)', 'steady output')

%% (b)
T = 1;
M = 30; tau = 5*T;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);

t = 0:0.05:24*T;


fm = 0; 
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
fmc1 = 2*real(c1*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


A = -(tau^-1*p(-tau^-1))/(exp(T/tau) - 1);
ym = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ym = ym + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ym = ym + c0*H0 + ymc1 + A*exp(-t/tau);

figure
plot(t/T, fm, 'b-', t/T, ymst,'r-', t/T, ym, 'k-');
grid on;
title('half-wave rectifier, M = 30, t = 5T');
xlabel('t/T')
axis([0 24 -0.2 1.2])
legend('half-wave f(t)', 'steady output', 'exact output')

%--------------------------------------------
T = 1;
M = 30; tau = 10*T;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);

t = 0:0.05:24*T;
t = linspace(0, 24*T, 1001);

fm = 0; 
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
fmc1 = 2*real(c1*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


A = -(tau^-1*p(-tau^-1))/(exp(T/tau) - 1);
ym = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ym = ym + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ym = ym + c0*H0 + ymc1 + A*exp(-t/tau);

figure
plot(t/T, fm, 'b-', t/T, ymst,'r-', t/T, ym, 'k-');
grid on;
title('half-wave rectifier, M = 30, t = 10T');
xlabel('t/T')
axis([0 24 -0.2 1.2])
legend('half-wave f(t)', 'steady output', 'exact output')


%% (C)
T = 1;
M = 30; tau = 5*T;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = linspace(0, 24*T, 1001);

fm = 0; 
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
fmc1 = 2*real(c1*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


A = -(tau^-1*p(-tau^-1))/(exp(T/tau) - 1);
ym = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ym = ym + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ym = ym + c0*H0 + ymc1 + A*exp(-t/tau);



num = [0 0 1];
den = [0 tau 1];
H = tf(num,den);

ylsim = lsim(H, fm, t);

error_5T = norm(ylsim'-ym)


%----------------------------------------------------------------
T = 1;
M = 30; tau = 10*T;

w0 = 2*pi/T;
c0 = (1/pi);
p = @(s) (w0*(1 + exp(-s*T/2)))/(s^2 + w0^2);


t = linspace(0, 24*T, 1001);

fm = 0; 
% Summation
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    fm = fm + 2*real(ck.*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
fmc1 = 2*real(c1*exp(1i*1*w0*t));
fm = fm + c0 + fmc1;                           % Final value

% Start calculating ymst

H0 = 1;
H = @(s) (1/(1+s*tau));

ymst = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ymst = ymst + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymstc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ymst = ymst + c0*H0 + ymstc1;


A = -(tau^-1*p(-tau^-1))/(exp(T/tau) - 1);
ym = 0;
for k = 2:M
    ck = (1/T)*p(1i*k*w0);
    ym = ym + 2*real(ck*H(1i*k*w0)*exp(1i*k*w0*t));
end
c1 = 1/(4*1i);
ymc1 = 2*real(c1.*H(1i*1*w0).*exp(1i*1*w0*t));
ym = ym + c0*H0 + ymc1 + A*exp(-t/tau);


num = [0 0 1];
den = [0 tau 1];
H = tf(num,den);

ylsim = lsim(H, fm, t);

error_10T = norm(ylsim'-ym)


%--------------------------------------------------------------------------
%% 2

%% (A)
omega_0 = 0.2*pi;
omega_1 = 0.1*pi;
omega_2 = 0.3*pi;
L = 200;

s = @(n) sin(omega_0*n);
v = @(n) sin(omega_1*n) + sin(omega_2*n);
x = @(n) s(n) + v(n);

n = linspace(0, L-1, 1001);

plot(n, x(n), 'r-', n, s(n), 'k:');
title('x(n) and s(n)')
xlabel('time samples, n')
axis([0 200 -3 3])
grid on;

%% (B)
clear; 
L = 200;
n2 = 0:L-1;

Omega0 = 0.2*pi;
Omega1 = 0.1*pi;
Omega2 = 0.3*pi;

s = sin(Omega0*n2);
v = sin(Omega1*n2) + sin(Omega2*n2);
x = s + v;

omega_a = 0.15;
omega_b = 0.25;
M = 75;
n = 0:2*M;
A = 0.54+0.46*cos(pi*(n-M)/M);
B = sinc(omega_b.*(n-M))*omega_b;
C = sinc(omega_a.*(n-M))*omega_a;
h = A.*(B-C);

% filter x through h 
y = filter(h,1,x);

figure
plot(n2,y, 'r-', n2,s,':k');
ylim([-2,2]);
xlabel('n')
title('s(n) filtered x(n), 2M = 150')
grid on;

%% (C)
% This time filter v through h 
y = filter(h, 1, v);
figure
plot(n2, v, 'k:', n2, y, 'r-');
xlabel('n')
title('v(n) and filtered v(n), 2M = 150')
grid on;

%% (D) 
omega_0 = 0.2*pi;
omega_1 = 0.1*pi;
omega_2 = 0.3*pi;
w = pi*linspace(0, 0.4, 801); 
H = abs(freqz(h, 1, w)); 

figure
plot(w/pi, H, 'b-'); hold on;

arrow1 = annotation('arrow'); arrow1.Parent = gca;
arrow1.X = [omega_1/pi omega_1/pi]; arrow1.Y = [0 .1]; 

arrow0 = annotation('arrow'); arrow0.Parent = gca;
arrow0.X = [omega_0/pi omega_0/pi]; arrow0.Y = [0 .1]; 

arrow2 = annotation('arrow'); arrow2.Parent = gca;
arrow2.X = [omega_2/pi omega_2/pi]; arrow2.Y = [0 .1];

axis([0 0.4 0 1.1])
title('Magnitude Response |H(w)|, 2M = 150')
xlabel('w in units of pi')
grid on; 
hold off;

figure 
plot(w/pi, 20*log10(H), 'b-'); hold on;

arrow1 = annotation('arrow'); arrow1.Parent = gca;
arrow1.X = [omega_1/pi omega_1/pi]; arrow1.Y = [-100 -80]; 

arrow0 = annotation('arrow'); arrow0.Parent = gca;
arrow0.X = [omega_0/pi omega_0/pi]; arrow0.Y = [-100 -80]; 

arrow2 = annotation('arrow'); arrow2.Parent = gca;
arrow2.X = [omega_2/pi omega_2/pi]; arrow2.Y = [-100 -80];

axis([0 0.4 -100 5])
title('Magnitude Response |H(w)|, 2M = 150')
ylabel('dB')
xlabel('w in units of pi')
grid on;
hold off;


%% (E)
% (A)
omega_0 = 0.2*pi;
omega_1 = 0.1*pi;
omega_2 = 0.3*pi;
L = 200;

s = @(n) sin(omega_0*n);
v = @(n) sin(omega_1*n) + sin(omega_2*n);
x = @(n) s(n) + v(n);

n = linspace(0, L-1, 1001);

figure
plot(n, x(n), 'r-', n, s(n), 'k:');
title('x(n) and s(n)')
xlabel('time samples, n')
axis([0 200 -3 3])
grid on;

% (B)
clear; 
omega_0 = 0.2*pi;
omega_1 = 0.1*pi;
omega_2 = 0.3*pi;
L = 200;
n2 = 0:L-1;

Omega0 = 0.2*pi;
Omega1 = 0.1*pi;
Omega2 = 0.3*pi;

s = sin(Omega0*n2);
v = sin(Omega1*n2) + sin(Omega2*n2);
x = s + v;

omega_a = 0.15;
omega_b = 0.25;
M = 100;
n = 0:2*M;
A = 0.54+0.46*cos(pi*(n-M)/M);
B = sinc(omega_b.*(n-M))*omega_b;
C = sinc(omega_a.*(n-M))*omega_a;
h = A.*(B-C);

% filter x through h 
y = filter(h,1,x);

figure
plot(n2,y, 'r-', n2,s,':k');
ylim([-2,2]);
xlabel('n')
title('s(n) filtered x(n), 2M = 200')
grid on;

% (C)
% This time filter v through h 
y = filter(h, 1, v);
figure
plot(n2, v, 'k:', n2, y, 'r-'); 
xlabel('n')
title('v(n) and filtered v(n), 2M = 200')

% (D) 
w = pi*linspace(0, 0.4, 801); 
H = abs(freqz(h, 1, w)); 

figure
plot(w/pi, H, 'b-'); hold on;

arrow1 = annotation('arrow'); arrow1.Parent = gca;
arrow1.X = [omega_1/pi omega_1/pi]; arrow1.Y = [0 .1]; 

arrow0 = annotation('arrow'); arrow0.Parent = gca;
arrow0.X = [omega_0/pi omega_0/pi]; arrow0.Y = [0 .1]; 

arrow2 = annotation('arrow'); arrow2.Parent = gca;
arrow2.X = [omega_2/pi omega_2/pi]; arrow2.Y = [0 .1]; 

axis([0 0.4 0 1.1])
title('Magnitude Response |H(w)|, 2M = 200');
xlabel('w in units of pi');
grid on;

hold off;

figure 
plot(w/pi, 20*log10(H), 'b-'); hold on;
axis([0 0.4 -100 5]);

arrow1 = annotation('arrow'); arrow1.Parent = gca;
arrow1.X = [omega_1/pi omega_1/pi]; arrow1.Y = [-100 -80]; 

arrow0 = annotation('arrow'); arrow0.Parent = gca;
arrow0.X = [omega_0/pi omega_0/pi]; arrow0.Y = [-100 -80]; 

arrow2 = annotation('arrow'); arrow2.Parent = gca;
arrow2.X = [omega_2/pi omega_2/pi]; arrow2.Y = [-100 -80]; 

title('Magnitude Response |H(w)|, 2M = 200');
ylabel('dB');
xlabel('w in units of pi');
grid;
hold off;

%% (F)
clear;
M = 75;
omega_0 = 0.2*pi;
omega_1 = 0.1*pi;
omega_2 = 0.3*pi;
omega_a = 0.15*pi;
omega_b = 0.25*pi;

w = linspace(0, 0.4*pi, 1001); 

h_2 = @(n) (omega_b/pi.*(sinc(omega_b*(n-M)/pi)) - omega_a/pi.*(sinc(omega_a*(n-M)/pi)));

for k = 0:2*M

H_2(k+1,:) = h_2(k)*exp(-1i*w*k);

end

H_2 = abs(sum(H_2));

figure
plot(w/pi,H_2, 'b-'); hold on;

grid; ylim([0 1.1]);

arrow1 = annotation('arrow'); arrow1.Parent = gca;
arrow1.X = [omega_1/pi omega_1/pi]; arrow1.Y = [0 0.1]; 

arrow0 = annotation('arrow'); arrow0.Parent = gca;
arrow0.X = [omega_0/pi omega_0/pi]; arrow0.Y = [0 0.1]; 

arrow2 = annotation('arrow'); arrow2.Parent = gca;
arrow2.X = [omega_2/pi omega_2/pi]; arrow2.Y = [0 0.1]; 

title('Magnitude Response |H(w)|, 2M = 150');
xlabel('w is in units of pi');
axis([0 0.4 0 1.2]);
hold off;

clear; 
omega_0 = 0.2*pi;
omega_1 = 0.1*pi;
omega_2 = 0.3*pi;
omega_a = 0.15*pi;
omega_b = 0.25*pi;

w = linspace(0, 0.4*pi, 1001); 
M = 100; 
h_2 = @(n) (omega_b/pi.*(sinc(omega_b*(n-M)/pi)) - omega_a/pi.*(sinc(omega_a*(n-M)/pi)));

for k = 0:2*M

H_2(k+1,:) = h_2(k)*exp(-1i*w*k);

end


H_2 = abs(sum(H_2));

figure
plot(w/pi,H_2, 'b-'); hold on;

grid; ylim([0 1.1]);

arrow1 = annotation('arrow'); arrow1.Parent = gca;
arrow1.X = [omega_1/pi omega_1/pi]; arrow1.Y = [0 0.1]; 

arrow0 = annotation('arrow'); arrow0.Parent = gca;
arrow0.X = [omega_0/pi omega_0/pi]; arrow0.Y = [0 0.1]; 

arrow2 = annotation('arrow'); arrow2.Parent = gca;
arrow2.X = [omega_2/pi omega_2/pi]; arrow2.Y = [0 0.1]; 

title('Magnitude Response |H(w)|, 2M = 200');
xlabel('w is in units of pi');
axis([0 0.4 0 1.2]);
hold off;
