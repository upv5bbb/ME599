close all, clear all,clc
%% P2
CPU = zeros(1,6);
GPU = [0.027808 0.026784 0.026656 0.118048 1.329184 10.335424];
Shared = [0.011744 0.011616 0.015328 0.055520 0.768672 4.608000 ];
N = ones(1,6);
for i = 1:length(N)
    N(1,i)=N(1,i)*10^i;
end
figure(1)
plot(log10(N),GPU,log10(N),Shared,log10(N),CPU)
legend('GPU Time','GPU with Shared Memory','CPU Time')
xlabel('10 to the power of')
ylabel('Time in ms')
title('HW3P2')
axis([1 6 -0.5 12])

%% P3
true=1.0437619617094618;
steps=[10 100 200 2000 20000];
RK4err = [0.063235 0.000002 0 0.000001 0.000004];
RKPercentage = 100*RK4err/true;
RK4Time = [0.754784 5.292329 9.725664 94.600288 936.079163];
EUerr=[0.557895 0.256829 0.023683 0.002346];
EUPercentage = 100*EUerr/true;
EUTime=[1.064736  2.5166 23.15788 229.879456];

figure(2)
plot(RK4Time)
title('EXE Time VS. Steps')
xlabel('execution time at 10 100 200 2000 20000 steps')
figure(3)
semilogy(steps,RKPercentage)
title('ABS error vs Steps')
figure(4)
semilogy(steps,RKPercentage./RK4Time)
title('abs err/ms vs steps')