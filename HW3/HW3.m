CPU = zeros(1,6);
GPU = [0.027808 0.026784 0.026656 0.118048 1.329184 10.335424];
Shared = [0.011744 0.011616 0.015328 0.055520 0.768672 4.608000 ];
N = ones(1,6);
for i = 1:length(N)
    N(1,i)=N(1,i)*10^i;
end
plot(log10(N),GPU,log10(N),Shared,log10(N),CPU)
legend('GPU Time','GPU with Shared Memory','CPU Time')
xlabel('10 to the power of')
ylabel('Time in ms')
title('HW3P2')
axis([1 6 -0.5 12])