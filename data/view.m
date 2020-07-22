clc;clear;

load('simulation_result.mat');
matrix = double(matrix);
matrix(:,1) = (matrix(:,1) - 395)*0.02;
matrix(:,2) = (matrix(:,2) - 335)*0.02;
scatter(matrix(:,1),matrix(:,2));
hold on;

ee = zeros(1,100);
for i = 1:100
    ee(i) = sqrt(matrix(i,1)^2+matrix(i,2)^2);
end
ee = sort(ee);
CEP = ee(50)

theta=0:2*pi/3600:2*pi;
Circle1=CEP*cos(theta);
Circle2=CEP*sin(theta);
plot(Circle1,Circle2,'m','Linewidth',1);
plot(0, 0, '.', 'Linewidth',2);
% axis([-0.8 0.4 -0.2 1]);
axis([-0.3 0.3 -0.3 0.3]);
grid on;
insert arrow;
hold off;