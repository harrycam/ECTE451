% Helper function to reset walking robot simulation with different initial conditions
%
% Copyright 2019 The MathWorks, Inc.

function in = ResetFcn(in)

global x_pos y_pos ang_start
    
pos = [2.1092    5.0546;
       2.625     2.0199;
       -0.84563  3.1904;
       -0.061246 1.5213;
       -0.84563  5.0762;
       3.4094    3.754;
       1 6;
       -1 5;
       1 1;
       -1 1;
       3 1;
       3.5 3;
       3.5 3.5;
       0.5 4.5;
       3 5;];
   
ang = -3.14:0.5:3.14;

pos_i = randi(15);
ang_i = randi(length(ang));

x_pos = pos(pos_i,1);
%x_pos = 1.6; (3)
x_pos = 1.2;

y_pos = pos(pos_i,2);
%y_pos = 1.6; (3)
y_pos = 5.6;
%ang_start = ang(ang_i);
ang_start = pi-0.1;
end