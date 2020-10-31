function in = ResetFcn(in)

%% This function is run everytime one episode of training finishes. It specifies the starting specifications for the environment

global x_pos y_pos ang_start

x_pos = 1.2; % X start position
y_pos = 5.6; % Y start position
ang_start = pi-0.1; % Angle start position

end
