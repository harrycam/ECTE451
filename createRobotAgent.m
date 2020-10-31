%% This is the main file to be run once a trained agent has been saved to the workspace

%% SET UP ENVIRONMENT

clc
%clear
close all

global x_pos y_pos ang_start

Ts = 0.025; % Agent sample time
Tf = 30;    % Simulation end time

% Speedup options
useFastRestart = false;

% Create the observation info
numObs = 6; % Number of observation inputs
observationInfo = rlNumericSpec([numObs 1]);
observationInfo.Name = 'observations';

% create the action info
numAct = 2; % Number of action outputs
actionInfo = rlNumericSpec([numAct 1],'LowerLimit',-2,'UpperLimit', 2);
actionInfo.Name = 'wheel_velocity';
% Environment

mdl = 'RoboBlockRL'; % Name of Simulink file
load_system(mdl);
blk = [mdl,'/RL Agent'];
env = rlSimulinkEnv(mdl,blk,observationInfo,actionInfo);
env.ResetFcn = @(in)ResetFcn(in); % Initialise reset function to specify starting location
%function to allow changing of parameters on reset - investigate this

if ~useFastRestart
   env.UseFastRestart = 'off';
end
%% CREATE NEURAL NETWORKS

createDDPGNetworks; % Create NN representations of actor and critic
                     
%% CREATE AND TRAIN AGENT

createDDPGOptions; % Options for RL
agent = rlDDPGAgent(actor,critic,agentOptions); % Initialise learning agent from NNs
trainingResults = train(agent,env,trainingOptions) % Train agent with RL algorithm
%exp1 = sim(agent,env) % Simulate trained agent

%% SAVE AGENT

reset(agent); % Clears the experience buffer
curDir = pwd;
saveDir = 'savedAgents';
cd(saveDir)
save(['trainedAgent_2D_' datestr(now,'mm_DD_YYYY_HHMM')],'agent');
save(['trainingResults_2D_' datestr(now,'mm_DD_YYYY_HHMM')],'trainingResults');
cd(curDir)
