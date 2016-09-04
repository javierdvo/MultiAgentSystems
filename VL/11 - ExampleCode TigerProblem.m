clear all;
close all;

T = 10;
gamma = 0.95;
%Definne POMDP

% Observation matrix (only valid if listen is taken)
% the rows of O contain p(o_i | s) for all s
O = [0.85, 0.15; 0.15, 0.85];

%Define transition matrix (the tiger stays where it is, independent of
%action)
% the rows of P contain p(s_{t+1} = s_i | s) for all s
P = [1 0; 0 1];

% initial belief... we do not know where the tiger is!
b = [0.5; 0.5];

%alpha for opening left door
Lambda_A{1} = [-100; 10];
%alpha for opening right door
Lambda_A{2} = [10; -100];
%alpha for listening
Lambda_A{3} = [-1; -1];

% Omega_T ... alpha vectors for last time step are given by alpha vectors
% of reward function

Omega{T} = Lambda_A;

%belief space samples for visualization
b_s(1,:) = linspace(0,1,100);
b_s(2,:) = 1 - b_s(1,:);

fprintf('Size of Omega, iteration %d: %d\n', 0, length(Omega{T}));

for t = T-1 : -1 : 1
    %visualize linear functions for timestep t + 1
    

    V_alpha = zeros(length(Omega{t+1}), 100);
    for i = 1:length(Omega{t+1})
        V_alpha(i,:) = b_s' * Omega{t+1}{i};        
    end
    V_max = max(V_alpha);
    figure(1);
    clf;
    plot(b_s(1,:), V_alpha, 'g');
    hold on;
    plot(b_s(1,:), V_max, 'b', 'LineWidth', 2);    
    
    % Very simple pruning strategy, just throw aways all alpha where the
    % line is below the V_max - 0.1 everywhere
       
    pruning = true(length(Omega{t+1}),1);
    for i = 1:length(Omega{t+1})
        pruning(i) = all(V_alpha(i,:) < V_max - 0.01);        
    end
    plot(b_s(1,:), V_alpha(~pruning,:), 'k');    
    Omega{t+1} = Omega{t+1}(~pruning);
    fprintf('After pruning: %d\n', length(Omega{t+1}));
    pause;           
    
    % Now construct alpha vectors for previous time step
    % There is only one action with observations, listen
    % Listen has 2 observations
    for i = 1:length(Omega{t+1})
        Omega_a3o1{t}{i} = (Omega{t + 1}{i}' * diag(O(1,:))*P)';
        Omega_a3o2{t}{i} = (Omega{t + 1}{i}' * diag(O(2,:))*P)';        
    end
    
    % Now we have to create all combinations of the alpha vectors 
    counter = 1;
    for i = 1:length(Omega_a3o1{t})
        for j = 1:length(Omega_a3o2{t})
            Omega_a3{t}{counter} = Omega_a3o1{t}{i} + Omega_a3o2{t}{j};            
            % Also add reward alpha here (set is only of size 1)
            Omega_a3{t}{counter} = gamma * Omega_a3{t}{counter} + Lambda_A{3};
            counter = counter + 1;
        end    
    end
    
    % For the other 2 actions, the future value is zero (you get eaten or
    % you find the treasure)
    Omega_a1{t} = Lambda_A{1};
    Omega_a2{t} = Lambda_A{2};
    
    Omega{t} = [Omega_a1{t}, Omega_a2{t}, Omega_a3{t}];
    fprintf('Size of Omega, iteration %d: %d\n', T - t, length(Omega{t}));
end