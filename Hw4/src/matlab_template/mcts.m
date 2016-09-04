% Horizon
T= ...

% Observation probabilities (state x action x observation -> probability)
O = zeros(2,3,3);

% Reward (state x action -> reward)
R = zeros(2,3)

% Transition probabilities (state x action x nextState -> probability)
P = zeros(2,3,2); 

pomdp = POMDP(T, O, R, P, terminalActions);
terminalActions = [];
history = [];

rootNode = HistoryNode([], history,  pomdp.getBeliefFromHistory(history, [0.5;0.5]), pomdp);
for i=1:1e5
    % 1.select
    [selected, state, rewards, alreadyExpanded] = rootNode.select([],[]);
    
    % 2.expand
    if ~alreadyExpanded       
        [expanded, state, immediateReward] = selected.expand(state);
        rewards(end+1) = immediateReward;
    else
        expanded = selected;
    end
    
    % 3.simulate
    rewards(end+1) = expanded.rollout(state);
    
    % 4.update
    expanded.update(rewards, numel(rewards));
end
