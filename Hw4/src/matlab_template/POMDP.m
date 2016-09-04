classdef POMDP < handle

    properties
       	T % horizon
        O % Observation matrix
        P % Transition matrix
        R % Reward matrix
        terminalActions
        numObservations
        numActions
        numStates
    end
    
    methods
        function obj = POMDP(T, O, R, P, terminalActions)
            [obj.numStates, obj.numActions, ~] = size(P);
            obj.T = T;
            obj.O = O;
            obj.R = R;
            obj.P = P;
            obj.terminalActions = terminalActions;
            obj.numObservations = size(O,2);
        end
        
        function isTerminal = isTerminalAction(obj, action)
            isTerminal = any(action==obj.terminalActions);
        end
        
        function nextState = sampleNextState(obj, s, a)
            p_next = obj.P(s,a,:);
            randomNum = rand();
            accSum=0;
            for i=1:numel(p_next)
                accSum = accSum + p_next(:,:,i);
                if accSum >= randomNum
                    nextState = i;
                    break;
                end
            end
        end
        
        function observation = sampleObservation(obj, s, a)
            p_observation = obj.O(s,a,:);
            randomNum = rand();
            accSum=0;
            for i=1:numel(p_observation)
                accSum = accSum + p_observation(i);
                if accSum >= randomNum
                    observation = i;
                    break;
                end
            end
        end
        
        function belief = getBeliefFromHistory(obj, history, initialBelief)
            belief = initialBelief;
            for i=1:numel(history)
                belief = obj.updateBelief(belief, history(i).action, history(i).observation);
            end
        end
        
        function newBelief = updateBelief(obj, oldBelief, action, observation)
            %newBelief = oldBelief;
            newBelief = obj.O(:,action,observation) .* (permute(obj.P(:,action,:),[1,3,2]) * oldBelief);
            newBelief = newBelief ./ sum(newBelief);
        end
        
        function reward = getImmediateReward(obj, s, a)
            reward = obj.R(s,a);
        end
    end
    
end

