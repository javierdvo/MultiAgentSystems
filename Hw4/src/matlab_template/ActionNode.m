classdef ActionNode < TreeNode

    properties
        action
    end
    
    methods
        function obj = ActionNode(parent, action, pomdp)
            obj@TreeNode(parent, pomdp);            
            obj.action = action;
            obj.children = cell(1, pomdp.numObservations);
        end
        
        function [treeNode, state, reward, alreadyExpanded] = select(obj, sampledState, reward)
            a = obj.action;
            state = obj.pomdp.sampleNextState(sampledState,a);
            newObservation = obj.pomdp.sampleObservation(state, a);
            if ~isempty(obj.children{newObservation})
                % the observation is already in the tree 
                [treeNode, state, reward, alreadyExpanded] = obj.children{newObservation}.select(state, reward);
            else
                % the observation is new, hence a new HistoryNode has to be
                % created
                alreadyExpanded = true;
                newBelief = obj.pomdp.updateBelief(obj.parent.beliefState, obj.action, newObservation);
                newHistory = obj.parent.history();
                newHistory(end+1).action = a;
                newHistory(end).observation = newObservation;
                treeNode = HistoryNode(obj, newHistory, newBelief, obj.pomdp);
                obj.children{newObservation} = treeNode;
            end
        end
        
        function [] = update(obj, reward, idx)
            obj.reward = obj.reward + sum(reward(idx:end));
            obj.visitationCount = obj.visitationCount + 1;
            obj.parent.update(reward, idx);
        end
        
    end   
end

