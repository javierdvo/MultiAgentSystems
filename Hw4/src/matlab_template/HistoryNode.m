classdef HistoryNode < TreeNode
    
    properties
        history
        beliefState
        isTerminal = false
        % is this node initialized or constructed by empty constructor?
        isInitialized = false
    end
    
    methods
        function obj = HistoryNode(parent, history, beliefState, pomdp)
            if nargin == 0
                return
            end
            obj@TreeNode(parent, pomdp);
            obj.children = cell(1, 0);
            obj.history = history;
            obj.beliefState = beliefState;
            obj.isInitialized = true;
            if ~isempty(obj.history) && (numel(obj.history) >= pomdp.T || pomdp.isTerminalAction(history(end).action))
                obj.isTerminal = true;
            end
        end
        
        function reward = rollout(obj, sampledState)
            reward = 0;
            if obj.isTerminal
                return;
            end
            
            s = sampledState;
            for i=numel(obj.history)+1:obj.pomdp.T
		% ...
            end
        end
        
        function [newChild, state, reward] = expand(obj, sampledState)
            assert(~obj.isFullyExpanded);
            
            nextAction = numel(obj.children)+1;
            actionNode = ActionNode(obj, nextAction, obj.pomdp);
            obj.children{end+1} = actionNode;
            reward = obj.pomdp.getImmediateReward(sampledState,nextAction);
            if numel(obj.children) == obj.pomdp.numActions
                obj.isFullyExpanded = true;
            end
            [newChild, state, reward] = actionNode.select(sampledState, reward);
        end
        
        function [treeNode, state, pastReward, alreadyExpanded] = select(obj, sampledState, pastReward)
            alreadyExpanded = false;
            
            if isempty(sampledState)
                if rand()<=obj.beliefState(1)
                    sampledState = 1;
                else
                    sampledState = 2;
                end
            end
            
            if obj.isTerminal
                treeNode = obj;
                state = sampledState;
                alreadyExpanded = true;
                return;
            elseif ~obj.isFullyExpanded
                treeNode = obj;
                state = sampledState;
                return;
            end
            
            bestUcbIdx = ...;
                        
            pastReward(end+1) = obj.pomdp.getImmediateReward(sampledState, bestUcbIdx);
            [treeNode, state, pastReward, alreadyExpanded] = obj.children{bestUcbIdx}.select(sampledState, pastReward);
        end
        
        function [] = update(obj, reward, idx)
            obj.reward = obj.reward + sum(reward(idx:end));
            obj.visitationCount = obj.visitationCount + 1;
            if ~isempty(obj.parent)
                obj.parent.update(reward, idx-1);
            end
        end
        
        function [] = printHistory(obj)
            for i =1:numel(obj.history)
                if obj.history(i).action == 1
                    fprintf('-> opened left door ');
                elseif obj.history(i).action == 2
                    fprintf('-> listened ');
                else
                    fprintf('-> opened right door ');
                end
                if obj.history(i).observation == 2
                    fprintf(' (heard something from the left) \n');
                elseif obj.history(i).observation == 3
                    fprintf(' (heard something from the right) \n');
                else fprintf('\n');
                end
            end
            
        end
    end
    
end

