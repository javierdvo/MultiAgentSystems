classdef TreeNode < handle
    
    properties
        parent
        children
        
        % Have all possible childs been added to the tree?
        isFullyExpanded
        
        % How often did we select this node?
        visitationCount = 0
        
        % How much reward did we get in total?
        reward = 0
               
        
        % pomdp settings
        pomdp
    end
    
    methods
        
        function obj = TreeNode(parent, pomdp)
            obj.pomdp = pomdp;
            obj.parent = parent;
            obj.children = TreeNode.empty;
            obj.isFullyExpanded = false;
        end
        
        function TreeNode = getBestPath(obj)
            if numel(obj.children) == 0
                TreeNode = obj;
                return;
            end
            
            maxVisits = 0;
            for i=1 : numel(obj.children)
                if ~isempty(obj.children{i}) && (obj.children{i}.visitationCount >= maxVisits)
                    maxVisits = obj.children{i}.visitationCount;
                    idx = i;
                end
            end
            TreeNode = obj.children{idx}.getBestPath();
        end

        
    end
    
end

