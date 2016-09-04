function [ r ] = reward( s1, s2, world )
% INPUT:
%       s1:     state of the runner as index
%       s2:     state of the catcher as column index
%       world:  description of the target states for the runner
% OUTPUT:
%       r:      reward for the current states


if (s1(1)==2 && s1(2)==s2) % catcher catches runner
    r = -10;
elseif world(sub2ind(size(world),s1(1),s1(2)))==1 % runner reaches target state
    r = 10;
else
    r = 0;
end

