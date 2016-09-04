function [ u, s ,r ] = lcp_nash( PayOffTable, xStart )
% computes a Nash equilibrium
% INPUT:
%   PayOffTable: [m x n x 2] matrix with the pay-offs for both players,
%                with m, the number of actions for player 1 and n, the number
%                of actions for player 2.
%   xStart:      the initial vector for the variables x.
% OUTPUT:
%   u:           the utilities for both players
%   s:           the strategies for both players
%   r:           the slack variables
    
    b = ...
    
    A = ...
    
    %  ip_lcp assumes the following formulation of the lcp problem,
    %
    %      (A*x - y + b)_i  = 0,
    %              x_i*y_i  = 0,
    %            x_i, y_i  >= 0,
    %
    %  where A and b are user defined and y are the slack variables
    %  and x is a local solution.
    
    [x, y] = ip_lcp(A,b,xStart);
    
    u = ...
    s = ...
    r = ...
end

