function [ x ] = maxmin( P )
%   minmax with slack variables
%   INPUT:
%       P : [m x n] Matrix with the payoff for player 1. The actions of
%       player 1 correspond to the rows of P.
%   OUTPUT:
%       x : vector with maxmin value and maxmin strategy
%
    
    f = ...;
    
    A = ...;
    
    b = ...;
    
    Aeq = ...;
    
    beq = ...;
    
    x = linprog(f,A,b,Aeq,beq);
    
end

