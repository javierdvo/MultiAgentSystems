function [ x ] = maxmin( P )

    [n, m] = size(P);
    
    f = [-1 ; zeros(n,1)];
    
    A = [
        ones(m,1) -P'
        zeros(n,1) -eye(n)
        ];
    
    b = zeros(1,n+m);
    
    Aeq = [0 ones(1, n)];
    
    beq = 1;
    
    opt = optimoptions('linprog','Display','off','Algorithm','active-set');
    x = linprog(f,A,b,Aeq,beq,[],[],[],opt);
    
end

