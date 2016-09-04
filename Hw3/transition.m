function [ s1_next, s2_next ] = transition( s1, s2, a1, a2, max1, max2 )
%max1=5,max2=4
switch a1
    case 1%up case
        if s1(1)~=max1%Stability
            s1_next=[s1(1)-1, s1(2)];
        else 
            s1_next=s1;
        end
    case 2%left case
        if s1(2)~=1 %Stability
            s1_next=[s1(1), s1(2)-1];
        else 
            s1_next=s1;
        end
    case 3%left case
        if s1(2)~=max2%stability
            s1_next=[s1(1), s1(2)+1];
        else 
            s1_next=s1;
        end
    case 4
            s1_next=s1;
end



switch a2
    case 1%left case
        if s2(2)~=1%Stability
            s2_next=s2-1;
        else 
            s2_next=s2;
        end
    case 2%right case
        if s2(2)~=max2%Stability
            s2_next=s2+1;
        else 
            s2_next=s2;
        end
    case 3
            s2_next=s2;
end

end

