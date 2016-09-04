clear variables
close all
clc

defense_game = [
    1 1 1 1
    0 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0
    ];


gamma = .7;
beta = 1;
alpha = .5;
n_episodes = 2500;
m = 5;
n = 4;
n_actions_1=4;
n_actions_2=3;
plotting_mode = 2;
clim = [-40 20];
beta_factor = .999;

% initialize Q function for the states and actions of both agents
Q = round(10*rand(n*m,n,n_actions_1,n_actions_2))/10000;
pi1 = zeros(n*m,n,n_actions_1,n_actions_2))/10000;
% initial state distributions as anonymous functions
ag1_init = @() [5,randi([1,4])];
ag2_init = @() randi([1,4]);

% the actions of the runner agent as [n_actions_1 x 2] matrix
ag1_actions = [[1,0];[0,-1];[0,1];[0,0]];
    
% the actions of the catcher agent as [n_actions_2 x 2] matrix
ag2_actions = [[0,-1];[0,1];[0,0]];

% transition probabilities for the runner (first is the selected action,
% the rest are the other actions)
P1 = [.91 .03 .03 .03];
% transition probabilities for the catcher
P2 = [.9 .05 .05];

a1_stats=zeros(n*m,n,n_actions_1);
a2_stats=zeros(n*m,n,n_actions_2);
% for each episode
for i = 1:n_episodes
    % initialize states
    s1_t = ag1_init();
    s2_t = ag2_init();

    % initialize reward
    r = 0;
    t = 0;
    
    % any state that gives a reward is a terminal state or stop after 30
    % steps
    while(r == 0 && t < 30)
        t = t + 1;
        r = marl.reward(s1_t,s2_t,defense_game);
        if r == 0
            % Q table for the current state. You might want to have a look at
            % the definition of the function squeeze here.
            %VMM=maxmin(pi1)
            %vkplus1ix = ndi2lin([ones(1, length(action)) state'], a.sizes.q);
            Qt = squeeze(Q(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,:,:)); %Not sure about this one, this gets the stage game but i am not sure if my interpretation is correct
            % value and best policy of the current state for the runner
            [V1,V1ind] = max(Qt);
            % do random steps with prob epsilon
            if rand < beta
                pi1 = datasample(ag1_actions,n_actions_2);%random samples
            else
                pi1 = ag1_actions(V1ind,:);%Obtains the actions that maximize the v function
                % normalize policy
                pi1 = bsxfun(@rdivide,abs(pi1),sum(abs(pi1)));
                pi1(isnan(pi1))=0;
            end
            
            Qt2 = -Qt';
            % value of the current state for the catcher
            [V2,V2ind] = max(Qt2);
            
            % do random steps with prob epsilon
            if rand < beta
                pi2 = datasample(ag2_actions,n_actions_1);%random samples
            else
                pi2 = ag2_actions(V2ind,:);%Obtains the actions that maximize the v function
                % normalize policy
                pi2 = bsxfun(@rdivide,abs(pi2),sum(abs(pi2)));
                pi2(isnan(pi2))=0;
            end
            
            % select the actions according to the policies
            [~,ActInd1]=max(a1_stats(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,:),[],3);
            [~,ActInd2]=max(a2_stats(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,:),[],3);
            
            a1_t_p = pi1(ActInd2,:);%statistics estimation. not sure if correct either
            a2_t_p = pi2(ActInd1,:);
            
            % sample from the transition probability distribution

            a1_noise=cat(1,a1_t_p,ag1_actions(~ismember(ag1_actions,a1_t_p,'rows'),:));%Puts the selected action from the policy as the 1st value
            a2_noise=cat(1,a2_t_p,ag2_actions(~ismember(ag2_actions,a2_t_p,'rows'),:));
            
            % get the actual actions
            a1_t = datasample(a1_noise,1,'Weights',P1);%Samples with the appropiate weights
            a2_t = datasample(a2_noise,1,'Weights',P2);
            [~,ActTaken1]=ismember(ag1_actions,a1_t,'rows');%Finds action index
            [~,ActTaken2]=ismember(ag2_actions,a2_t,'rows');
            a1_stats(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,find(ActTaken1))=a1_stats(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,find(ActTaken1))+1;%increases history w.r.t state
            a2_stats(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,find(ActTaken2))=a2_stats(sub2ind([m,n],s1_t(1),s1_t(2)),s2_t,find(ActTaken2))+1;
            % observe the reward
            
            % transition if not in a terminal state
            [s1_next, s2_next] = marl.transition(s1_t,s2_t,find(ActTaken1),find(ActTaken2),m,n);
        else
            s1_next = s1_t;
            s2_next = s2_t;
        end

        
        % plotting
        if plotting_mode == 1
            figure(1); 
                imagesc(defense_game);
                hold on;
                [s1_sub(1), s1_sub(2)] = ind2sub([5,4],s1_t);
                plot(s1_sub(2),s1_sub(1),'*r');
                plot(s2_t,2,'or');
                hold off;
            figure(2);
                [V, ~] = max(min(Q,[],4),[],3);
                for j = 1:4;
                    subplot(1,4,j);
                    imagesc(reshape(V(:,j),5,4),clim);
                end;
            figure(3);
                [V, ~] = max(min(-Q,[],3),[],4);
                for j = 1:20;
                    [xp, yp] = ind2sub([5,4],j);
                    j_ = sub2ind([4,5],yp,xp);
                    subplot(5,4,j_);
                    imagesc(V(j,:),clim);
                end;
            drawnow;
        end
        
        % compute V_mm
        V_mm = maxmin(pi1(find(ActTaken1))*Q(sub2ind([m n],s1_next(1),s1_next(2)),s2_next,find(ActTaken1),find(ActTaken2)));%Missing
         
        % update state action value function
        Q(s1_t,s2_t,a1_t_p,a2_t_p) =(1-a)*Qt + alpha * nxt_reward1 + gamma* V_mm;%Update function
        
        % update current state
        s1_t = s1_next;
        s2_t = s2_next;
    end
    
    % plotting
    if plotting_mode == 2
        figure(2);
            [V, ~] = max(min(Q,[],4),[],3);
            for j = 1:4;
                subplot(5,8,[4 12 20 28 36]+j);
                imagesc(reshape(V(:,j),5,4),clim);
            end;
        figure(2);
            [V, ~] = max(min(-Q,[],3),[],4);
            for j = 1:20;
                [xp, yp] = ind2sub([5,8],j);
                j_ = sub2ind([8,5],yp,xp);
                subplot(5,8,j_);
                imagesc(V(j,:),clim);
            end;
        drawnow;
    end
    
    % update beta
	beta = beta_factor * beta;
    
    fprintf('%d - %f\n',i,beta);
end
 
figure(2);
    [V, ~] = max(min(Q,[],4),[],3);
    for j = 1:4;
        subplot(5,8,[4 12 20 28 36]+j);
        imagesc(reshape(V(:,j),5,4),clim);
    end;
figure(2);
    [V, ~] = max(min(-Q,[],3),[],4);
    for j = 1:20;
        [xp, yp] = ind2sub([5,8],j);
        j_ = sub2ind([8,5],yp,xp);
        subplot(5,8,j_);
        imagesc(V(j,:),clim);
    end;