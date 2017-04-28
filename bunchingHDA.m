




function outputCell=bunchingHDA(Target,Source,T_int_Target,T_int_Source,C_int,C_ref,alpha,outerLoopIter,innerLoopIter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Bunching.HDA methdod will be appear in..
%INPUTS:
%Target---> the target data of size Ntarget x dt + 1 (last col is label col)
%Source--->the source  dataof size Nsource x ds +1 (last col is label col)
%T_int_Target---> initial T^T matrix of size d_c x d_t
%T_int_Source--->initial T^S matrix of size d_c x d_s
%C_int---> inital C matrix of size d_c x K
% C_ref---> refence C matrix with size the same as C
% alpha---> regularization parameter
%outerLoopIter---> the number of ieteration of Bunching.HDA
%innerLoopIter---> the number of iteration in the gradient descents
%OUTPUTS:
%outputCell---> three cells where each corrosponds to one mapping matrix
% FIRAT ISMAILOGLU (firism@gmail.com)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputCell=cell(3,1);

targetX=Target(:,1:end-1);%feature matrix of the target data
[N_target,dt]=size(targetX);
targetY=Target(:,end);% labels of the target

sourceX=Source(:,1:end-1);%feature matrix of the source data
sourceY=Source(:,end);% labels of the source
[N_source,ds]=size(sourceX);

sourceClasses=unique(sourceY);
diffClassNumb=length(sourceClasses);
Y=eye(diffClassNumb);

beta=0.5; %parameter of the backtraing in gradient descent


%%%installing the initial mapping matrices
T_target=T_int_Target;
T_source=T_int_Source;
C=C_int;

dc=size(T_int_Target,1);% common space dimension



%storing the classes in the target and the source
classCellTarget=cell(diffClassNumb,1);% each cell stores one class members
classCellSource=cell(diffClassNumb,1);
for i=1:diffClassNumb
   classCellTarget{i}= targetX(find(targetY==i),:); %no labels included
   classCellSource{i}=sourceX(find(sourceY==i),:);
end

%starting Bunching.HDA iteraions
for outerIter=1:outerLoopIter

disp(outerIter);

 %IMPROVE-T^T
 
    for innerLoop=1:innerLoopIter
  
      t=1;
      W_target=zeros(dc,dt);
    
    % computing the gradient matrix for the target
   for k=1:dc %for each binary problem
        matrixK=zeros(N_target,2);
        for ii=1:N_target
            matrixK(ii,1)=sigmoid(-(C(k,:)*Y(:,targetY(ii))))*sigmoid((T_target(k,:)*targetX(ii,:)'));
            matrixK(ii,2)=sigmoid(C(k,:)*Y(:,targetY(ii)))*sigmoid(-(T_target(k,:)*targetX(ii,:)'));
        end
        for i=1:dt %for each dimension
          W_target(k,i)=targetX(:,i)'*matrixK(:,1)-targetX(:,i)'*matrixK(:,2);
        end
   end
 
    %Backtracking
    A=bunchingCostTarget(Target,(T_target-t*W_target),C,diffClassNumb);
    if isnan(A), A=Inf; end; % A=NaN occurs when the stepsize is too large
    B=bunchingCostTarget(Target,T_target,C,diffClassNumb);
    myalpha=1/4; % this can be anything in (0,0.5).
    while A >B-myalpha*t*(norm(W_target))^2
        t=beta*t;    
        T_target_trial=T_target-t*W_target;
        A=bunchingCostTarget(Target,T_target_trial,C,diffClassNumb);
        if isnan(A), A=Inf; end; % A=NaN occurs when the stepsize is too large
    end
  
    T_target=T_target-t*W_target; %updating T_target
 
    end %for inner loop
 
    %IMPROVE-T^S
 for innerLoop=1:innerLoopIter
  
    t=1;
    W_source=zeros(dc,ds);
     
    tic
   for k=1:dc 
        matrixK=zeros(N_source,2);
        for ii=1:N_source
            matrixK(ii,1)=sigmoid(-(C(k,:)*Y(:,sourceY(ii))))*sigmoid((T_source(k,:)*sourceX(ii,:)'));
            matrixK(ii,2)=sigmoid(C(k,:)*Y(:,sourceY(ii)))*sigmoid(-(T_source(k,:)*sourceX(ii,:)'));
        end
        for i=1:ds %for each dimension
          W_source(k,i)=sourceX(:,i)'*matrixK(:,1)-sourceX(:,i)'*matrixK(:,2);
        end
   end
   
     %%Backtracing
    A=bunchingCostSource(Source,(T_source-t*W_source),C,diffClassNumb);
    if isnan(A), A=Inf; end; % A=NaN occurs when the stepsize is too large
    B=bunchingCostSource(Source,T_source,C,diffClassNumb);
    myalpha=1/4; % this can be anything in (0,0.5).
    while A >B-myalpha*t*(norm(W_source))^2
        t=beta*t;    
        T_source_trial=T_source-t*W_source;
        A=bunchingCostSource(Source,T_source_trial,C,diffClassNumb);
        if isnan(A), A=Inf; end; % A=NaN occurs when the stepsize is too large
    end

    T_source=T_source-t*W_source;
  
        
end%end of source gradient descent
    %}
    
    
   %IMPROVE-C
 
    for i=1:dc %for dimension
           
        for j=1:diffClassNumb %class
            
            T_y=classCellTarget{j};%class j of the target data
            S_y=classCellSource{j};
            n_t=size(T_y,1);
            n_s=size(S_y,1);
            S1=sum(T_target(i,:)*T_y');
            S2=sum(T_source(i,:)*S_y');
            C(i,j)=(alpha*C_ref(i,j)+(S1+S2)/(n_t+n_s))/(1+alpha);
        end
    end
 
    
 
   


end %end of outer loop 
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
 
 
 outputCell{1}=T_target;
 outputCell{2}=T_source;
 outputCell{3}=C;
   
end %main function ends

function L1=bunchingCostTarget(Target,T_tar,C,classNumb)
%the cost caused by T_target

nTarget=size(Target,1);

classMatrix=eye(classNumb);

L1=0;
for i=1:nTarget
    y=classMatrix(:,Target(i,end));
    x=Target(i,1:end-1)';
    p1=sigmoid(C*y);
    p2=sigmoid(T_tar*x); 
    L1=L1+theKL(p1,p2);
end
end

function L2=bunchingCostSource(Source,T_sor,C,classNumb)
%the cost caused by T_tar

nSource=size(Source,1);

classMatrix=eye(classNumb);

L2=0;

for i=1:nSource
    y=classMatrix(:,Source(i,end));
    x=Source(i,1:end-1)';
    p1=sigmoid(C*y);
    p2=sigmoid(T_sor*x); 
    L2=L2+theKL(p1,p2);
end
end
































        