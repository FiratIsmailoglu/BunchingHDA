

 function accVector=classification_with_bunching(testX,trainX,trainY,testY,sourceX,sourceY,alpha,outerLoop,innerLoop)
%This first calls for bunchingHDA.m then performs classsifcation in the
%common space
%Inputs: testX: nTest x dTarget feature space (without labels)for test
%testY labels of the tested instances
%trainX: nTrainXdTarget feature space for training target data
%trainY labels of target training instances
%sourceX nSourcex dSource feature space of source instances
% alpha reg. parameter of Bunching.HDA
%outerloop and innerLoop iteration number for Bunching and the gradient
%descents 
%Outputs: two accuracies: first one is for bunching.HDA.NN the second bunching.HDA.Pr

accVector=zeros(2,1); %foutput initilization



nTest=size(testX,1);
dTarget=size(trainX,2)+1;  % +1 was added to avoid scaling
dSource=size(sourceX,2)+1; % +1 was added to avoid scaling

%%%%%%%%%%%%% normalization%%%%%%%
targetX=[testX;trainX];
targetX=[ones(size(targetX,1),1),targetX]; % dummy feature added to avoid scaling and data normalization

for i=1:dTarget
 theMin=min(targetX(:,i));
 theRange=range(targetX(:,i));
 if theRange~=0
 targetX(:,i)=(targetX(:,i)-theMin)/theRange;
 else
     targetX(:,i)=0.5;
 end
end

targetX=targetX./repmat(sum(targetX,2),1,dTarget);

testX=targetX(1:nTest,:);
trainX=targetX(nTest+1:end,:);

sourceX=[ones(size(sourceX,1),1),sourceX]; % dummy feature added to avoid scaling and data normalization


for i=1:dSource
 theMin=min(sourceX(:,i));
 theRange=range(sourceX(:,i));
 if theRange~=0
sourceX(:,i)=(sourceX(:,i)-theMin)/theRange;
 else
     sourceX(:,i)=0.5;
 end
end

sourceX=sourceX./repmat(sum(sourceX,2),1,dSource);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Target=[trainX trainY];
Source=[sourceX sourceY];

diffClassNumb=length(unique(sourceY)); 
codeMatrix=eye(diffClassNumb)'; %one can change the code matrix, e.g. exhaustive code matrix can be used
C_ref=2*codeMatrix-1; % to replace all zeros by minus one


n=size(codeMatrix,1);


%%%initial mapping matrices%%%%%
T_int_Target=rand(n,dTarget);
T_int_Source=rand(n,dSource);
C_int=C_ref;


outputCell=bunchingHDA(Target,Source,T_int_Target,T_int_Source,C_int,C_ref,alpha,outerLoop,innerLoop);

T_target=outputCell{1};
T_source=outputCell{2};
C=outputCell{3};
 
predicts=zeros(nTest,2); %the firs colomun is for the predictions of bunching.HDA.NN and the seconfd column is for Bunching.HDA.pR

target_inCommonSpace= sigmoid(trainX*T_target');
source_inCommonSpace= sigmoid(sourceX*T_source');

commonSpace=[[target_inCommonSpace trainY];[source_inCommonSpace sourceY]];
commonX=commonSpace(:,1:end-1);
commonY=commonSpace(:,end);
nCommon=size(commonX,1);


for i=1:nTest
    test_instance=sigmoid(testX(i,:)*T_target');
    
        distVector=zeros(nCommon,1);
    for j=1:nCommon
     distVector(j)= theKL(test_instance,commonX(j,:));
    end
   [~,ind]=min(distVector);
    predicts(i,1)=commonY(ind(1));
    
    distVector=zeros(diffClassNumb,1);
    
    for j=1:diffClassNumb
distVector(j)=KLdiv(test_instance,sigmoid(C(:,j)'));
    end
   [~,ind]=min(distVector);
    predicts(i,2)=ind;

   
end


accVector(1,1)=((sum(predicts(:,1)==testY))/nTest)*100;%NN acc
accVector(2,1)=((sum(predicts(:,2)==testY))/nTest)*100;%Pr acc






