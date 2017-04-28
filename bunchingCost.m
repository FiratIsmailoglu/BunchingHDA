

function cost=bunchingCost(Target,Source,T_tar,T_sor,C,Cref,alpha,classNumb)


nTarget=size(Target,1);
nSource=size(Source,1);
classMatrix=eye(classNumb);

L1=0;
for i=1:nTarget
    y=classMatrix(:,Target(i,end));
    x=Target(i,1:end-1)';
    p1=sigmoid(C*y);
    p2=sigmoid(T_tar*x); 
    L1=L1+KLdiv(p1,p2);
end

L2=0;

for i=1:nSource
    y=classMatrix(:,Source(i,end));
    x=Source(i,1:end-1)';
    p1=sigmoid(C*y);
    p2=sigmoid(T_sor*x); 
    L2=L2+KLdiv(p1,p2);
end

R1=0;
for i=1:nTarget
     y=classMatrix(:,Target(i,end));
     p1=sigmoid(C*y);
     p2=sigmoid(Cref*y);
     R1=R1+KLdiv(p1,p2);
end
 
R2=0;
for i=1:nSource
     y=classMatrix(:,Source(i,end));
     p1=sigmoid(C*y);
     p2=sigmoid(Cref*y);
     R2=R2+KLdiv(p1,p2);
end

cost=L1+L2+alpha*(R1+R2);
    
end 
    
    
