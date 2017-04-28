

  function result=KLdiv(p1,p2)

 
  
s=length(p1);
t=0;
for i=1:s

    t=t+p1(i)*log(p1(i)/p2(i))+(1-p1(i))*log((1-p1(i))/(1-p2(i)));
end
result=t;
end
