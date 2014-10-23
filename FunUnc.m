function y = FunUnc(X)
   global tmp;
   y = sum((tmp(:,6:9)*X'-tmp(:,5)).^2);
end