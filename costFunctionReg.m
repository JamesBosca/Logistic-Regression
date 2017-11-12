function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z =size(X);

w=X*theta;

h=sigmoid(w);
L=0;

 for j=2:z(2)
L=L+(lambda/(2*m))*(theta(j))^2;
 end
 
for i=1:m
      
J=J+(1/m)*(-y(i)*log(h(i))-(1-y(i))*log(1-h(i)));
 
end

J=J+L;


temp=theta;
temp(1)=0;
del= (lambda/m)*(temp);
grad=(1/m)*transpose(X)*(h-y);

grad=grad+del;






% =============================================================

    
end
