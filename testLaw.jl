using LinearAlgebra
using Random

n = 10
A = 1/n * ones(n,n)

for i in 1:n
    A[i,i] = 1
end

println(rank(A))