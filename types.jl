abstract type ClassificationModel end

struct AdvTreeModel <: ClassificationModel
  theta_node::Matrix{Float64}
  theta_edge::Array{Float64,3}
end

struct CRFTreeModel <: ClassificationModel
  theta_node::Matrix{Float64}
  theta_edge::Array{Float64,3}
end