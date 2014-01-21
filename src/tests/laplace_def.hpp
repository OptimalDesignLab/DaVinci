/**
 * \file laplace_def.hpp
 * \brief definition of Laplace evaluator methods
 * \author Jason Hicken <jason.hicken@gmail.com>
 */

#include <boost/assert.hpp>
#include "Teuchos_Tuple.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

namespace davinci {
//==============================================================================
template <typename NodeT, typename ScalarT>
Laplace<NodeT,ScalarT>::Laplace() {

}
//==============================================================================
template <typename NodeT, typename ScalarT>
void Laplace<NodeT,ScalarT>::MemoryRequired(
    int& mesh_offset, map<string,int>& mesh_map_offset,
    int& soln_offset, map<string,int>& soln_map_offset,
    int& resid_offset, map<string,int>& resid_map_offset) const {
  BOOST_ASSERT_MSG(mesh_offset >= 0, "mesh_offset must be >= 0");
  BOOST_ASSERT_MSG(soln_offset >= 0, "soln_offset must be >= 0");
  BOOST_ASSERT_MSG(resid_offset >= 0, "resid_offset must be >= 0");
  mesh_map_offset["jacob_cub"] = mesh_offset;
  mesh_offset += num_elems_*num_cub_points_;
  mesh_map_offset["grads_transformed"] = mesh_offset;
  mesh_offset += num_elems_*num_ref_basis_*num_cub_points_*dim_;
  mesh_map_offset["grads_transformed_weighted"] = mesh_offset;
  mesh_offset += num_elems_*num_ref_basis_*num_cub_points_*dim_;
  resid_map_offset["solution_grad"] = resid_offset;
  resid_offset += num_elems_*num_cub_points_*dim_;
  resid_map_offset["residual"] = resid_offset;
  resid_offset += num_elems_*num_ref_basis_;
}
//==============================================================================
template <typename NodeT, typename ScalarT>
void Laplace<NodeT,ScalarT>::SetDataViews(
    ArrayRCP<NodeT>& mesh_data, map<string,int>& mesh_map_offset,
    ArrayRCP<ScalarT>& soln_data, map<string,int>& soln_map_offset,
    ArrayRCP<ResidT>& resid_data, map<string,int>& resid_map_offset) {
  using Teuchos::tuple;
  // views of inputs
  jacob_inv_ =
      GenerateConstView(mesh_data, mesh_map_offset.at("jacob_inv"),
                        tuple(num_elems_, num_cub_points_, dim_, dim_));
  jacob_det_ =
      GenerateConstView(mesh_data, mesh_map_offset.at("jacob_det"),
                        tuple(num_elems_, num_cub_points_));
  solution_coeff_ =
      GenerateConstView(soln_data, soln_map_offset.at("solution_coeff"),
                        tuple(num_elems_, num_ref_basis_));
  // views of outputs
  jacob_cub_ =
      GenerateView(mesh_data, mesh_map_offset.at("jacob_cub"),
                   tuple(num_elems_, num_cub_points_));
  grads_transformed_ =
      GenerateView(mesh_data, mesh_map_offset.at("grads_transformed"),
                   tuple(num_elems_, num_ref_basis_, num_cub_points_, dim_));
  grads_transformed_weighted_ =
      GenerateView(mesh_data, mesh_map_offset.at("grads_transformed_weighted"),
                   tuple(num_elems_, num_ref_basis_, num_cub_points_, dim_));
  solution_grad_ =
      GenerateView(resid_data, resid_map_offset.at("solution_grad"),
                   tuple(num_elems_, num_cub_points_, dim_));
  residual_ =
      GenerateView(resid_data, resid_map_offset.at("residual"),
                   tuple(num_elems_, num_ref_basis_));
}
//==============================================================================
template <typename NodeT, typename ScalarT>
void Laplace<NodeT,ScalarT>::Evaluate(
    const RCP<CellTopology>& topology,
    const FieldContainer<double>& cub_points,
    const FieldContainer<double>& cub_weights,
    const FieldContainer<double>& basis_vals,
    const FieldContainer<double>& basis_grads) {
  typedef Intrepid::FunctionSpaceTools FST;
  // transform to physical coordinates
  FST::HGRADtransformGRAD<NodeT>(*grads_transformed_, *jacob_inv_, basis_grads);
  // compute weighted measure
  FST::computeCellMeasure<NodeT>(*jacob_cub_, *jacob_det_, cub_weights);
  // multiply values with weighted measure
  FST::multiplyMeasure<NodeT>(*grads_transformed_weighted_,
                              *jacob_cub_, *grads_transformed_);
  // compute solution gradient
  solution_grad_->initialize(); // set solution gradient to zero
  FST::evaluate<ResidT>(*solution_grad_, *solution_coeff_, *grads_transformed_);
  for (int i = 0; i < num_elems_; ++i)
    for (int j = 0; j < num_cub_points_; ++j)
      for (int k = 0; k < dim_; k++)
        std::cout << (*solution_grad_)(i,j,k) << " ";
  // integrate to get residual
  FST::integrate<ResidT>(*residual_, *solution_grad_,
                         *grads_transformed_weighted_,
                         Intrepid::COMP_BLAS);
}
//==============================================================================
}
