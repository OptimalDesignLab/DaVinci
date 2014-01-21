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
  jacob_inv_ = Teuchos::rcp(new FieldContainer<NodeT>(
      Teuchos::tuple(num_elems_, num_cub_points_, dim_, dim_),
      mesh_data.persistingView(mesh_map_offset.at("jacob_inv"),
                               num_elems_*num_cub_points_*dim_*dim_)) );
  jacob_det_ = Teuchos::rcp(new FieldContainer<NodeT>(
      Teuchos::tuple(num_elems_, num_cub_points_),
      mesh_data.persistingView(mesh_map_offset.at("jacob_det"),
                               num_elems_*num_cub_points_)) );
  jacob_cub_ = Teuchos::rcp(new FieldContainer<NodeT>(
      Teuchos::tuple(num_elems_, num_cub_points_),
      mesh_data.persistingView(mesh_map_offset.at("jacob_cub"),
                               num_elems_*num_cub_points_)) );
  grads_transformed_ = Teuchos::rcp(new FieldContainer<NodeT>(
      Teuchos::tuple(num_elems_, num_ref_basis_, cub_points_, dim_),
      mesh_data.persistingView(
          mesh_map_offset.at("grads_transformed"),
          num_elems_*num_ref_basis_*num_cub_points_*dim_)) );
  grads_transformed_weighted_ = Teuchos::rcp(new FieldContainer<NodeT>(
      Teuchos::tuple(num_elems_, num_ref_basis_, cub_points_, dim_),
      mesh_data.persistingView(
          mesh_map_offset.at("grads_transformed_weighted"),
          num_elems_*num_ref_basis_*num_cub_points_*dim_)) );
  solution_grad_ = Teuchos::rcp(new FieldContainer<ResidT>(
      Teuchos::tuple(num_elems_, cub_points_, dim_),
      resid_data.persistingView(resid_map_offset.at("solution_grad"),
                                num_elems_*num_cub_points_*dim_)) );
  residual_ = Teuchos::rcp(new FieldContainer<ResidT>(
      Teuchos::tuple(num_elems_, num_ref_basis_),
      resid_data.persistingView(resid_map_offset.at("residual"),
                               num_elems_*num_ref_basis_)) );
  solution_coeff_ = Teuchos::rcp(new FieldContainer<ScalarT>(
      Teuchos::tuple(num_elems_, num_ref_basis_),
      soln_data.persistingView(soln_map_offset.at("solution_coeff"),
                               num_elems_*num_ref_basis_)) );
#if 0
  solution_coeff_ = GenerateView(soln_map_offset.at("solution_coeff"), soln_data,
                                 Teuchos::tuple(num_elems_, num_ref_basis_));
#endif
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
  FST::HGRADtransformGRAD<NodeT>(grads_transformed_, jacob_inv_, basis_grads);
  // compute weighted measure
  FST::computeCellMeasure<NodeT>(weighted_measure_, jacob_det_, cub_weights);
  // multiply values with weighted measure
  FST::multiplyMeasure<NodeT>(grads_transformed_weighted_,
                              weighted_measure_, grads_transformed_);
  // compute solution gradient
  solution_grad_.initialize();
  FST::evaluate<ResidT>(solution_grad_, solution_coeff_, grads_transformed_);
  // integate to get residual
  FST::integrate<ResidT>(residual_, solution_grad_, grads_transformed_weighted_,
                         COMP_BLAS); // INTREPID_...
}
//==============================================================================
}
